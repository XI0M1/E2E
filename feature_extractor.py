"""
工作负载特征提取器 (Workload Feature Extractor)
功能：从离线采样数据中动态提取工作负载特征向量

设计理念：
1. 特征不是预定义的，而是从实际执行统计中学习得出
2. 每个workload的特征基于它在50个参数配置下的平均表现
3. 特征反映workload对参数的敏感性和执行特性

工作流程：
Phase 1 (采样) → 生成 offline_sample.jsonl (650行) 
                    ↓
                (新增) feature_extractor.py
                    ↓
                分析离线采样数据
                    ↓
                生成 SuperWG/feature/tpch.json
                    ↓
                Phase 2 使用该特征
"""

import json
import numpy as np
from typing import Dict, List
from collections import defaultdict
import os


class WorkloadFeatureExtractor:
    """
    从离线采样数据中提取工作负载特征向量
    """
    
    def __init__(self, offline_sample_path: str, database: str):
        """
        初始化特征提取器
        
        参数：
            offline_sample_path: offline_sample.jsonl 的路径
            database: 数据库名称 (用于输出文件路径)
        """
        self.offline_sample_path = offline_sample_path
        self.database = database
        self.features = {}  # {workload_id: [30维特征向量]}
        self.workload_data = defaultdict(list)  # {workload_id: [所有样本列表]}
    
    def load_samples(self) -> bool:
        """
        加载离线采样数据
        
        从 offline_sample.jsonl 读取所有样本并按workload分组
        """
        try:
            with open(self.offline_sample_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    workload = sample['workload']
                    self.workload_data[workload].append(sample)
            
            print(f"✓ 加载了 {len(self.workload_data)} 个workload的采样数据")
            for wl, samples in self.workload_data.items():
                print(f"  {wl}: {len(samples)} 个样本")
            
            return True
        except FileNotFoundError:
            print(f"✗ 离线采样文件不存在: {self.offline_sample_path}")
            return False
        except Exception as e:
            print(f"✗ 加载采样数据失败: {e}")
            return False
    
    def extract_features(self) -> Dict[str, List[float]]:
        """
        为每个workload提取30维特征向量
        
        过程：
        1. 对每个workload的所有样本统计
        2. 计算30个特征维度
        3. 保存到self.features
        
        返回：
            {workload_id: [30维特征]}
        """
        print("\n=== 提取工作负载特征 ===\n")
        
        for workload_id, samples in self.workload_data.items():
            print(f"处理: {workload_id} ({len(samples)}个样本)")
            
            # 创建30维特征向量
            feature = [0.0] * 30
            
            # [0-4] 主要参数的平均设置值
            sb_values = [s['config'].get('shared_buffers', 131072) for s in samples]
            feature[0] = min(np.mean(sb_values) / 2000000, 1.0)
            
            wm_values = [s['config'].get('work_mem', 4096) for s in samples]
            feature[1] = min(np.mean(wm_values) / 1000000, 1.0)
            
            ecs_values = [s['config'].get('effective_cache_size', 524288) for s in samples]
            feature[2] = min(np.mean(ecs_values) / 2000000, 1.0)
            
            mwm_values = [s['config'].get('maintenance_work_mem', 65536) for s in samples]
            feature[3] = min(np.mean(mwm_values) / 1000000, 1.0)
            
            cct_values = [s['config'].get('checkpoint_completion_target', 0.5) for s in samples]
            feature[4] = np.mean(cct_values)
            
            # [5-15] 参数的敏感度
            param_variance = self._compute_parameter_sensitivity(samples)
            for i, (param, variance) in enumerate(sorted(param_variance.items())[:11]):
                feature[5 + i] = min(variance, 1.0)
            
            # [16-25] 执行特性
            tps_values = [s['tps'] for s in samples]
            tps_mean = np.mean(tps_values)
            tps_std = np.std(tps_values)
            feature[16] = tps_std / tps_mean if tps_mean > 0 else 0
            
            tps_min = np.min(tps_values)
            tps_max = np.max(tps_values)
            feature[17] = (tps_max - tps_min) / tps_min if tps_min > 0 else 0
            
            cache_ratios = [s['inner_metrics'].get('cache_hit_ratio', 0.5) for s in samples]
            feature[18] = np.mean(cache_ratios) if cache_ratios else 0.5
            
            cache_std = np.std(cache_ratios) if len(cache_ratios) > 1 else 0
            cache_mean = np.mean(cache_ratios)
            feature[19] = cache_std / cache_mean if cache_mean > 0 else 0
            
            cpu_usages = [s['inner_metrics'].get('cpu_usage', 50) for s in samples]
            feature[20] = np.mean(cpu_usages) / 100 if cpu_usages else 0.5
            
            active_conns = [s['inner_metrics'].get('active_connections', 5) for s in samples]
            feature[21] = min(np.mean(active_conns) / 100, 1.0)
            
            db_sizes = [s['inner_metrics'].get('database_size', 5*10**9) for s in samples]
            feature[22] = min(np.mean(db_sizes) / (100 * 10**9), 1.0)
            
            feature[23] = 1.0 / (1.0 + feature[16])
            feature[24] = 0.5
            feature[25] = self._compute_sensitivity_score(samples, feature[16], feature[19])
            
            # [26-29] 额外统计
            feature[26] = np.std(cpu_usages) / 100 if len(cpu_usages) > 1 else 0
            feature[27] = np.mean([s['inner_metrics'].get('latency_ms', 5) 
                                  for s in samples if isinstance(s['inner_metrics'].get('latency_ms'), (int, float))]) / 100 if samples else 0
            feature[28] = np.std([s['inner_metrics'].get('latency_ms', 5) 
                                 for s in samples if isinstance(s['inner_metrics'].get('latency_ms'), (int, float))]) / 100 if samples else 0
            feature[29] = float(len(samples)) / 50
            
            feature = [max(0, min(1, f)) for f in feature]
            
            self.features[workload_id] = feature
            print(f"  ✓ 特征维度: {len(feature)}, 值范围: [{min(feature):.3f}, {max(feature):.3f}]")
        
        return self.features
    
    def _compute_parameter_sensitivity(self, samples: List[Dict]) -> Dict[str, float]:
        """计算每个参数的敏感度"""
        param_tps_corr = {}
        all_params = set()
        for s in samples:
            all_params.update(s['config'].keys())
        
        tps_values = np.array([s['tps'] for s in samples])
        
        for param in list(all_params)[:20]:  # 只取前20个参数
            param_values = []
            for s in samples:
                pv = s['config'].get(param, 0)
                if isinstance(pv, (int, float)):
                    param_values.append(float(pv))
                else:
                    param_values.append(0)
            
            if len(param_values) == 0:
                continue
            
            param_values = np.array(param_values)
            
            if np.std(param_values) > 0 and np.std(tps_values) > 0:
                corr = np.abs(np.corrcoef(param_values, tps_values)[0, 1])
                if not np.isnan(corr):
                    param_tps_corr[param] = corr
        
        return param_tps_corr
    
    def _compute_sensitivity_score(self, samples: List[Dict], 
                                   tps_variance: float, cache_variance: float) -> float:
        """计算综合的参数敏感度评分"""
        sensitivity = (tps_variance * 0.6 + cache_variance * 0.4)
        return min(sensitivity, 1.0)
    
    def save_features(self, output_dir: str = 'SuperWG/feature') -> bool:
        """保存提取的特征向量到JSON文件"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f'{self.database}.json')
            
            with open(output_path, 'w') as f:
                json.dump(self.features, f, indent=2)
            
            print(f"\n✓ 特征向量已保存到: {output_path}")
            print(f"  包含 {len(self.features)} 个workload的特征向量")
            
            return True
            
        except Exception as e:
            print(f"✗ 保存特征向量失败: {e}")
            return False
    
    def extract_and_save(self) -> bool:
        """完整的特征提取流程"""
        print("\n" + "="*70)
        print("工作负载特征向量提取")
        print("="*70)
        
        if not self.load_samples():
            return False
        
        self.extract_features()
        
        if not self.save_features():
            return False
        
        print("\n✓ 特征提取完成！\n")
        return True


def extract_workload_features(offline_sample_path: str, database: str) -> bool:
    """便捷函数：一键提取workload特征"""
    extractor = WorkloadFeatureExtractor(offline_sample_path, database)
    return extractor.extract_and_save()


if __name__ == '__main__':
    # 示例用法
    extract_workload_features(
        offline_sample_path='offline_sample/offline_sample_tpch.jsonl',
        database='tpch'
    )
