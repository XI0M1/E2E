"""
SFT训练数据构建器 (Training Data Builder)
功能：从离线采样数据构建LLaMA-Factory/Hugging Face SFT格式的训练数据

工作流程：
offline_sample.jsonl (650行)
    ↓
1. 筛选高质量样本 (200-300条)
   - 对每个workload，筛选TPS最高的几个配置 + 少量次优项
2. 提取Workload Statistics（SQL统计仰自workload_file）
3. 获取Query Plans（已保存在样本中）
4. 整理Inner Metrics（系统运行时指标）
5. 组装成SFT格式
    ↓
training_sft_data.jsonl
    {
      "instruction": "系统提示...",
      "input": "Workload Statistics + Query Plans + Inner Metrics",
      "output": "最优参数配置（JSON格式）"
    }
    ↓
Phase 2 LLM微调
"""

import json
import re
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np


class TrainingDataBuilder:
    """
    构建SFT训练数据
    """
    
    def __init__(self, offline_sample_path: str, output_path: str):
        """
        初始化训练数据构建器
        
        参数：
            offline_sample_path: offline_sample.jsonl 的路径
            output_path: 输出training_sft_data.jsonl的路径
        """
        self.offline_sample_path = offline_sample_path
        self.output_path = output_path
        self.samples = []
        self.training_data = []
        self.workload_search_dirs = self._build_workload_search_dirs()

    def _build_workload_search_dirs(self) -> List[str]:
        """构建 workload 搜索目录，兼容 data/ 与旧目录结构。"""
        env_dirs = os.environ.get('WORKLOAD_SEARCH_DIRS', '')
        dirs = [d for d in env_dirs.split(os.pathsep) if d]
        dirs.extend([
            'data',
            os.path.join('data', 'olap'),
            os.path.join('data', 'oltp'),
            'olap_workloads',
            'oltp_workloads',
            '.',
        ])

        seen = set()
        results = []
        for directory in dirs:
            normalized = os.path.normpath(directory)
            if normalized not in seen:
                seen.add(normalized)
                results.append(directory)
        return results

    def resolve_workload_path(self, sample: Dict) -> str:
        """优先使用样本中的真实路径，否则按文件名回退搜索。"""
        workload = sample.get('workload', '')
        workload_file = sample.get('workload_file', '')

        for candidate in [workload, workload_file]:
            if candidate and os.path.exists(candidate):
                return candidate

        basename = os.path.basename(workload_file or workload)
        if not basename:
            return ''

        for directory in self.workload_search_dirs:
            candidate = os.path.join(directory, basename)
            if os.path.exists(candidate):
                return candidate

        return ''
    
    def load_samples(self) -> bool:
        """加载离线采样数据"""
        try:
            with open(self.offline_sample_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    self.samples.append(sample)
            
            print(f"✓ 加载了 {len(self.samples)} 个采样数据")
            return True
        except FileNotFoundError:
            print(f"✗ 离线采样文件不存在: {self.offline_sample_path}")
            return False
        except Exception as e:
            print(f"✗ 加载采样数据失败: {e}")
            return False
    
    def select_high_quality_samples(self, min_count: int = 200, max_count: int = 300) -> List[Dict]:
        """
        筛选高质量样本
        
        策略：
        - 按workload分组
        - 对每个workload，排序后取：
          * 前 top_k 个TPS最高的（最优样本）
          * 随机选择一些次优样本（多样性）
        
        参数：
            min_count: 最少保留样本数
            max_count: 最多保留样本数
        
        返回：
            list: 筛选后的样本列表
        """
        print("\n=== 筛选高质量样本 ===\n")
        
        # 按workload分组
        workload_samples = defaultdict(list)
        for sample in self.samples:
            workload = sample['workload']
            workload_samples[workload].append(sample)
        
        selected = []
        
        for workload, samples in workload_samples.items():
            # 按TPS排序（降序）
            sorted_samples = sorted(samples, key=lambda s: s['tps'], reverse=True)
            
            # 取前1/3作为顶级样本
            top_count = max(1, len(sorted_samples) // 3)
            selected.extend(sorted_samples[:top_count])
            
            # 从剩余的中随机选择1/6
            remaining = sorted_samples[top_count:]
            secondary_count = max(1, len(remaining) // 6)
            secondary = np.random.choice(
                remaining, 
                size=min(secondary_count, len(remaining)), 
                replace=False
            ).tolist()
            selected.extend(secondary)
            
            print(f"  {workload}: {len(sorted_samples)} → {len(selected) - sum(1 for s in selected[:-len(selected) if len(selected) > len(samples) else 0])} (Top {top_count} + {len(secondary)} secondary)")
        
        # 最终确保在范围内
        if len(selected) > max_count:
            # 按TPS排序后取前max_count个
            selected = sorted(selected, key=lambda s: s['tps'], reverse=True)[:max_count]
        
        print(f"\n✓ 最终筛选样本数: {len(selected)}/{len(self.samples)}")
        return selected
    
    def extract_workload_statistics(self, workload_file: str) -> Dict[str, any]:
        """
        从workload文件提取SQL统计信息
        
        返回：
            dict: {
                'total_sql': int,
                'read_write_ratio': str,
                'order_by_percent': float,
                'group_by_percent': float,
                'join_count': int,
                'agg_functions': dict,
                'table_frequency': dict
            }
        """
        stats = {
            'total_sql': 0,
            'read_ratio': 0,
            'write_ratio': 0,
            'order_by_percent': 0,
            'group_by_percent': 0,
            'join_count': 0,
            'aggregation_count': 0,
            'table_count': 0
        }
        
        try:
            if not os.path.exists(workload_file):
                return stats
            
            with open(workload_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().upper()
            
            # 移除注释
            lines = content.split('\n')
            cleaned_lines = [
                line for line in lines 
                if not line.strip().startswith('--') and 
                   not line.strip().startswith('#')
            ]
            content = '\n'.join(cleaned_lines)
            
            # 按分号分割SQL
            sqls = content.split(';')
            sqls = [s.strip() for s in sqls if s.strip()]
            
            stats['total_sql'] = len(sqls)
            
            # 统计读写
            select_count = sum(1 for s in sqls if s.startswith('SELECT'))
            write_count = sum(1 for s in sqls if any(s.startswith(op) for op in ['INSERT', 'UPDATE', 'DELETE']))
            total = select_count + write_count if write_count > 0 else len(sqls)
            
            stats['read_ratio'] = round(select_count / total * 100, 1) if total > 0 else 100
            stats['write_ratio'] = round(write_count / total * 100, 1) if total > 0 else 0
            
            # 统计ORDER BY
            order_by_count = sum(1 for s in sqls if 'ORDER BY' in s)
            stats['order_by_percent'] = round(order_by_count / len(sqls) * 100, 1) if sqls else 0
            
            # 统计GROUP BY
            group_by_count = sum(1 for s in sqls if 'GROUP BY' in s)
            stats['group_by_percent'] = round(group_by_count / len(sqls) * 100, 1) if sqls else 0
            
            # 统计JOIN
            stats['join_count'] = sum(s.count('JOIN') for s in sqls)
            
            # 统计聚合函数
            agg_pattern = r'(COUNT|SUM|AVG|MAX|MIN|GROUP_CONCAT)\s*\('
            stats['aggregation_count'] = sum(len(re.findall(agg_pattern, s)) for s in sqls)
            
            # 统计表个数
            table_pattern = r'FROM\s+(\w+)|JOIN\s+(\w+)'
            all_tables = set()
            for s in sqls:
                matches = re.findall(table_pattern, s)
                for match in matches:
                    table = match[0] if match[0] else match[1]
                    if table:
                        all_tables.add(table)
            stats['table_count'] = len(all_tables)
            
        except Exception as e:
            print(f"  ! 提取workload统计失败: {e}")
        
        return stats
    
    def format_metrics_text(self, inner_metrics: Dict) -> str:
        """
        格式化inner_metrics为文本
        
        返回：
            str: 格式化的指标文本
        """
        lines = []
        
        # 关键指标优先显示
        priority_metrics = [
            ('cache_hit_ratio', '缓存命中率', lambda x: f"{x*100:.1f}%"),
            ('xact_commit', '事务提交数', lambda x: f"{x/1000:.1f}k"),
            ('active_connections', '活跃连接数', lambda x: f"{int(x)}"),
            ('tup_returned', '返回元组数', lambda x: f"{x/1000:.1f}k"),
            ('disk_read_count', '磁盘读取次数', lambda x: f"{x/1e6:.1f}M"),
            ('cpu_usage', 'CPU使用率', lambda x: f"{x:.1f}%"),
        ]
        
        for key, label, formatter in priority_metrics:
            if key in inner_metrics:
                try:
                    value = formatter(inner_metrics[key])
                    lines.append(f"- {label}: {value}")
                except:
                    pass
        
        # 其他指标
        for key, value in inner_metrics.items():
            if key not in [m[0] for m in priority_metrics]:
                try:
                    if isinstance(value, float):
                        lines.append(f"- {key}: {value:.2f}")
                    else:
                        lines.append(f"- {key}: {value}")
                except:
                    pass
        
        return '\n'.join(lines)
    
    def format_config_as_human_readable(self, config: Dict) -> str:
        """
        将参数配置转换为人类可读的文本
        
        返回：
            str: 格式化的配置文本
        """
        # 单位映射
        units = {
            'shared_buffers': 'KB',
            'work_mem': 'KB',
            'maintenance_work_mem': 'KB',
            'effective_cache_size': 'KB',
            'temp_buffers': 'KB',
            'wal_buffers': 'KB'
        }
        
        lines = []
        priority_params = [
            'shared_buffers', 'work_mem', 'effective_cache_size', 
            'max_connections', 'maintenance_work_mem'
        ]
        
        # 优先参数
        for param in priority_params:
            if param in config:
                value = config[param]
                # 转换为可读的格式
                if param in units and isinstance(value, (int, float)):
                    kb_value = int(value)
                    if kb_value >= 1048576:  # >= 1GB
                        readable_value = f"{kb_value / 1048576:.1f}GB"
                    elif kb_value >= 1024:  # >= 1MB
                        readable_value = f"{kb_value / 1024:.1f}MB"
                    else:
                        readable_value = f"{kb_value}KB"
                else:
                    readable_value = str(value)
                
                lines.append(f'"{param}": "{readable_value}"')
        
        # 其他参数保持数值
        for param, value in sorted(config.items()):
            if param not in priority_params:
                lines.append(f'"{param}": {json.dumps(value)}')
        
        return '{\n  ' + ',\n  '.join(lines) + '\n}'
    
    def build_training_samples(self, selected_samples: List[Dict]) -> List[Dict]:
        """
        构建SFT格式的训练样本
        
        参数：
            selected_samples: 筛选后的高质量样本
        
        返回：
            list: 训练样本列表
        """
        print("\n=== 构建SFT训练数据 ===\n")
        
        training_samples = []
        
        for i, sample in enumerate(selected_samples, 1):
            try:
                workload_path = self.resolve_workload_path(sample)
                
                # 提取workload统计
                workload_stats = self.extract_workload_statistics(workload_path)
                
                # 提取Query Plans（前3条SQL）
                query_plans_text = sample.get('query_plans', '')
                if query_plans_text:
                    # 只保留前3条SQL的执行计划
                    plans_list = query_plans_text.split('=== SQL')
                    plans_list = [p for p in plans_list if p.strip()][:3]
                    query_plans_text = '=== SQL' + '=== SQL'.join(plans_list)
                
                # 格式化inner metrics
                metrics_text = self.format_metrics_text(sample.get('inner_metrics', {}))
                
                # 构建input部分
                input_text = f"""Workload Statistics:
- Total SQL: {workload_stats['total_sql']}
- Read-Write Ratio: {workload_stats['read_ratio']}% / {workload_stats['write_ratio']}%
- ORDER BY Proportion: {workload_stats['order_by_percent']}%
- GROUP BY Proportion: {workload_stats['group_by_percent']}%
- JOIN Count: {workload_stats['join_count']}
- Aggregation Functions: {workload_stats['aggregation_count']}
- Table Count: {workload_stats['table_count']}

Query Plans (Top 3):
{query_plans_text[:1000] if query_plans_text else 'N/A'}

Internal Metrics:
{metrics_text}"""
                
                # 构建output部分（参数配置）
                output_json = self.format_config_as_human_readable(sample['config'])
                
                # 系统提示
                instruction = """你是一个资深的PostgreSQL参数调优专家。你的任务是根据工作负载的特征和性能指标，推荐最优的数据库参数配置方案。

请严格按照以下要求：
1. 分析工作负载的SQL统计特性（读写比、聚合函数、JOIN等）
2. 考虑执行计划中的操作类型和cost估计
3. 根据内部性能指标（缓存命中率、I/O强度等）调整参数
4. 输出必须是严格的JSON格式，直接给出参数字典
5. 不要添加任何解释性文字，只输出JSON"""
                
                training_sample = {
                    'instruction': instruction,
                    'input': input_text,
                    'output': output_json
                }
                
                training_samples.append(training_sample)
                
                if i % 50 == 0:
                    print(f"  已处理 {i}/{len(selected_samples)} 个样本")
                
            except Exception as e:
                print(f"  ! 样本{i}构建失败: {e}")
                continue
        
        print(f"\n✓ 构建完成，共 {len(training_samples)} 个训练样本")
        return training_samples
    
    def save_training_data(self, training_samples: List[Dict]) -> bool:
        """
        保存训练数据到JSONL文件
        
        参数：
            training_samples: 训练样本列表
        
        返回：
            bool: 是否保存成功
        """
        try:
            os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for sample in training_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"\n✓ 训练数据已保存到: {self.output_path}")
            print(f"  共 {len(training_samples)} 个样本，文件大小: {os.path.getsize(self.output_path) / 1024 / 1024:.2f}MB")
            
            return True
            
        except Exception as e:
            print(f"✗ 保存训练数据失败: {e}")
            return False
    
    def build_and_save(self, min_samples: int = 200, max_samples: int = 300) -> bool:
        """
        完整的训练数据构建流程
        
        参数：
            min_samples: 最少样本数
            max_samples: 最多样本数
        
        返回：
            bool: 是否全部成功
        """
        print("\n" + "="*70)
        print("SFT训练数据构建")
        print("="*70)
        
        # 1. 加载样本
        if not self.load_samples():
            return False
        
        # 2. 筛选高质量样本
        selected_samples = self.select_high_quality_samples(min_samples, max_samples)
        
        # 3. 构建训练数据
        training_samples = self.build_training_samples(selected_samples)
        
        # 4. 保存
        if not self.save_training_data(training_samples):
            return False
        
        print("\n✓ 训练数据构建完成！\n")
        return True


def build_training_data(offline_sample_path: str, output_path: str) -> bool:
    """
    便捷函数：一键构建训练数据
    
    用法：
        build_training_data('offline_sample/offline_sample_tpch.jsonl', 'training_sft_data.jsonl')
    """
    builder = TrainingDataBuilder(offline_sample_path, output_path)
    return builder.build_and_save()


if __name__ == '__main__':
    # 示例用法
    build_training_data(
        offline_sample_path='offline_sample/offline_sample_tpch.jsonl',
        output_path='training_data/training_sft_data.jsonl'
    )
