"""
向量库模块（Vector Library）
功能：管理工作负载特征向量，用于相似度计算和工作负载映射
"""

import json
import numpy as np
from typing import Dict, List, Tuple


class VectorLibrary:
    """
    向量库类
    
    职责：
        1. 加载工作负载特征向量
        2. 计算向量相似度
        3. 查找最相似的工作负载
        4. 支持多种距离度量方法
    """
    
    def __init__(self, database: str, feature_path: str = None):
        """
        初始化向量库
        
        参数：
            database (str): 数据库名称
            feature_path (str): 特征文件路径，若为None则使用默认路径
        """
        self.database = database
        if feature_path is None:
            feature_path = f'SuperWG/feature/{database}.json'
        
        self.feature_path = feature_path
        self.features = self._load_features()
    
    def _load_features(self) -> Dict[str, List[float]]:
        """
        加载工作负载特征向量
        
        返回：
            dict: 工作负载ID -> 特征向量的映射
        """
        try:
            with open(self.feature_path, 'r') as f:
                features = json.load(f)
            print(f"已加载{len(features)}个工作负载的特征向量")
            return features
        except FileNotFoundError:
            print(f"警告：特征文件不存在: {self.feature_path}")
            return {}
        except Exception as e:
            print(f"加载特征文件失败: {e}")
            return {}
    
    def euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算欧几里得距离
        
        参数：
            vec1, vec2: 特征向量
        
        返回：
            float: 距离值
        """
        return np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2))
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度
        
        参数：
            vec1, vec2: 特征向量
        
        返回：
            float: 相似度值（范围0-1）
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def find_most_similar(self, target_feature: List[float], k: int = 3,
                         metric: str = 'cosine') -> List[str]:
        """
        查找最相似的k个工作负载
        
        参数：
            target_feature (list): 目标工作负载特征向量
            k (int): 返回最相似的k个工作负载
            metric (str): 距离度量方法 ('cosine' 或 'euclidean')
        
        返回：
            list: 最相似的工作负载ID列表（按相似度排序）
        """
        if not self.features:
            print("特征库为空")
            return []
        
        similarities = {}
        
        for workload_id, feature_vector in self.features.items():
            try:
                if metric == 'cosine':
                    sim = self.cosine_similarity(target_feature, feature_vector)
                elif metric == 'euclidean':
                    sim = -self.euclidean_distance(target_feature, feature_vector)
                else:
                    sim = self.cosine_similarity(target_feature, feature_vector)
                
                similarities[workload_id] = sim
            except Exception as e:
                print(f"计算相似度失败 ({workload_id}): {e}")
        
        # 按相似度排序，返回top-k
        sorted_workloads = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_workloads[:k]]
    
    def get_feature(self, workload_id: str) -> List[float]:
        """
        获取指定工作负载的特征向量
        
        参数：
            workload_id (str): 工作负载ID
        
        返回：
            list: 特征向量，若不存在则返回None
        """
        return self.features.get(workload_id)
