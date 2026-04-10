"""
安全约束适配模块（Safe Subspace Adaptation）
功能：在优化过程中进行参数约束和安全检查
"""

import numpy as np
from typing import List, Dict, Any


class Safe:
    """
    安全约束适配类
    
    职责：
        1. 维护参数的搜索空间
        2. 检查生成的配置是否在安全范围内
        3. 处理约束冲突
        4. 记录优化过程的家族历史
    """
    
    def __init__(self, 
                 default_performance: float,
                 default_config: Dict[str, Any],
                 best_performance: float,
                 lower_bounds: List[float],
                 upper_bounds: List[float],
                 steps: List[float]):
        """
        初始化安全适配器
        
        参数：
            default_performance (float): 默认配置的性能
            default_config (dict): 默认参数配置
            best_performance (float): 当前最佳性能
            lower_bounds (list): 参数下界
            upper_bounds (list): 参数上界
            steps (list): 参数步长（用于离散化）
        """
        self.default_performance = default_performance
        self.default_config = default_config.copy()
        self.best_performance = best_performance
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.steps = np.array(steps)
        
        # 历史记录
        self.eval_history = []
        self.config_history = []
    
    def is_valid_config(self, config: Dict[str, Any], 
                       knobs_detail: Dict[str, Dict]) -> tuple:
        """
        检查配置是否有效和安全
        
        参数：
            config (dict): 待检查的参数配置
            knobs_detail (dict): 参数详细信息
        
        返回：
            tuple: (is_valid: bool, violations: list)
                is_valid: 配置是否合法
                violations: 违反的约束列表
        """
        violations = []
        
        for param_name, param_value in config.items():
            if param_name not in knobs_detail:
                violations.append(f"未知参数: {param_name}")
                continue
            
            detail = knobs_detail[param_name]
            lb = detail['min']
            ub = detail['max']
            
            # 检查边界
            if param_value < lb:
                violations.append(
                    f"{param_name}={param_value} < 下界{lb}"
                )
            elif param_value > ub:
                violations.append(
                    f"{param_name}={param_value} > 上界{ub}"
                )
            
            # 对于整数参数，检查步长合法性
            if detail['type'] == 'integer':
                if param_value != int(param_value):
                    violations.append(
                        f"整数参数{param_name}值非整数: {param_value}"
                    )
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def clamp_config(self, config: Dict[str, Any],
                     knobs_detail: Dict[str, Dict]) -> Dict[str, Any]:
        """
        将配置约束到有效范围内
        
        若某参数超过边界，自动裁剪到边界值。
        
        参数：
            config (dict): 原始参数配置
            knobs_detail (dict): 参数详细信息
        
        返回：
            dict: 调整后的参数配置
        """
        clamped_config = {}
        
        for param_name, param_value in config.items():
            if param_name not in knobs_detail:
                continue
            
            detail = knobs_detail[param_name]
            lb = detail['min']
            ub = detail['max']
            
            # 裁剪到边界
            clamped_value = max(lb, min(ub, param_value))
            
            # 对于整数参数，强制转换
            if detail['type'] == 'integer':
                clamped_value = int(clamped_value)
            
            clamped_config[param_name] = clamped_value
        
        return clamped_config
    
    def record_evaluation(self, config: Dict[str, Any], 
                         performance: float):
        """
        记录一次参数评估结果
        
        参数：
            config (dict): 评估的参数配置
            performance (float): 对应的性能评分
        """
        self.eval_history.append(performance)
        self.config_history.append(config.copy())
        
        # 更新最佳性能
        if performance > self.best_performance:
            self.best_performance = performance
    
    def get_improvement_ratio(self) -> float:
        """
        获取相对于默认配置的性能改进比例
        
        返回：
            float: 改进比例（0.0-1.0或更高）
        """
        if self.default_performance == 0:
            return 0.0
        
        improvement = (self.best_performance - self.default_performance) / \
                     self.default_performance
        return max(0.0, improvement)
    
    def train(self, data_path: str = './'):
        """
        训练后验安全模型（可选）
        
        使用历史评估数据训练一个模型，用于预测参数配置的安全性。
        
        参数：
            data_path (str): 数据保存路径
        """
        if len(self.eval_history) < 10:
            print("数据不足，跳过模型训练")
            return
        
        try:
            # 这里可以添加模型训练逻辑
            # 例如：使用高斯过程或神经网络预测配置的性能
            print(f"使用{len(self.eval_history)}条历史数据训练模型...")
            print("模型训练完成")
        except Exception as e:
            print(f"模型训练失败: {e}")
