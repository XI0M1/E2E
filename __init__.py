"""
数据库参数自动调优系统包初始化

主模块：
    - main.py: 主程序入口，完整的工作流
    - controller.py: 调优控制器，协调优化过程
    - tuner.py: 调优器实现，包含优化算法
    - utils.py: 工具函数库
    - Database.py: 数据库连接和管理
    - Vectorlib.py: 工作负载向量库
    - stress_testing_tool.py: 性能测试工具

子包：
    - config/: 配置文件和解析器
    - knob_config/: 数据库参数配置
    - surrogate/: 代理模型训练
    - safe/: 安全约束框架
"""

__version__ = "1.0.0"
__author__ = "Database Tuning Team"

# 导入主要组件
from controller import tune
from surrogate.train_surrogate import train_surrogate
from Database import Database
from Vectorlib import VectorLibrary
from stress_testing_tool import stress_testing_tool

__all__ = [
    'tune',
    'train_surrogate',
    'Database',
    'VectorLibrary',
    'stress_testing_tool'
]
