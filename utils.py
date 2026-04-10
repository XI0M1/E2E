"""
工具函数模块（Utilities）
功能：提供日志记录、文件操作、数据处理等通用工具函数
"""

import json
import logging
import time
import os
import pandas as pd
from typing import Dict, List, Any


def get_logger(path: str) -> logging.Logger:
    """
    获取或创建日志记录器
    
    参数：
        path (str): 日志文件保存路径
    
    返回：
        logger: 配置好的日志记录器对象
    
    功能：
        1. 创建日志文件目录（若不存在）
        2. 配置日志格式和级别
        3. 避免重复添加handler
    
    日志格式：
        [时间:文件名#行号:日志级别]: 日志信息
        例如：[2024-03-16 10:30:45,123:tuner#L150:INFO]: 配置应用成功
    """
    # 确保日志目录存在
    log_dir = os.path.dirname(path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 为每个日志文件创建独立的logger实例
    logger_name = os.path.basename(path)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 创建文件handler
        file_handler = logging.FileHandler(path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '[%(asctime)s:%(filename)s#L%(lineno)d:%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 同时添加console输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def load_json(file_path: str) -> Dict[str, Any]:
    """
    读取JSON文件
    
    参数：
        file_path (str): JSON文件路径
    
    返回：
        dict: 解析后的JSON对象
    
    异常：
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON解析错误: {e}", e.doc, e.pos)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2):
    """
    保存数据到JSON文件
    
    参数：
        data (dict): 要保存的数据
        file_path (str): 保存路径
        indent (int): JSON缩进空格数，默认为2
    """
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    读取CSV文件到pandas DataFrame
    
    参数：
        file_path (str): CSV文件路径
    
    返回：
        DataFrame: pandas数据框
    """
    return pd.read_csv(file_path, encoding='utf-8')


def save_dataframe_to_csv(df: pd.DataFrame, file_path: str):
    """
    保存pandas DataFrame到CSV文件
    
    参数：
        df (DataFrame): pandas数据框
        file_path (str): 保存路径
    """
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    df.to_csv(file_path, index=False, encoding='utf-8')


def format_time(seconds: float) -> str:
    """
    将秒数格式化为易读的时间字符串
    
    参数：
        seconds (float): 秒数
    
    返回：
        str: 格式化的时间字符串
        例如：3661秒 -> "1h 1m 1s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def measure_time(func):
    """
    函数执行时间测量装饰器
    
    用法：
        @measure_time
        def my_function():
            # 函数体
            pass
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{func.__name__} 执行耗时: {format_time(elapsed)}")
        return result
    return wrapper


def dict_to_string(d: Dict[str, Any], indent: int = 0) -> str:
    """
    将字典递归转换为易读的字符串格式
    
    参数：
        d (dict): 要转换的字典
        indent (int): 缩进级别
    
    返回：
        str: 格式化的字符串
    """
    lines = []
    indent_str = "  " * indent
    
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}{key}:")
            lines.append(dict_to_string(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{indent_str}{key}: [")
            for item in value:
                if isinstance(item, dict):
                    lines.append(dict_to_string(item, indent + 1))
                else:
                    lines.append(f"{indent_str}  {item}")
            lines.append(f"{indent_str}]")
        else:
            lines.append(f"{indent_str}{key}: {value}")
    
    return "\n".join(lines)


def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    比较两个参数配置，返回差异
    
    参数：
        config1 (dict): 第一个配置
        config2 (dict): 第二个配置
    
    返回：
        dict: 包含差异的字典
            {'changed': {...}, 'only_in_config1': {...}, 'only_in_config2': {...}}
    """
    differences = {
        'changed': {},           # 两个配置都有但值不同的参数
        'only_in_config1': {},   # 仅在config1中的参数
        'only_in_config2': {}    # 仅在config2中的参数
    }
    
    all_keys = set(config1.keys()) | set(config2.keys())
    
    for key in all_keys:
        if key in config1 and key in config2:
            if config1[key] != config2[key]:
                differences['changed'][key] = {
                    'config1': config1[key],
                    'config2': config2[key]
                }
        elif key in config1:
            differences['only_in_config1'][key] = config1[key]
        else:
            differences['only_in_config2'][key] = config2[key]
    
    return differences


def validate_config(config: Dict[str, Any], 
                   knobs_detail: Dict[str, Dict]) -> tuple:
    """
    验证参数配置的合法性
    
    参数：
        config (dict): 待验证的配置
        knobs_detail (dict): 参数详细说明
    
    返回：
        tuple: (is_valid: bool, errors: list, warnings: list)
    """
    errors = []
    warnings = []
    
    for param_name, param_value in config.items():
        if param_name not in knobs_detail:
            warnings.append(f"参数 {param_name} 未定义")
            continue
        
        detail = knobs_detail[param_name]
        
        # 检查各项合理性
        if isinstance(param_value, float) and detail['type'] == 'integer':
            if param_value != int(param_value):
                errors.append(f"{param_name} 应为整数，得 {param_value}")
        
        if param_value < detail['min']:
            errors.append(f"{param_name} 低于最小值 {detail['min']}")
        elif param_value > detail['max']:
            errors.append(f"{param_name} 超过最大值 {detail['max']}")
    
    return len(errors) == 0, errors, warnings


def summarize_optimization_results(results_file: str) -> Dict[str, Any]:
    """
    汇总优化结果
    
    参数：
        results_file (str): 结果JSON文件路径
    
    返回：
        dict: 包含优化统计信息的字典
    """
    try:
        results = load_json(results_file)
        
        summary = {
            'total_evaluations': len(results.get('history', [])),
            'best_performance': results.get('best_performance', 0),
            'improvement': results.get('improvement_ratio', 0),
            'tuning_time': results.get('total_time', 0),
            'timestamp': results.get('timestamp', '')
        }
        
        return summary
    except Exception as e:
        print(f"读取结果文件失败: {e}")
        return {}

