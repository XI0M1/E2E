"""
调优控制器（Controller）
功能：协调整个调优流程，包括初始化调优器、执行调优、保存结果等

云环境版本：直接连接本地PostgreSQL，不使用SSH
"""

import os
import json
import logging
from datetime import datetime
from tuner import tuner
from Database import Database
import utils


def tune(workload, args):
    """
    执行数据库参数调优的主控制函数
    
    参数：
        workload (str): 工作负载标识符，用于区分不同的数据库工作负载
        args (dict): 包含所有配置参数的字典，从config.ini解析而来
    
    返回：
        调优结果被保存到文件，返回None
    
    工作流程：
        1. 配置日志记录器
        2. 初始化数据库连接
        3. 初始化压力测试工具
        4. 执行调优过程
        5. 保存调优结果到文件
        6. 记录调优统计信息
    """
    
    # 1. 配置日志系统
    # 为每个工作负载创建独立的日志文件，便于追踪调优过程
    log_dir = args['tuning_config'].get('log_dir', './logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"tuning_{workload}_{timestamp}.log")
    args['tuning_config']['log_path'] = log_path
    
    logger = utils.get_logger(log_path)
    logger.info(f"===== 开始调优任务: {workload} =====")
    logger.info(f"调优方法: {args['tuning_config'].get('tuning_method', 'SMAC')}")
    
    try:
        # 2. 初始化数据库连接
        logger.info("正在连接数据库...")
        db = Database(args)
        logger.info("✓ 数据库连接成功")
        
        # 3. 初始化调优器
        logger.info("正在初始化调优器...")
        tuning_engine = tuner(args, db, logger)
        logger.info("✓ 调优器初始化成功")
        
        # 4. 执行调优
        logger.info("开始执行参数优化...")
        tuning_engine.tune()
        logger.info("✓ 参数优化完成")
        
        # 5. 保存调优结果
        result_data = {
            'workload': workload,
            'timestamp': timestamp,
            'status': 'completed',
            'tuning_method': args['tuning_config'].get('tuning_method', 'SMAC'),
            'warmup_method': args['tuning_config'].get('warmup_method', 'default')
        }
        
        result_dir = args['tuning_config'].get('result_dir', './results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        result_path = os.path.join(result_dir, f"result_{workload}_{timestamp}.json")
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        logger.info(f"✓ 调优结果已保存到: {result_path}")
        
        # 5. 记录成功                
        logger.info(f"===== 调优任务完成: {workload} =====\n")
        
        return result_data
        
    except Exception as e:
        # 异常处理：记录错误信息，便于调试
        logger.error(f"调优过程中出错: {str(e)}", exc_info=True)
        logger.error(f"===== 调优任务失败: {workload} =====\n")
        raise
