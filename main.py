"""
主程序入口（Main Entry Point）
功能：数据库参数自动调优系统的完整工作流

工作流程：
    1. 解析命令行参数（数据库主机、类型、数据路径）
    2. 加载全局配置文件
    3. 第一阶段：离线采样
       - 使用直接数据库连接运行工作负载
       - 收集性能数据用于训练代理模型
       - 仅处理部分工作负载（<10个或前13个）以节省时间
       
    4. 训练代理模型（生成式大语言模型）
       - 使用离线采样数据进行微调
       - 生成参数配置的快速推荐模型
       
    5. 第二阶段：使用代理模型优化
       - 对所有工作负载使用训练好的代理模型
       - 无需再次执行数据库查询，大大加快优化速度
"""

import os
import sys
import argparse
import logging
from controller import tune
from config import parse_config
from surrogate.train_surrogate import train_surrogate
from feature_extractor import extract_workload_features
from training_data_builder import build_training_data


def setup_logging():
    """
    配置全局日志系统
    
    返回：
        logger: 全局日志记录器
    """
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger('Main')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(log_dir, 'main.log'))
        formatter = logging.Formatter(
            '[%(asctime)s - %(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def validate_workload_path(datapath: str, database_type: str) -> str:
    """
    验证数据路径并返回有效的工作负载目录
    
    参数：
        datapath (str): 数据路径
        database_type (str): 数据库类型
    
    返回：
        str: 验证后的工作负载路径
    """
    # 处理路径中可能的特殊字符
    if datapath == '/path':
        print("警告：使用默认路径'/path'，可能无法找到工作负载")
        print("建议通过--datapath指定实际数据路径")
        return datapath
    
    return datapath


if __name__ == "__main__":
    # ==================== 第一步：解析命令行参数 ====================
    print("=" * 70)
    print("基于LLM的数据库参数调优系统")
    print("=" * 70)
    
    parser = argparse.ArgumentParser(
        description="基于LLM的数据库参数调优系统"
    )
    parser.add_argument(
        '--host',
        type=str,
        default='192.168.1.100',
        help="数据库主机地址 (默认: 192.168.1.100)这里还需要再根据具体的配置进行修改"
    )
    parser.add_argument(
        '--database',  # 这里的这个参数的类型会在后续的代码中被用来区分不同的数据库类型或工作负载前缀
        type=str,
        default='tpch',
        help="数据库类型或工作负载前缀 (默认: tpch)"
    )
    parser.add_argument(
        '--datapath',
        type=str,
        default='/path',
        help="工作负载数据路径 (默认: /path)"
    )
    
    cmd = parser.parse_args()
    
    # ==================== 第二步：初始化日志和配置 ====================
    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("开始数据库参数调优任务")
    logger.info(f"主机: {cmd.host}, 数据库: {cmd.database}, 数据路径: {cmd.datapath}")
    
    # 加载配置文件
    logger.info("加载配置文件: config/config.ini")
    args = parse_config.parse_args("config/config.ini")
    
    # 使用命令行参数覆盖配置文件中的值
    args['ssh_config']['host'] = cmd.host
    args['database_config']['database'] = cmd.database
    args['database_config']['datapath'] = cmd.datapath
    args['tuning_config']['offline_sample'] += f"_{cmd.host}"
    
    # 验证数据路径
    datapath = validate_workload_path(cmd.datapath, cmd.database)
    
    # ==================== 第三步：发现工作负载 ====================
    logger.info(f"从 {datapath} 搜索工作负载...")
    
    try:
        all_files = os.listdir(datapath)
        workloads = [f for f in all_files if f.startswith(cmd.database)]
        logger.info(f"发现 {len(workloads)} 个工作负载")
        print(f"\n发现工作负载: {workloads}\n")
    except Exception as e:
        logger.error(f"无法访问工作负载路径: {e}")
        sys.exit(1)
    
    if len(workloads) == 0:
        logger.warning("未找到匹配的工作负载，程序退出")
        sys.exit(0)
    
    # ==================== 第四步：第一阶段 - 离线采样 ====================
    """
    离线采样阶段：
    - 目的：收集足够的(参数配置, 性能)数据对，用于训练代理模型
    - 方法：直接连接数据库，测试多种参数配置
    - 优化：仅对部分工作负载采样以节省时间
    
    分两种情况：
    1. 如果工作负载数 < 10: 对所有工作负载采样
    2. 如果工作负载数 >= 10: 仅对前13个工作负载采样
    """
    print("=" * 70)
    print("第一阶段：离线采样")
    print("=" * 70)
    logger.info("=" * 70)
    logger.info("开始离线采样阶段 (Direct Method)")
    logger.info("=" * 70)
    
    # 确保使用直接方法（非代理模型）
    args['benchmark_config']['tool'] = 'direct'
    
    samples_to_process = workloads if len(workloads) < 10 else workloads[:13]
    logger.info(f"将处理 {len(samples_to_process)} 个工作负载进行采样")
    
    successful_samples = 0
    failed_samples = 0
    
    for workload in samples_to_process:
        try:
            workload_id = os.path.splitext(workload)[0]  # 移除文件扩展名
            logger.info(f"------- 采样工作负载: {workload_id} -------")
            
            # 更新配置中的工作负载路径
            args['benchmark_config']['workload_path'] = os.path.join(datapath, workload)
            
            # 调用调优函数进行采样
            tune(workload_id, cmd.host, args)
            successful_samples += 1
            
        except Exception as e:
            logger.error(f"采样失败 [{workload}]: {e}")
            failed_samples += 1
            continue
    
    logger.info(f"离线采样完成: 成功={successful_samples}, 失败={failed_samples}")
    
    # ==================== 第四A步：提取工作负载特征 ====================
    print("\n" + "=" * 70)
    print("第四阶段A：提取工作负载特征向量")
    print("=" * 70)
    logger.info("=" * 70)
    logger.info("开始提取工作负载特征向量")
    logger.info("=" * 70)
    
    try:
        offline_sample_file = args['tuning_config'].get('offline_sample', f'offline_sample_{cmd.host}') + '.jsonl'
        logger.info(f"从 {offline_sample_file} 提取特征向量...")
        
        if extract_workload_features(offline_sample_file, cmd.database):
            logger.info("✓ 特征向量提取成功")
            print("✓ 特征向量提取完成\n")
        else:
            logger.warning("✗ 特征向量提取失败，继续执行其他步骤")
    except Exception as e:
        logger.error(f"特征向量提取异常: {e}")
        logger.warning("将继续执行后续步骤")
    
    # ==================== 第四B步：构建SFT训练数据 ====================
    print("=" * 70)
    print("第四阶段B：构建SFT训练数据")
    print("=" * 70)
    logger.info("=" * 70)
    logger.info("开始构建SFT训练数据")
    logger.info("=" * 70)
    
    try:
        offline_sample_file = args['tuning_config'].get('offline_sample', f'offline_sample_{cmd.host}') + '.jsonl'
        output_sft_file = f'training_data/training_sft_data_{cmd.database}.jsonl'
        
        logger.info(f"从离线采样数据构建SFT训练数据...")
        logger.info(f"输入: {offline_sample_file}")
        logger.info(f"输出: {output_sft_file}")
        
        if build_training_data(offline_sample_file, output_sft_file):
            logger.info("✓ SFT训练数据构建成功")
            print("✓ SFT训练数据构建完成\n")
        else:
            logger.warning("✗ SFT训练数据构建失败，继续执行其他步骤")
    except Exception as e:
        logger.error(f"SFT训练数据构建异常: {e}")
        logger.warning("将继续执行后续步骤")
    
    # ==================== 第五步：训练代理模型 ====================
    print("\n" + "=" * 70)
    print("第二阶段：训练代理模型")
    print("=" * 70)
    logger.info("=" * 70)
    logger.info("开始训练代理模型")
    logger.info("=" * 70)
    
    try:
        logger.info(f"训练代理模型: {cmd.database}")
        train_surrogate(cmd.database)
        logger.info("代理模型训练完成")
    except Exception as e:
        logger.error(f"代理模型训练失败: {e}")
        logger.warning("将继续使用直接方法，不使用代理模型")
    
    # ==================== 第六步：第二阶段 - 使用代理模型优化 ====================
    """
    使用代理模型阶段：
    - 目的：对所有工作负载快速生成优化参数
    - 方法：使用训练好的生成式大语言模型推荐参数
    - 优势：无需多轮数据库查询，速度快，成本低
    """
    print("\n" + "=" * 70)
    print("第三阶段：使用代理模型优化")
    print("=" * 70)
    logger.info("=" * 70)
    logger.info("开始使用代理模型优化所有工作负载")
    logger.info("=" * 70)
    
    # 使用代理模型作为调优工具
    args['benchmark_config']['tool'] = 'surrogate'
    args['surrogate_config']['model_path'] = f'surrogate/{cmd.database}.pkl'
    args['surrogate_config']['feature_path'] = f'SuperWG/feature/{cmd.database}.json'
    
    surrogate_successful = 0
    surrogate_failed = 0
    
    for workload in workloads:
        try:
            workload_id = os.path.splitext(workload)[0]
            logger.info(f"------- 优化工作负载: {workload_id} (使用代理模型) -------")
            
            # 设置工作负载路径
            args['benchmark_config']['workload_path'] = f'SuperWG/res/gpt_workloads/{workload}'
            
            # 调用调优函数
            tune(workload_id, cmd.host, args)
            surrogate_successful += 1
            
        except Exception as e:
            logger.error(f"优化失败 [{workload}]: {e}")
            surrogate_failed += 1
            continue
    
    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("调优任务完成！")
    print("=" * 70)
    print(f"\n统计信息：")
    print(f"  离线采样阶段: 成功={successful_samples}, 失败={failed_samples}")
    print(f"  代理模型优化: 成功={surrogate_successful}, 失败={surrogate_failed}")
    print(f"  总体成功率: {successful_samples + surrogate_successful}/{len(samples_to_process) + len(workloads)}")
    print(f"\n详细日志请查看: ./logs/main.log\n")
    
    logger.info("=" * 70)
    logger.info("所有任务完成")
    logger.info(f"离线采样: {successful_samples} 成功, {failed_samples} 失败")
    logger.info(f"代理模型优化: {surrogate_successful} 成功, {surrogate_failed} 失败")
    logger.info("=" * 70)