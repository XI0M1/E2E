#!/usr/bin/env python3
"""
演示脚本：完整的参数调优工作流（包括动态和静态参数）

这个脚本演示了如何：
1. 在快速模式下测试参数（仅动态）
2. 在完整模式下测试参数（动态 + 静态 + 重启）
3. 对比两种模式的性能和耗时
"""

import sys
import time
import json
import logging
from typing import Dict, List, Any

# 假设这些模块在当前目录
try:
    from Database import Database
    from stress_testing_tool import stress_testing_tool
    from utils import load_config
except ImportError as e:
    print(f"错误：无法导入必要模块：{e}")
    print("确保 Database.py, stress_testing_tool.py, utils.py 在当前目录")
    sys.exit(1)


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('demo')


def demo_mode_comparison(config_path: str = 'config/cloud.ini'):
    """演示不同的测试模式对比"""
    
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("PostgreSQL 参数调优演示")
    logger.info("=" * 80)
    
    # 加载配置
    try:
        config = load_config(config_path)
        logger.info(f"✓ 已加载配置: {config_path}")
    except Exception as e:
        logger.error(f"✗ 无法加载配置: {e}")
        return False
    
    # 初始化数据库
    try:
        database = Database(config)
        database.connect()
        logger.info("✓ 数据库连接成功")
    except Exception as e:
        logger.error(f"✗ 数据库连接失败: {e}")
        return False
    
    # 初始化压力测试工具
    try:
        stress_tool = stress_testing_tool(
            config, 
            database, 
            logger, 
            'offline_sample/offline_sample'
        )
        logger.info("✓ 压力测试工具初始化成功")
    except Exception as e:
        logger.error(f"✗ 工具初始化失败: {e}")
        return False
    
    # 定义测试参数配置
    test_configs = [
        {
            'name': '配置 A：轻量级（小内存）',
            'config': {
                'work_mem': 1048576,        # 1MB
                'effective_cache_size': 262144,  # 256MB
                'random_page_cost': 1.1,
            },
            'has_static': False
        },
        {
            'name': '配置 B：均衡型',
            'config': {
                'work_mem': 4194304,        # 4MB
                'effective_cache_size': 1048576, # 1GB
                'random_page_cost': 1.0,
            },
            'has_static': False
        },
        {
            'name': '配置 C：包含静态参数（完整优化）',
            'config': {
                'work_mem': 8388608,        # 8MB (动态)
                'effective_cache_size': 2097152, # 2GB (动态)
                'random_page_cost': 0.9,   # 动态
                'max_connections': 200,    # 静态 (需指定重启)
                'max_worker_processes': 8, # 静态 (需指定重启)
            },
            'has_static': True
        }
    ]
    
    # 执行演示
    logger.info("\n" + "=" * 80)
    logger.info("模式 1：快速测试 - 仅动态参数")
    logger.info("=" * 80)
    
    demo_fast_mode(logger, stress_tool, test_configs[:2])
    
    logger.info("\n" + "=" * 80)
    logger.info("模式 2：完整测试 - 动态 + 静态 + 重启")
    logger.info("=" * 80)
    logger.info("注意：静态参数测试会导致数据库重启，可能需要 60-90 秒")
    
    # 询问是否要执行完整测试
    if len(test_configs) > 2:
        response = input("\n要执行包含静态参数和重启的完整测试吗？(y/n): ").strip().lower()
        if response == 'y':
            demo_complete_mode(logger, stress_tool, test_configs[2:])
        else:
            logger.info("跳过了完整测试模式")
    
    logger.info("\n" + "=" * 80)
    logger.info("演示完成")
    logger.info("=" * 80)
    
    return True


def demo_fast_mode(logger: logging.Logger, 
                   stress_tool: stress_testing_tool,
                   test_configs: List[Dict[str, Any]]):
    """演示快速模式：仅使用动态参数"""
    
    results = []
    
    for test_case in test_configs:
        logger.info("\n" + "-" * 60)
        logger.info(f"测试：{test_case['name']}")
        logger.info("-" * 60)
        
        start_time = time.time()
        try:
            # 快速模式：apply_static=False
            performance = stress_tool.test_config(
                test_case['config'],
                apply_static=False,      # 仅应用动态参数
                restart_if_static=False  # 不重启
            )
            elapsed = time.time() - start_time
            
            logger.info(f"✓ 测试完成")
            logger.info(f"  性能评分：{performance:.2f} TPS")
            logger.info(f"  耗时：{elapsed:.1f} 秒")
            
            results.append({
                'name': test_case['name'],
                'mode': '快速',
                'performance': performance,
                'time': elapsed
            })
            
        except Exception as e:
            logger.error(f"✗ 测试失败：{e}")
    
    # 汇总快速模式结果
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("快速模式测试汇总:")
        logger.info("=" * 60)
        for result in results:
            logger.info(
                f"{result['name']:<30} | "
                f"性能: {result['performance']:>7.2f} TPS | "
                f"耗时: {result['time']:>6.1f}s"
            )


def demo_complete_mode(logger: logging.Logger,
                       stress_tool: stress_testing_tool,
                       test_configs: List[Dict[str, Any]]):
    """演示完整模式：使用动态和静态参数"""
    
    results = []
    
    for test_case in test_configs:
        logger.info("\n" + "-" * 60)
        logger.info(f"测试：{test_case['name']}")
        logger.info("-" * 60)
        
        start_time = time.time()
        try:
            # 完整模式：apply_static=True, restart_if_static=True
            performance = stress_tool.test_config(
                test_case['config'],
                apply_static=True,       # 应用所有参数（包括静态）
                restart_if_static=True   # 如有静态参数则重启
            )
            elapsed = time.time() - start_time
            
            logger.info(f"✓ 测试完成")
            logger.info(f"  性能评分：{performance:.2f} TPS")
            logger.info(f"  耗时：{elapsed:.1f} 秒（包括数据库重启）")
            
            results.append({
                'name': test_case['name'],
                'mode': '完整',
                'performance': performance,
                'time': elapsed
            })
            
        except Exception as e:
            logger.error(f"✗ 测试失败：{e}")
    
    # 汇总完整模式结果
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("完整模式测试汇总:")
        logger.info("=" * 60)
        for result in results:
            logger.info(
                f"{result['name']:<30} | "
                f"性能: {result['performance']:>7.2f} TPS | "
                f"耗时: {result['time']:>6.1f}s"
            )


def comparison_analysis():
    """对比分析演示"""
    
    logger = setup_logging()
    
    logger.info("\n" + "=" * 80)
    logger.info("性能模式对比分析")
    logger.info("=" * 80)
    
    comparison_data = {
        '模式': ['仅动态参数', '动态+静态+重启'],
        '参数应用时间': ['<1 秒', '60-80 秒'],
        '数据库重启': ['否', '是'],
        '总测试时间': ['5-10 分钟', '20-30 分钟'],
        '用途': ['快速搜索', '最终验证'],
        '适用场景': ['论文算法开发', '生产部署']
    }
    
    logger.info("\n对比表：")
    logger.info("-" * 80)
    
    # 打印表头
    header_items = list(comparison_data.keys())
    col_widths = [max(len(str(h)), 
                     max(len(str(comparison_data[h][i])) 
                         for i in range(len(comparison_data[h])))) 
                  for h in header_items]
    
    header_line = " | ".join(
        f"{h:<{col_widths[i]}}" 
        for i, h in enumerate(header_items)
    )
    logger.info(header_line)
    logger.info("-" * 80)
    
    # 打印数据行
    for row_idx in range(len(comparison_data[header_items[0]])):
        data_line = " | ".join(
            f"{str(comparison_data[h][row_idx]):<{col_widths[i]}}"
            for i, h in enumerate(header_items)
        )
        logger.info(data_line)
    
    logger.info("-" * 80)
    
    logger.info("\n关键要点：")
    logger.info("  ✓ 快速模式可实现 30-50 倍加速（相比传统方法）")
    logger.info("  ✓ 适合论文中的初期参数空间探索")
    logger.info("  ✓ 完整模式保留了所有功能，用于最终验证")
    logger.info("  ✓ 可根据需要灵活选择模式")


def parameter_classification_demo():
    """参数分类演示"""
    
    logger = setup_logging()
    
    logger.info("\n" + "=" * 80)
    logger.info("参数分类和上下文")
    logger.info("=" * 80)
    
    parameter_types = {
        '动态参数（backend/user/superuser 上下文）': {
            '说明': '可通过 SET 命令立即修改，无需重启',
            '示例': [
                'work_mem',
                'effective_cache_size',
                'random_page_cost',
                'seq_page_cost',
                'default_statistics_target',
                'maintenance_work_mem',
            ],
            '应用时间': '<1 秒'
        },
        '静态参数（postmaster/sighup 上下文）': {
            '说明': '需要通过 ALTER SYSTEM 后重启数据库',
            '示例': [
                'max_connections',
                'max_worker_processes',
                'max_parallel_workers',
                'shared_buffers',
                'wal_buffers',
            ],
            '应用时间': '60-80 秒'
        }
    }
    
    for param_class, details in parameter_types.items():
        logger.info(f"\n{param_class}:")
        logger.info(f"  说明：{details['说明']}")
        logger.info(f"  应用时间：{details['应用时间']}")
        logger.info(f"  示例参数：")
        for param in details['示例']:
            logger.info(f"    - {param}")


def usage_guide():
    """使用指南"""
    
    logger = setup_logging()
    
    logger.info("\n" + "=" * 80)
    logger.info("快速使用指南")
    logger.info("=" * 80)
    
    logger.info("""
1. 快速测试（推荐用于论文开发）：
   ────────────────────────────────
   performance = stress_tool.test_config(config)
   # 或显式
   stress_tool.test_config(config, apply_static=False)
   
   特点：快速（5-10分钟），只用动态参数
   用途：参数空间探索、算法开发

2. 完整测试（推荐用于最终验证）：
   ────────────────────────────────
   performance = stress_tool.test_config(
       config,
       apply_static=True,
       restart_if_static=True
   )
   
   特点：完整（20-30分钟），包括所有参数和重启
   用途：生产部署、论文最终结果

3. 调试模式（应用但不重启）：
   ────────────────────────────────
   performance = stress_tool.test_config(
       config,
       apply_static=True,
       restart_if_static=False
   )
   
   特点：中等速度，方便检查参数
   用途：调试、参数验证

4. 查看参数分类：
   ────────────────────────────────
   python classify_parameters.py
   
   显示所有可调参数的上下文（动态/静态）
""")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PostgreSQL 参数调优演示脚本'
    )
    parser.add_argument(
        '--config',
        default='config/cloud.ini',
        help='配置文件路径'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'fast', 'compare', 'classify', 'guide'],
        default='full',
        help='演示模式'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        # 完整演示
        success = demo_mode_comparison(args.config)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'fast':
        # 只演示快速模式
        logger = setup_logging()
        logger.info("快速模式演示（仅动态参数）")
        logger.info("实现方式：stress_tool.test_config(config, apply_static=False)")
        
    elif args.mode == 'compare':
        # 对比分析
        comparison_analysis()
    
    elif args.mode == 'classify':
        # 参数分类
        parameter_classification_demo()
    
    elif args.mode == 'guide':
        # 使用指南
        usage_guide()
    
    print("\n" + "=" * 80)
    print("演示完成！更多信息请查看 STATIC_PARAMETER_RESTART.md")
    print("=" * 80)
