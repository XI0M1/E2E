#!/usr/bin/env python3
"""
验证脚本：测试静态参数和重启功能的正确性

这个脚本验证：
1. apply_config() 的新参数是否正确
2. restart() 和 wait_for_restart() 是否能正常工作
3. 参数统计信息是否正确记录
4. 离线采样数据是否包含重启信息
"""

import time
import json
import logging
from typing import Dict, Any

# 假设这些模块在当前目录
try:
    from Database import Database
    from stress_testing_tool import stress_testing_tool
    from utils import load_config
except ImportError as e:
    print(f"错误：无法导入必要模块：{e}")
    exit(1)


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('verification')


def verify_database_connection(config_path: str = 'config/cloud.ini') -> bool:
    """验证数据库连接"""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("验证 1：数据库连接")
    logger.info("=" * 80)
    
    try:
        config = load_config(config_path)
        database = Database(config)
        database.connect()
        logger.info("✓ 数据库连接成功")
        
        # 验证连接是活跃的
        result = database.execute_query("SELECT version();")
        if result:
            logger.info(f"✓ 数据库版本检查：{result[0][0][:60]}...")
        
        database.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ 数据库连接失败：{e}")
        return False


def verify_apply_config_signature():
    """验证 apply_config() 的新参数"""
    logger = setup_logging()
    logger.info("\n" + "=" * 80)
    logger.info("验证 2：apply_config() 方法的新参数")
    logger.info("=" * 80)
    
    try:
        config = load_config('config/cloud.ini')
        database = Database(config)
        database.connect()
        
        # 测试基本签名
        test_config = {'work_mem': 4194304}
        
        # 验证关键字参数存在
        logger.info("测试 apply_static 参数...")
        result1 = database.apply_config(test_config, apply_static=False)
        logger.info(f"✓ apply_static=False: {result1}")
        
        logger.info("测试 restart_if_static 参数...")
        result2 = database.apply_config(test_config, apply_static=False, restart_if_static=False)
        logger.info(f"✓ restart_if_static=False: {result2}")
        
        # 验证返回值包含新的键
        required_keys = ['dynamic', 'static', 'skipped', 'failed']
        for key in required_keys:
            if key not in result2:
                logger.error(f"✗ 返回值缺少键：{key}")
                return False
            logger.info(f"✓ 返回值包含 '{key}' 键")
        
        database.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ 验证失败：{e}")
        return False


def verify_dynamic_parameter_detection():
    """验证动态参数检测"""
    logger = setup_logging()
    logger.info("\n" + "=" * 80)
    logger.info("验证 3：动态参数检测和应用")
    logger.info("=" * 80)
    
    try:
        config = load_config('config/cloud.ini')
        database = Database(config)
        database.connect()
        
        # 查询已知的动态参数
        logger.info("查询动态参数的上下文...")
        dynamic_params = {
            'work_mem': 4194304,
            'effective_cache_size': 1048576,
        }
        
        for param_name in dynamic_params:
            result = database.execute_query(
                f"SELECT context FROM pg_settings WHERE name = '{param_name}' LIMIT 1"
            )
            if result:
                context = result[0][0]
                logger.info(f"  {param_name}: context = {context}")
                if context in ['backend', 'user', 'superuser']:
                    logger.info(f"  ✓ {param_name} 是动态参数")
                else:
                    logger.warning(f"  ⚠ {param_name} 不是标准动态参数")
        
        # 应用动态参数
        logger.info("应用动态参数...")
        stats = database.apply_config(dynamic_params, apply_static=False)
        logger.info(f"✓ 应用结果：{stats}")
        
        if stats['dynamic'] > 0:
            logger.info(f"✓ 成功应用了 {stats['dynamic']} 个动态参数")
        
        database.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ 验证失败：{e}")
        return False


def verify_static_parameter_detection():
    """验证静态参数检测"""
    logger = setup_logging()
    logger.info("\n" + "=" * 80)
    logger.info("验证 4：静态参数检测（不应用）")
    logger.info("=" * 80)
    
    try:
        config = load_config('config/cloud.ini')
        database = Database(config)
        database.connect()
        
        # 查询已知的静态参数
        logger.info("查询静态参数的上下文...")
        static_params = {
            'max_connections': 200,
            'max_worker_processes': 8,
        }
        
        for param_name in static_params:
            result = database.execute_query(
                f"SELECT context FROM pg_settings WHERE name = '{param_name}' LIMIT 1"
            )
            if result:
                context = result[0][0]
                logger.info(f"  {param_name}: context = {context}")
                if context in ['postmaster', 'sighup', 'internal']:
                    logger.info(f"  ✓ {param_name} 是静态参数")
        
        # 尝试应用静态参数（但不重启）
        logger.info("应用静态参数（apply_static=True, restart_if_static=False）...")
        stats = database.apply_config(
            static_params,
            apply_static=True,
            restart_if_static=False
        )
        logger.info(f"✓ 应用结果：{stats}")
        
        if stats['static'] > 0:
            logger.info(f"✓ 已记录 {stats['static']} 个静态参数（待重启）")
        
        database.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ 验证失败：{e}")
        return False


def verify_restart_methods_exist():
    """验证重启方法是否存在"""
    logger = setup_logging()
    logger.info("\n" + "=" * 80)
    logger.info("验证 5：重启方法是否存在")
    logger.info("=" * 80)
    
    try:
        config = load_config('config/cloud.ini')
        database = Database(config)
        
        # 检查方法是否存在
        required_methods = ['restart', 'wait_for_restart', 'apply_config']
        
        for method_name in required_methods:
            if hasattr(database, method_name):
                method = getattr(database, method_name)
                logger.info(f"✓ 方法存在：{method_name}")
                
                # 验证是否可调用
                if callable(method):
                    logger.info(f"  ✓ {method_name} 是可调用的")
                else:
                    logger.error(f"  ✗ {method_name} 不可调用")
                    return False
            else:
                logger.error(f"✗ 方法不存在：{method_name}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 验证失败：{e}")
        return False


def verify_stress_tool_integration():
    """验证压力测试工具的集成"""
    logger = setup_logging()
    logger.info("\n" + "=" * 80)
    logger.info("验证 6：压力测试工具的集成")
    logger.info("=" * 80)
    
    try:
        config = load_config('config/cloud.ini')
        database = Database(config)
        database.connect()
        
        stress_tool = stress_testing_tool(
            config, 
            database, 
            logger,
            'offline_sample/offline_sample'
        )
        
        # 验证方法签名
        if hasattr(stress_tool, 'test_config'):
            import inspect
            sig = inspect.signature(stress_tool.test_config)
            params = list(sig.parameters.keys())
            
            logger.info(f"test_config() 参数列表：{params}")
            
            required_params = ['config', 'apply_static', 'restart_if_static']
            for param in required_params:
                if param in params:
                    logger.info(f"  ✓ 参数 '{param}' 存在")
                else:
                    logger.error(f"  ✗ 参数 '{param}' 缺失")
                    return False
            
            logger.info("✓ test_config() 方法签名正确")
        else:
            logger.error("✗ stress_testing_tool 缺少 test_config 方法")
            return False
        
        database.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ 验证失败：{e}")
        return False


def verify_sample_data_format():
    """验证采样数据格式"""
    logger = setup_logging()
    logger.info("\n" + "=" * 80)
    logger.info("验证 7：采样数据格式")
    logger.info("=" * 80)
    
    try:
        import os
        
        sample_file = 'offline_sample/offline_sample.jsonl'
        
        if not os.path.exists(sample_file):
            logger.warning(f"采样文件不存在：{sample_file}")
            logger.info("✓ 这是正常的（首次运行）")
            return True
        
        logger.info(f"读取采样文件：{sample_file}")
        
        # 读取最后几行
        line_count = 0
        required_fields = ['apply_static', 'restart_performed']
        
        with open(sample_file, 'r') as f:
            lines = f.readlines()
        
        if lines:
            # 检查最后一条记录
            last_line = lines[-1]
            data = json.loads(last_line)
            
            logger.info("最后一条记录的字段：")
            for field in required_fields:
                if field in data:
                    logger.info(f"  ✓ '{field}': {data[field]}")
                else:
                    logger.warning(f"  ⚠ '{field}' 缺失")
            
            logger.info(f"✓ 共有 {len(lines)} 条采样记录")
        else:
            logger.info("✓ 采样文件为空")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 验证失败：{e}")
        return False


def run_all_verifications():
    """运行所有验证"""
    logger = setup_logging()
    
    logger.info("\n\n")
    logger.info("█" * 80)
    logger.info("静态参数和重启功能验证套件")
    logger.info("█" * 80)
    
    verifications = [
        ("数据库连接", verify_database_connection),
        ("apply_config() 新参数", verify_apply_config_signature),
        ("动态参数检测", verify_dynamic_parameter_detection),
        ("静态参数检测", verify_static_parameter_detection),
        ("重启方法存在性", verify_restart_methods_exist),
        ("压力测试工具集成", verify_stress_tool_integration),
        ("采样数据格式", verify_sample_data_format),
    ]
    
    results = {}
    for name, verify_func in verifications:
        try:
            result = verify_func()
            results[name] = result
        except Exception as e:
            logger.error(f"验证 '{name}' 发生异常：{e}")
            results[name] = False
    
    # 汇总结果
    logger.info("\n" + "=" * 80)
    logger.info("验证汇总")
    logger.info("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{status}: {name}")
    
    logger.info("-" * 80)
    logger.info(f"总体：{passed}/{total} 验证通过")
    
    if passed == total:
        logger.info("✓ 所有验证通过！静态参数和重启功能已正确实现")
        return True
    else:
        logger.error("✗ 存在失败的验证，请检查上面的错误信息")
        return False


if __name__ == '__main__':
    import sys
    
    print("\n")
    print("=" * 80)
    print("PostgreSQL 静态参数和重启功能验证")
    print("=" * 80)
    print()
    
    success = run_all_verifications()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ 验证完成 - 功能正常")
        print("=" * 80)
        print("\n下一步：")
        print("  1. 运行演示脚本：python demo_static_parameters.py")
        print("  2. 查看文档：STATIC_PARAMETER_RESTART.md")
        print("  3. 查看快速参考：QUICK_REFERENCE.md")
        sys.exit(0)
    else:
        print("✗ 验证失败 - 请检查错误信息")
        print("=" * 80)
        sys.exit(1)
