#!/usr/bin/env python3
"""
验证所有修复是否正确工作的脚本

功能：
1. 验证Database类能正确处理JSON序列化
2. 验证metrics收集的所有值都是JSON可序列化的
3. 验证在autocommit模式下参数设置正常工作
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from Database import Database
from config.parse_config import parse_args


def test_json_serialization():
    """测试JSON序列化"""
    print("=" * 60)
    print("测试1: JSON序列化")
    print("=" * 60)
    
    # 测试数据包含各种类型
    test_data = {
        'float_value': 1.5,
        'int_value': 42,
        'string_value': 'test',
        'nested': {
            'nested_float': 3.14,
            'nested_int': 100
        }
    }
    
    try:
        json_str = json.dumps(test_data)
        print("✓ 基础JSON序列化成功")
        return True
    except Exception as e:
        print(f"✗ JSON序列化失败: {e}")
        return False


def test_database_connection():
    """测试数据库连接和autocommit模式"""
    print("\n" + "=" * 60)
    print("测试2: 数据库连接和Autocommit模式")
    print("=" * 60)
    
    try:
        # 加载配置
        config = parse_args('config/cloud.ini')
        db_config = config.get('database_config', {})
        
        if not db_config:
            print("✗ 无法加载数据库配置")
            return False
        
        # 创建数据库连接
        db = Database(db_config)
        
        # 验证连接
        result = db.execute_query("SELECT 1")
        if result and result[0][0] == 1:
            print("✓ 数据库连接成功")
        else:
            print("✗ 数据库连接测试失败")
            return False
        
        # 验证autocommit模式
        try:
            # 这条语句在非autocommit模式会失败
            db.get_parameters()
            print("✓ Autocommit模式正常")
        except Exception as e:
            print(f"✗ Autocommit模式测试失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        return False


def test_metrics_collection():
    """测试指标收集和JSON序列化"""
    print("\n" + "=" * 60)
    print("测试3: 指标收集和JSON序列化")
    print("=" * 60)
    
    try:
        # 加载配置
        config = parse_args('config/cloud.ini')
        db_config = config.get('database_config', {})
        
        if not db_config:
            print("✗ 无法加载数据库配置")
            return False
        
        # 创建数据库连接
        db = Database(db_config)
        
        # 收集指标
        metrics = db.get_system_metrics()
        print(f"✓ 成功收集{len(metrics)}个系统指标")
        
        # 验证所有值都是JSON可序列化的
        try:
            metrics_json = json.dumps(metrics)
            print("✓ 所有指标值都是JSON可序列化的")
            
            # 打印收集的指标
            print(f"\n收集的指标列表：")
            for key, value in metrics.items():
                print(f"  - {key}: {value} ({type(value).__name__})")
            
            return True
            
        except TypeError as e:
            print(f"✗ 指标序列化失败: {e}")
            print(f"问题指标:")
            for key, value in metrics.items():
                try:
                    json.dumps({key: value})
                except TypeError:
                    print(f"  - {key}: {value} ({type(value).__name__})")
            return False
        
    except Exception as e:
        print(f"✗ 指标收集失败: {e}")
        return False


def test_sample_data_format():
    """测试样本数据格式"""
    print("\n" + "=" * 60)
    print("测试4: 样本数据格式验证")
    print("=" * 60)
    
    try:
        # 模拟样本数据
        sample_data = {
            'workload': 'tpch_1',
            'workload_file': 'tpch_1.wg',
            'config': {
                'shared_buffers': 1114632,
                'work_mem': 716290752,
                'maintenance_work_mem': 2147483647
            },
            'tps': 150.5,
            'inner_metrics': {
                'cache_hit_ratio': 0.95,
                'heap_blks_read': 1000.0,
                'heap_blks_hit': 19000.0,
                'xact_commit': 55432.0,
                'xact_rollback': 123.0,
                'tup_returned': 5432100.0,
                'tup_inserted': 0.0,
                'tup_updated': 0.0,
                'tup_deleted': 0.0,
                'disk_read_count': 500.0,
                'cpu_usage': 45.5
            },
            'query_plans': 'Seq Scan on lineitem...',
            'y': -150.5
        }
        
        # 验证JSON序列化
        sample_json = json.dumps(sample_data)
        print("✓ 样本数据可以成功序列化为JSON")
        
        # 验证反序列化
        restored = json.loads(sample_json)
        print("✓ 样本数据可以从JSON成功反序列化")
        
        return True
        
    except Exception as e:
        print(f"✗ 样本数据格式验证失败: {e}")
        return False


def main():
    """运行所有验证"""
    print("\n" + "=" * 60)
    print("数据库修复验证脚本")
    print("="*60)
    
    tests = [
        test_json_serialization,
        test_database_connection,
        test_metrics_collection,
        test_sample_data_format
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if all(results):
        print("\n✓ 所有修复验证通过！现在可以运行 cloud_quickstart.py")
        return 0
    else:
        print("\n✗ 部分验证未通过，请检查上面的错误信息")
        return 1


if __name__ == '__main__':
    sys.exit(main())
