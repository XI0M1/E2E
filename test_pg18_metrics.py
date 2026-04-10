#!/usr/bin/env python3
"""
验证 PostgreSQL 18 兼容性改造的详细测试脚本
检查所有指标收集逻辑
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from Database import Database
from config.parse_config import parse_args


def test_all_metrics():
    """测试所有指标收集"""
    print("=" * 80)
    print("PostgreSQL 18 指标收集详细测试")
    print("=" * 80)
    
    try:
        config = parse_args('config/cloud.ini')
        db_config = config.get('database_config', {})
        
        db = Database(db_config)
        
        # 收集指标
        print("\n[1/3] 收集系统指标...")
        metrics = db.get_system_metrics()
        
        # 验证收集结果
        print("\n[2/3] 验证指标类型和值...")
        
        expected_metrics = {
            # 缓存相关
            'heap_blks_read': (float, "堆块读取数"),
            'heap_blks_hit': (float, "堆块命中数"),
            'cache_hit_ratio': (float, "缓存命中率 (0-1)"),
            
            # 数据库相关
            'database_size': (float, "数据库大小 (字节)"),
            'active_connections': (float, "活跃连接数"),
            
            # 事务相关
            'xact_commit': (float, "事务提交数"),
            'xact_rollback': (float, "事务回滚数"),
            
            # 元组操作（PG18 新格式）
            'seq_tup_read': (float, "顺序扫描读取的行数"),
            'idx_tup_fetch': (float, "索引扫描获取的行数"),
            'tup_inserted': (float, "插入的元组数 (n_tup_ins)"),
            'tup_updated': (float, "更新的元组数 (n_tup_upd)"),
            'tup_deleted': (float, "删除的元组数 (n_tup_del)"),
            'live_tuples': (float, "活跃行数 (n_live_tup)"),
            'dead_tuples': (float, "死行数 (n_dead_tup)"),
            
            # I/O 和 CPU
            'disk_read_count': (float, "磁盘读取次数"),
            'cpu_usage': (float, "CPU 使用率 (%)"),
        }
        
        # 逐个检查指标
        print("\n指标详细检查：\n")
        
        validation_results = []
        for metric_name, (expected_type, description) in sorted(expected_metrics.items()):
            if metric_name in metrics:
                value = metrics[metric_name]
                actual_type = type(value)
                
                # 类型检查
                type_ok = isinstance(value, expected_type)
                
                # 值范围检查（特殊指标）
                range_ok = True
                if metric_name == 'cache_hit_ratio':
                    range_ok = 0 <= value <= 1
                elif metric_name == 'cpu_usage':
                    range_ok = 0 <= value <= 100
                
                # 非负值检查（大多数计数指标）
                non_negative = True
                if metric_name not in ['cache_hit_ratio', 'cpu_usage']:
                    non_negative = value >= 0
                
                status = "✓" if (type_ok and range_ok and non_negative) else "✗"
                validation_results.append(type_ok and range_ok and non_negative)
                
                print(f"{status} {metric_name:25s} = {value:15.2f} | {description}")
                if not (type_ok and range_ok and non_negative):
                    print(f"  └─ 警告: 类型={actual_type.__name__}, 范围检查={'✓' if range_ok else '✗'}, 非负={'✓' if non_negative else '✗'}")
            else:
                print(f"✗ {metric_name:25s} = 缺失             | {description}")
                validation_results.append(False)
        
        # JSON 序列化测试
        print("\n[3/3] JSON 序列化测试...")
        try:
            metrics_json = json.dumps(metrics, indent=2)
            print("✓ 所有指标可以成功序列化为 JSON")
            print(f"  JSON 大小: {len(metrics_json)} 字节")
        except TypeError as e:
            print(f"✗ JSON 序列化失败: {e}")
            validation_results.append(False)
        
        # 总结
        print("\n" + "=" * 80)
        print("验证总结")
        print("=" * 80)
        
        passed = sum(validation_results)
        total = len(validation_results)
        
        print(f"\n通过验证: {passed}/{total}")
        
        if all(validation_results):
            print("\n✓ 所有指标验证通过！")
            print("✓ PG18 兼容性修改完成")
            print("✓ 可以运行 verify_fixes.py 和 cloud_quickstart.py")
            return 0
        else:
            print("\n✗ 部分指标未通过验证，请检查上面的警告")
            return 1
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def check_pg18_compatibility():
    """检查 PG18 兼容性信息"""
    print("\n" + "=" * 80)
    print("PostgreSQL 18 兼容性信息")
    print("=" * 80)
    
    compatibility_info = """
列名变更（旧 -> 新）:
  n_tup_returned     -> 分为 seq_tup_read + idx_tup_fetch
  n_tup_inserted     -> n_tup_ins
  n_tup_updated      -> n_tup_upd
  n_tup_deleted      -> n_tup_del

新增列（PG18）:
  n_live_tup         - 活跃行数
  n_dead_tup         - 死行数
  seq_tup_read       - 顺序扫描读取的行数
  idx_tup_fetch      - 索引扫描获取的行数
  n_tup_hot_upd      - HOT 更新的元组数
  n_tup_newpage_upd  - 新页面更新的元组数

指标计算调整:
  缓存命中率 = heap_blks_hit / (heap_blks_hit + heap_blks_read)
  总读取行数 = seq_tup_read + idx_tup_fetch
"""
    print(compatibility_info)


if __name__ == '__main__':
    check_pg18_compatibility()
    result = test_all_metrics()
    sys.exit(result)
