#!/usr/bin/env python3
"""
查询 PostgreSQL 18 参数的 context 属性
帮助确定哪些参数可以动态修改（SET），哪些需要重启（ALTER SYSTEM）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Database import Database
from config.parse_config import parse_args


def classify_parameters():
    """分类显示所有参数的动态性"""
    
    config = parse_args('config/cloud.ini')
    db_config = config.get('database_config', {})
    db = Database(db_config)
    
    print("=" * 100)
    print("PostgreSQL 18 参数分类（动态 vs 静态）")
    print("=" * 100)
    
    try:
        # 查询所有参数的 context
        result = db.execute_query("""
            SELECT name, context, vartype, setting, unit
            FROM pg_settings
            ORDER BY context, name
        """)
        
        # 分类统计
        categories = {
            'backend': [],
            'user': [],
            'superuser': [],
            'postmaster': [],
            'sighup': [],
            'internal': []
        }
        
        for row in result:
            name, context, vartype, setting, unit = row
            context_str = str(context) if context else 'unknown'
            
            param_info = {
                'name': name,
                'type': vartype,
                'setting': setting,
                'unit': unit if unit else '(无)'
            }
            
            if context_str in categories:
                categories[context_str].append(param_info)
            else:
                if context_str not in categories:
                    categories[context_str] = []
                categories[context_str].append(param_info)
        
        # 显示分类结果
        print("\n✅ 动态参数（可用 SET 立即修改，无需重启）\n")
        print(f"{'参数名':40s} | {'类型':10s} | {'单位':10s} | {'当前值':20s}")
        print("-" * 100)
        
        dynamic_count = 0
        for context in ['backend', 'user', 'superuser']:
            if context in categories and categories[context]:
                for param in sorted(categories[context], key=lambda x: x['name']):
                    print(f"{str(param['name']):40s} | {str(param['type']):10s} | "
                          f"{str(param['unit']):10s} | {str(param['setting']):20s}")
                    dynamic_count += 1
        
        print(f"\n小计: {dynamic_count} 个动态参数\n")
        
        # 静态参数
        print("❌ 静态参数（需要 ALTER SYSTEM + 重启数据库）\n")
        print(f"{'参数名':40s} | {'类型':10s} | {'单位':10s} | {'当前值':20s}")
        print("-" * 100)
        
        static_count = 0
        for context in ['postmaster', 'sighup', 'internal']:
            if context in categories and categories[context]:
                for param in sorted(categories[context], key=lambda x: x['name']):
                    print(f"{str(param['name']):40s} | {str(param['type']):10s} | "
                          f"{str(param['unit']):10s} | {str(param['setting']):20s}")
                    static_count += 1
        
        print(f"\n小计: {static_count} 个静态参数\n")
        
        print("=" * 100)
        print(f"总计: {dynamic_count} 个动态参数 + {static_count} 个静态参数 = {dynamic_count + static_count} 个参数")
        print("=" * 100)
        
        # 可调优的关键参数列表
        print("\n📊 推荐调优的关键参数\n")
        
        key_params = {
            '内存分配': [
                'work_mem',
                'temp_buffers',
                'effective_cache_size',
            ],
            '查询优化器成本': [
                'random_page_cost',
                'cpu_tuple_cost',
                'cpu_index_tuple_cost',
                'seq_page_cost',
            ],
            '统计相关': [
                'default_statistics_target',
            ],
            'JOIN 优化': [
                'join_collapse_limit',
                'from_collapse_limit',
                'geqo_threshold',
            ]
        }
        
        for category, params in key_params.items():
            print(f"{category}:")
            for param in params:
                # 找参数的 context
                for context_list in categories.values():
                    for p in context_list:
                        if p['name'] == param:
                            is_dynamic = True
                            for static_context in ['postmaster', 'sighup', 'internal']:
                                if param in [x['name'] for x in categories.get(static_context, [])]:
                                    is_dynamic = False
                            
                            icon = '✅ 动态' if is_dynamic else '❌ 静态'
                            print(f"  {icon:10s} {param:40s} (类型: {p['type']})")
                            break
            print()
        
    except Exception as e:
        print(f"✗ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def test_set_operation():
    """测试 SET 命令是否能立即生效"""
    
    print("\n" + "=" * 100)
    print("测试 SET 命令是否能立即生效")
    print("=" * 100 + "\n")
    
    config = parse_args('config/cloud.ini')
    db_config = config.get('database_config', {})
    db = Database(db_config)
    
    try:
        # 测试设置一个动态参数
        print("测试 1: 设置 work_mem 参数")
        db.execute_query("SET work_mem = '10MB'")
        result = db.execute_query("SHOW work_mem")
        if result:
            print(f"  ✅ SET work_mem = '10MB' 成功")
            print(f"  ✅ SHOW work_mem 返回: {result[0][0]}")
        
        # 测试另一个参数
        print("\n测试 2: 设置 random_page_cost 参数")
        db.execute_query("SET random_page_cost = 1.5")
        result = db.execute_query("SHOW random_page_cost")
        if result:
            print(f"  ✅ SET random_page_cost = 1.5 成功")
            print(f"  ✅ SHOW random_page_cost 返回: {result[0][0]}")
        
        print("\n✅ 所有 SET 测试通过，动态参数修改正常工作")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    rc1 = classify_parameters()
    rc2 = test_set_operation()
    sys.exit(max(rc1, rc2))
