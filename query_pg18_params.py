#!/usr/bin/env python3
"""
查询 PostgreSQL 18 中所有参数的实际取值范围
生成准确的 knob_config.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Database import Database
from config.parse_config import parse_args


def get_pg_parameter_ranges():
    """从 pg_settings 查询所有参数的范围"""
    
    config = parse_args('config/cloud.ini')
    db_config = config.get('database_config', {})
    
    db = Database(db_config)
    
    print("=" * 100)
    print("PostgreSQL 18 参数范围查询")
    print("=" * 100)
    
    try:
        # 查询所有参数信息
        result = db.execute_query("""
            SELECT name, setting, unit, boot_val, min_val, max_val, 
                   vartype, category, short_desc
            FROM pg_settings
            WHERE vartype IN ('integer', 'real', 'string', 'bool', 'enum')
            ORDER BY category, name
        """)
        
        print(f"\n总共找到 {len(result)} 个参数\n")
        
        # 关键参数列表（需要优化的参数）
        key_params = [
            'shared_buffers', 'work_mem', 'maintenance_work_mem', 'temp_buffers',
            'effective_cache_size', 'wal_buffers', 'checkpoint_timeout',
            'checkpoint_completion_target', 'wal_level', 'max_connections',
            'random_page_cost', 'cpu_tuple_cost', 'cpu_index_tuple_cost',
            'ssl', 'log_statement', 'log_duration'
        ]
        
        params_dict = {}
        
        print("关键参数范围信息：\n")
        print(f"{'参数名':30s} | {'类型':10s} | {'最小值':20s} | {'最大值':20s} | {'单位':10s}")
        print("-" * 130)
        
        for row in result:
            name, setting, unit, boot_val, min_val, max_val, vartype, category, short_desc = row
            
            if name in key_params:
                # 安全转换值
                min_display = str(min_val) if min_val else "N/A"
                max_display = str(max_val) if max_val else "N/A"
                unit_display = str(unit) if unit else "N/A"
                
                print(f"{str(name):30s} | {str(vartype):10s} | {min_display:20s} | {max_display:20s} | {unit_display:10s}")
                
                # 构建参数配置
                if vartype == 'integer':
                    try:
                        param_config = {
                            'type': 'integer',
                            'min': int(float(min_val)) if min_val else 0,
                            'max': int(float(max_val)) if max_val else 2147483647,
                            'default': int(float(setting)) if setting else 0,
                            'unit': str(unit) if unit else None,
                            'description': str(short_desc) if short_desc else "",
                            'category': str(category) if category else "Unknown"
                        }
                        params_dict[str(name)] = param_config
                    except Exception as e:
                        print(f"  ⚠️ 解析错误: {e}")
        
        print("\n" + "=" * 100)
        print("生成的参数配置")
        print("=" * 100 + "\n")
        
        # 输出为 JSON 格式
        output = json.dumps(params_dict, indent=2, ensure_ascii=False)
        print(output)
        
        # 保存到文件
        output_file = 'knob_config/knob_config_pg18_auto.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(params_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 已保存到: {output_file}")
        
        return params_dict
        
    except Exception as e:
        print(f"✗ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == '__main__':
    get_pg_parameter_ranges()
