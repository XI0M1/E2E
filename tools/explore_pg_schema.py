#!/usr/bin/env python3
"""
探索 PostgreSQL 18 中 pg_stat_user_tables 的实际列结构
"""

import psycopg2
from config.parse_config import parse_args

def explore_pg_stat_tables():
    """查询pg_stat_user_tables的实际列"""
    config = parse_args('config/cloud.ini')
    db_config = config.get('database_config', {})
    
    try:
        conn = psycopg2.connect(
            host=db_config.get('host', 'localhost'),
            port=int(db_config.get('port', 5432)),
            database=db_config.get('database', 'postgres'),
            user=db_config.get('username', 'postgres'),
            password=db_config.get('password', 'postgres')
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("=" * 70)
        print("pg_stat_user_tables 的所有列")
        print("=" * 70)
        
        # 查询表的列信息
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'pg_stat_user_tables'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        print(f"\n总共 {len(columns)} 列:\n")
        
        for i, (col_name, data_type) in enumerate(columns, 1):
            print(f"{i:2d}. {col_name:30s} | {data_type}")
        
        # 检查是否存在缺失的列
        print("\n" + "=" * 70)
        print("检查缺失的列")
        print("=" * 70)
        missing_cols = ['n_tup_returned', 'n_tup_inserted', 'n_tup_updated', 'n_tup_deleted']
        col_names = [col[0] for col in columns]
        
        for col in missing_cols:
            if col in col_names:
                print(f"✓ {col:30s} 存在")
            else:
                print(f"✗ {col:30s} 不存在")
        
        # 查询可能的替代列
        print("\n" + "=" * 70)
        print("可能的替代或相关列")
        print("=" * 70)
        
        related_patterns = ['tup', 'tuple', 'returned', 'inserted', 'updated', 'deleted', 'scan', 'idx']
        matched_cols = [c for c in col_names if any(p in c.lower() for p in related_patterns)]
        
        print("\n包含 tup/tuple/scan/idx 等关键词的列:\n")
        for col in sorted(matched_cols):
            print(f"  - {col}")
        
        # 尝试查询样本数据
        print("\n" + "=" * 70)
        print("pg_stat_user_tables 样本数据")
        print("=" * 70)
        
        cursor.execute("SELECT * FROM pg_stat_user_tables LIMIT 1")
        if cursor.description:
            print("\n第一行数据:\n")
            row = cursor.fetchone()
            if row:
                for col_name, value in zip([d[0] for d in cursor.description], row):
                    print(f"  {col_name:30s} = {value}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    explore_pg_stat_tables()
