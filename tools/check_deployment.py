#!/usr/bin/env python3
"""
AutoDL 云环境部署检查工具 (Deployment Checker)
功能：验证所有组件是否正确安装和配置

使用方式：
    python check_deployment.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_conda_env():
    """检查Conda环境"""
    print("\n[检查] Conda环境...")
    try:
        result = subprocess.run(['conda', 'info', '--json'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            # 检查是否有llm环境
            envs = info.get('envs', [])
            if 'llm' in envs or any('llm' in e for e in envs):
                print("  ✓ Conda llm环境已找到")
                print(f"    所在路径: {info.get('active_prefix', 'N/A')}")
                return True
    except Exception as e:
        print(f"  ! Conda 检查失败: {e}")
    
    print("  ✗ 未找到Conda llm环境")
    return False


def check_python():
    """检查Python版本"""
    print("\n[检查] Python环境...")
    print(f"  Python版本: {sys.version}")
    if '3.10' in sys.version or '3.9' in sys.version:
        print("  ✓ Python版本OK (3.9/3.10)")
        return True
    else:
        print("  ! Python版本可能不兼容，推荐使用3.9或3.10")
        return False


def check_python_packages():
    """检查必要的Python包"""
    print("\n[检查] Python依赖包...")
    
    required_packages = {
        'psycopg2': '用于PostgreSQL连接',
        'numpy': 'NumPy数值计算',
        'json': 'JSON处理',
    }
    
    success = True
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package:15} - {description}")
        except ImportError:
            print(f"  ✗ {package:15} - 【缺失】需要安装: pip install {package}")
            success = False
    
    return success


def check_postgresql():
    """检查PostgreSQL连接"""
    print("\n[检查] PostgreSQL数据库...")
    
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='tpch',
            user='dbuser',
            password='7684105'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"  ✓ PostgreSQL连接成功")
        print(f"    版本: {version[:40]}...")
        
        # 检查是否有TPC-H表
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema='public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        tpch_tables = ['lineitem', 'orders', 'customer', 'supplier', 'part', 'partsupp', 'nation', 'region']
        found_tables = [t for t in tpch_tables if t in tables]
        
        if found_tables:
            print(f"  ✓ TPC-H数据已导入 ({len(found_tables)}/{len(tpch_tables)} 个表)")
            for t in found_tables[:3]:
                print(f"    - {t}")
            if len(found_tables) > 3:
                print(f"    - ... 及其他 {len(found_tables) - 3} 个表")
        else:
            print("  ✗ 未找到TPC-H表")
            print("    请先导入TPC-H数据")
            return False
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ✗ PostgreSQL连接失败: {e}")
        print("    检查项:")
        print("      - host: localhost")
        print("      - port: 5432")
        print("      - database: tpch")
        print("      - user: dbuser")
        print("      - password: 7684105")
        return False


def check_workload_files():
    """检查workload文件"""
    print("\n[检查] Workload文件...")
    
    workload_dir = '/root/autodl-tmp/llm/data/olap/tpch'
    
    if not os.path.exists(workload_dir):
        print(f"  ✗ Workload目录不存在: {workload_dir}")
        return False
    
    print(f"  ✓ Workload目录存在: {workload_dir}")
    
    wg_files = [f for f in os.listdir(workload_dir) if f.endswith('.wg')]
    
    if not wg_files:
        print("  ✗ 未找到任何.wg文件")
        return False
    
    print(f"  ✓ 找到 {len(wg_files)} 个.wg文件:")
    for wf in sorted(wg_files)[:5]:
        print(f"    - {wf}")
    if len(wg_files) > 5:
        print(f"    - ... 及其他 {len(wg_files) - 5} 个文件")
    
    return True


def check_project_structure():
    """检查项目结构"""
    print("\n[检查] 项目结构...")
    
    required_files = {
        'main.py': '主程序入口',
        'cloud_quickstart.py': '云环境快速启动脚本',
        'config/config.ini': '配置文件',
        'config/cloud.ini': '云环境配置',
        'Database.py': '数据库模块',
        'stress_testing_tool.py': '压力测试工具',
        'feature_extractor.py': '特征提取器',
        'training_data_builder.py': '训练数据构建',
        'knob_config/knob_config.json': '参数配置',
    }
    
    success = True
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file_path:35} ({size:6} bytes)")
        else:
            print(f"  ✗ {file_path:35} 【缺失】")
            success = False
    
    return success


def check_output_directories():
    """检查输出目录"""
    print("\n[检查] 输出目录...")
    
    dirs_to_create = [
        './logs',
        './offline_sample',
        './training_data',
        './SuperWG/feature',
        './results'
    ]
    
    for dir_path in dirs_to_create:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  ✓ {dir_path} (已创建)")
            except Exception as e:
                print(f"  ✗ {dir_path} (创建失败: {e})")
                return False
    
    return True


def check_gpu():
    """检查GPU"""
    print("\n[检查] GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ GPU可用")
            print(f"    设备: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA版本: {torch.version.cuda}")
        else:
            print("  ! GPU不可用，但CPU模式也可以使用")
            return True
    except ImportError:
        print("  ! PyTorch未安装，无法检查GPU")
        return False
    
    return True


def main():
    """主检查函数"""
    print("=" * 70)
    print("AutoDL 云环境部署检查")
    print("=" * 70)
    
    checks = [
        ("Conda环境", check_conda_env()),
        ("Python版本", check_python()),
        ("Python依赖", check_python_packages()),
        ("PostgreSQL数据库", check_postgresql()),
        ("Workload文件", check_workload_files()),
        ("项目结构", check_project_structure()),
        ("输出目录", check_output_directories()),
        ("GPU支持", check_gpu()),
    ]
    
    print("\n" + "=" * 70)
    print("检查结果总结")
    print("=" * 70)
    
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {check_name}")
    
    all_passed = all(result for _, result in checks)
    
    if all_passed:
        print("\n✓ 所有检查都通过！可以开始用以下命令启动Phase 1:")
        print("\n  python cloud_quickstart.py --config config/cloud.ini --database tpch\n")
    else:
        print("\n✗ 存在一些问题，请根据上面的提示修复后再试\n")
    
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
