#!/usr/bin/env python3
"""
AutoDL云环境快速启动脚本 (Cloud QuickStart)
功能：简化Phase 1流程，一键运行离线采样+特征提取+SFT数据生成

使用方式：
    python cloud_quickstart.py

环境要求：
    - PostgreSQL 18 (localhost)
    - TPC-H数据已导入 (scale=1)
    - Workload文件已下载到 /root/autodl-tmp/llm/data/olap/tpch/
"""

import os
import sys
import json
import logging
import argparse
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Database import Database
from parameter_subsystem import ParameterExecutionSubsystem
from sampling_runtime import SamplingRunRecorder
from stress_testing_tool import stress_testing_tool
from feature_extractor import extract_workload_features
from training_data_builder import build_training_data
from config.parse_config import parse_args
from knob_config.parse_knob_config import get_knobs


def setup_logging(log_dir='./logs'):
    """配置全局日志"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"cloud_phase1_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s - %(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('CloudPhase1')
    logger.info("=" * 70)
    logger.info("AutoDL 云环境 - Phase 1 快速启动")
    logger.info("=" * 70)
    return logger


def verify_environment(config, logger):
    """
    验证运行环境
    
    检查内容：
    - PostgreSQL连接
    - workload目录存在
    - 必要的Python模块
    """
    logger.info("\n[步骤 1/6] 验证运行环境...\n")
    
    # 1. 检查数据库连接
    logger.info("检查PostgreSQL连接...")
    try:
        db = Database(config['database_config'])
        result = db.execute_query("SELECT version()")
        logger.info(f"✓ PostgreSQL连接成功")
        logger.info(f"  版本: {result[0][0][:50]}...")
        db.conn.close()
    except Exception as e:
        logger.error(f"✗ PostgreSQL连接失败: {e}")
        logger.error("请检查database_config配置：")
        logger.error(f"  host: {config['database_config'].get('host')}")
        logger.error(f"  port: {config['database_config'].get('port')}")
        logger.error(f"  database: {config['database_config'].get('database')}")
        logger.error(f"  username: {config['database_config'].get('username')}")
        return False
    
    # 2. 检查workload目录
    logger.info("\n检查workload目录...")
    workload_dir = config['database_config'].get('workload_datapath')
    if not os.path.exists(workload_dir):
        logger.error(f"✗ Workload目录不存在: {workload_dir}")
        return False
    
    workload_files = [f for f in os.listdir(workload_dir) if f.endswith('.wg')]
    if not workload_files:
        logger.error(f"✗ 未找到.wg文件: {workload_dir}")
        return False
    
    logger.info(f"✓ 发现{len(workload_files)}个workload文件:")
    for wf in sorted(workload_files)[:10]:  # 最多显示10个
        logger.info(f"  - {wf}")
    if len(workload_files) > 10:
        logger.info(f"  ... 及其他 {len(workload_files) - 10} 个文件")
    
    return True


def discover_workloads(config, logger):
    """
    发现workload文件列表
    
    返回：
        list: workload文件路径列表
    """
    workload_dir = config['database_config'].get('workload_datapath')
    workload_files = sorted([
        os.path.join(workload_dir, f) 
        for f in os.listdir(workload_dir) 
        if f.endswith('.wg')
    ])
    
    return workload_files


def build_phase1_config(
    base_config,
    workload_dir,
    database_name,
    sample_prefix,
    workload_limit=3,
    samples_per_workload=5,
    resume_sampling=False,
):
    """为 Phase 1 构建更稳定的运行配置。"""
    config = json.loads(json.dumps(base_config))
    config.setdefault('database_config', {})
    config.setdefault('tuning_config', {})
    config.setdefault('benchmark_config', {})
    config.setdefault('parameter_execution', {})

    config['database_config']['database'] = database_name
    config['database_config']['workload_datapath'] = workload_dir
    config['tuning_config']['offline_sample'] = sample_prefix
    config['benchmark_config']['tool'] = 'direct'
    config['benchmark_config']['timeout'] = config['benchmark_config'].get('timeout', '300')
    config['benchmark_config']['fetch_result_rows'] = 'false'
    config['benchmark_config']['fresh_session_per_test'] = 'true'
    config['benchmark_config']['workload_limit'] = str(workload_limit)
    config['benchmark_config']['samples_per_workload'] = str(samples_per_workload)
    config['benchmark_config']['resume_sampling'] = 'true' if resume_sampling else 'false'
    config['parameter_execution']['apply_reload'] = 'true'
    config['parameter_execution']['apply_restart'] = 'true'
    config['parameter_execution']['reload_if_needed'] = 'true'
    config['parameter_execution']['verify'] = 'true'
    config['parameter_execution']['health_check'] = 'true'
    config['parameter_execution']['rollback_on_failure'] = 'true'
    config['parameter_execution']['session_mode'] = 'always'
    return config


def generate_sample_config(knobs_detail, rng, max_knobs=None):
    """生成一份轻量随机参数配置，尽量保持参数类型正确。"""
    config = {}
    items = list(knobs_detail.items())
    if max_knobs is not None and max_knobs > 0:
        items = items[:max_knobs]

    for knob_name, knob_detail in items:
        knob_type = knob_detail.get('type')

        if knob_type == 'integer':
            min_val = int(knob_detail.get('min', 0))
            max_val = int(knob_detail.get('max', min_val))
            if max_val < min_val:
                min_val, max_val = max_val, min_val
            config[knob_name] = rng.randint(min_val, max_val) if max_val > min_val else min_val

        elif knob_type == 'float':
            min_val = float(knob_detail.get('min', 0.0))
            max_val = float(knob_detail.get('max', min_val))
            if max_val < min_val:
                min_val, max_val = max_val, min_val
            config[knob_name] = rng.uniform(min_val, max_val) if max_val > min_val else min_val

        elif knob_type == 'enum':
            enum_values = knob_detail.get('enum_values', [])
            if enum_values:
                config[knob_name] = rng.choice(enum_values)

    return config


def generate_safe_sample_config(knobs_detail, rng, parameter_subsystem, max_attempts=20, max_knobs=None):
    """Generate a random configuration that passes joint validation."""
    last_validation = None
    for _ in range(max_attempts):
        candidate = generate_sample_config(knobs_detail, rng, max_knobs=max_knobs)
        validation = parameter_subsystem.validate_config(candidate)
        last_validation = validation
        if validation.get('valid'):
            return validation.get('normalized_config', candidate)
    return None


def build_single_knob_test_value(db, knob_name: str, knob_detail: Dict[str, Any]) -> Optional[Any]:
    metadata = db.get_parameter_info(knob_name)
    if not metadata:
        return None

    vartype = metadata.get('vartype')
    current_value = metadata.get('setting')

    try:
        if knob_name == 'temp_file_limit':
            # 0 表示禁止临时文件，虽然语法合法，但对大量 OLAP workload 来说过于激进，
            # 更适合验证“能否运行”的候选值应选一个温和的正数。
            current_num = float(current_value)
            min_val = float(knob_detail.get('min', metadata.get('min_val', current_num)))
            max_val = float(knob_detail.get('max', metadata.get('max_val', current_num)))
            safe_candidate = min(max(1048576.0, max(min_val, 1.0)), max_val)  # 默认 1GB
            if safe_candidate == current_num and max_val > safe_candidate:
                safe_candidate = max(safe_candidate + 1.0, 1.0)
            return int(round(safe_candidate))

        if vartype == 'bool':
            return not (str(current_value).lower() in {'on', 'true', '1'})

        if vartype == 'enum':
            enum_values = knob_detail.get('enum_values', [])
            if not enum_values:
                return None
            current = str(current_value)
            for candidate in enum_values:
                if str(candidate) != current:
                    return candidate
            return None

        current_num = float(current_value)
        min_val = float(knob_detail.get('min', metadata.get('min_val', current_num)))
        max_val = float(knob_detail.get('max', metadata.get('max_val', current_num)))

        candidate = current_num + 1
        if candidate > max_val:
            candidate = max(current_num - 1, min_val)
        if candidate == current_num:
            return None

        if vartype == 'integer':
            return int(round(candidate))
        return candidate
    except Exception:
        return None


def choose_static_smoke_parameters(db, knobs_detail, limit=2):
    """选择少量更适合做 smoke test 的 postmaster 参数。"""
    preferred_order = [
        'max_wal_senders',
        'autovacuum_max_workers',
        'max_worker_processes',
        'max_parallel_workers',
        'max_connections',
    ]
    excluded = {'shared_buffers', 'huge_pages', 'shared_preload_libraries', 'wal_level'}

    selected = []
    for param_name in preferred_order:
        if param_name not in knobs_detail or param_name in excluded:
            continue
        metadata = db.get_parameter_info(param_name)
        if not metadata or metadata['context'] != 'postmaster':
            continue
        selected.append((param_name, metadata))
        if len(selected) >= limit:
            break
    return selected


def build_static_smoke_config(knobs_detail, selected_params, logger):
    """围绕当前值构造一组小扰动静态参数。"""
    smoke_config = {}
    original_config = {}

    for param_name, metadata in selected_params:
        knob_detail = knobs_detail.get(param_name, {})
        min_val = knob_detail.get('min', metadata.get('min_val'))
        max_val = knob_detail.get('max', metadata.get('max_val'))

        try:
            current_int = int(float(metadata['setting']))
            min_int = int(float(min_val)) if min_val not in (None, '') else current_int
            max_int = int(float(max_val)) if max_val not in (None, '') else current_int

            candidate = current_int + 1
            if candidate > max_int:
                candidate = max(current_int - 1, min_int)
            if candidate == current_int:
                logger.info(f"静态参数 {param_name} 已接近边界，跳过 smoke test")
                continue

            original_config[param_name] = current_int
            smoke_config[param_name] = candidate
        except Exception as exc:
            logger.warning(f"静态参数 {param_name} smoke config 构建失败: {exc}")

    return smoke_config, original_config


def run_static_parameter_smoke_test(config, logger):
    """
    小规模验证静态参数链路：
    ALTER SYSTEM -> restart -> verify -> restore -> restart -> verify
    """
    logger.info("\n[附加步骤] 开始静态参数 smoke test...\n")

    knob_config_file = config['tuning_config'].get('knob_config', 'knob_config/knob_config.json')
    knobs_detail = get_knobs(knob_config_file)
    db = Database(config['database_config'])
    parameter_subsystem = ParameterExecutionSubsystem.from_config(config, db, logger)

    try:
        selected_params = choose_static_smoke_parameters(
            db=db,
            knobs_detail=knobs_detail,
            limit=int(config['benchmark_config'].get('static_smoke_limit', 2)),
        )
        if not selected_params:
            logger.warning("没有找到适合 smoke test 的静态参数，跳过")
            return True

        smoke_config, original_config = build_static_smoke_config(knobs_detail, selected_params, logger)
        if not smoke_config:
            logger.warning("未构造出可执行的静态参数测试配置，跳过")
            return True

        logger.info(f"静态参数测试配置: {smoke_config}")
        logger.info(f"静态参数原始配置: {original_config}")

        apply_stats = parameter_subsystem.apply(
            smoke_config,
            apply_static=True,
            restart_if_static=True,
            force_new_session=True,
        )
        logger.info(f"静态参数应用结果: {apply_stats}")
        if not apply_stats.get('restarted'):
            logger.error("静态参数 smoke test 未完成重启，判定失败")
            return False

        restore_stats = parameter_subsystem.apply(
            original_config,
            apply_static=True,
            restart_if_static=True,
            force_new_session=True,
        )
        logger.info(f"静态参数恢复结果: {restore_stats}")
        if not restore_stats.get('restarted'):
            logger.error("静态参数恢复未完成重启，请手动检查实例状态")
            return False

        logger.info("静态参数 smoke test 完成")
        return True
    finally:
        db.close()


def run_all_parameter_validation(config, workload_files, logger):
    """
    对每个参数做一次单项验证，确保动态 / reload / restart 路径都能跑通。
    """
    logger.info("\n[附加步骤] 开始全参数单项验证...\n")

    knob_config_file = config['tuning_config'].get('knob_config', 'knob_config/knob_config.json')
    knobs_detail = get_knobs(knob_config_file)
    db = Database(config['database_config'])
    parameter_subsystem = ParameterExecutionSubsystem.from_config(config, db, logger)

    sample_dir = './offline_sample'
    os.makedirs(sample_dir, exist_ok=True)
    validation_sample_file = os.path.join(sample_dir, 'parameter_validation')
    reset_output_file(validation_sample_file + '.jsonl')

    tool = stress_testing_tool(
        config,
        db,
        logger,
        validation_sample_file,
        parameter_subsystem=parameter_subsystem,
    )
    if workload_files:
        tool.workload_file = workload_files[0]
        tool.benchmark_config['workload_path'] = workload_files[0]

    summary = {
        'tested': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'details': [],
    }

    try:
        for knob_name, knob_detail in knobs_detail.items():
            metadata = db.get_parameter_info(knob_name)
            if not metadata:
                summary['skipped'] += 1
                summary['details'].append({'parameter': knob_name, 'status': 'unknown'})
                continue

            candidate = build_single_knob_test_value(db, knob_name, knob_detail)
            if candidate is None:
                summary['skipped'] += 1
                summary['details'].append({'parameter': knob_name, 'status': 'no_candidate'})
                continue

            apply_static = metadata['context'] in {'sighup', 'postmaster'}
            restart_if_static = metadata['context'] == 'postmaster'

            logger.info(
                f"[validate] {knob_name}: context={metadata['context']}, candidate={candidate}, "
                f"apply_static={apply_static}, restart={restart_if_static}"
            )

            summary['tested'] += 1
            score = tool.test_config(
                {knob_name: candidate},
                apply_static=apply_static,
                restart_if_static=restart_if_static,
            )
            passed = score > 0
            if passed:
                summary['passed'] += 1
            else:
                summary['failed'] += 1

            summary['details'].append({
                'parameter': knob_name,
                'context': metadata['context'],
                'candidate': candidate,
                'score': score,
                'status': 'passed' if passed else 'failed',
            })
    finally:
        db.close()

    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, f"parameter_validation_{config['database_config']['database']}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"全参数单项验证完成: {summary}")
    logger.info(f"验证报告已保存: {report_path}")
    return summary['failed'] == 0


def reset_output_file(path):
    """重置输出文件，避免旧样本混入新结果。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8'):
        pass


def run_offline_sampling(config, workload_files, logger):
    """
    执行Phase 1：离线采样
    
    参数：
        config: 全局配置
        workload_files: workload文件列表
        logger: 日志记录器
    
    返回：
        str: offline_sample.jsonl文件路径
    """
    logger.info("\n[步骤 2/6] 执行离线采样 (Phase 1)...\n")
    
    workload_limit = int(config['benchmark_config'].get('workload_limit', 3))
    samples_per_workload = int(config['benchmark_config'].get('samples_per_workload', 5))
    resume_enabled = str(config['benchmark_config'].get('resume_sampling', 'false')).lower() == 'true'

    # 只处理前 N 个 workload 进行测试
    test_workloads = workload_files[:workload_limit]
    logger.info(f"本次将采样 {len(test_workloads)} 个workload（测试）：")
    for wf in test_workloads:
        logger.info(f"  - {os.path.basename(wf)}")
    
    # 初始化数据库连接
    db = Database(config['database_config'])
    
    # 初始化采样工具
    sample_dir = './offline_sample'
    os.makedirs(sample_dir, exist_ok=True)
    sample_file = os.path.join(sample_dir, 'offline_sample')
    metadata_file = os.path.join(sample_dir, 'sampling_metadata.jsonl')
    if not resume_enabled:
        reset_output_file(sample_file + '.jsonl')
        reset_output_file(metadata_file)

    parameter_subsystem = ParameterExecutionSubsystem.from_config(config, db, logger)
    recorder = SamplingRunRecorder(metadata_file, resume=resume_enabled)
    tool = stress_testing_tool(
        config,
        db,
        logger,
        sample_file,
        parameter_subsystem=parameter_subsystem,
    )
    
    # 简单的采样：每个workload测试3-5个随机参数配置
    logger.info("\n执行参数采样...")
    
    knob_config_file = config['tuning_config'].get('knob_config', 'knob_config/knob_config.json')
    knobs_detail = get_knobs(knob_config_file)
    
    sample_count = 0
    import random
    rng = random.Random(42)
    
    for workload_path in test_workloads:
        workload_name = os.path.splitext(os.path.basename(workload_path))[0]
        logger.info(f"\n采样 {workload_name}...")
        
        # 更新workload路径
        config['benchmark_config']['workload_path'] = workload_path
        tool.benchmark_config['workload_path'] = workload_path
        tool.workload_file = workload_path

        baseline_config = {}
        for knob_name, knob_detail in knobs_detail.items():
            if 'default' in knob_detail:
                baseline_config[knob_name] = knob_detail['default']

        try:
            logger.info("  [baseline] 测试默认参数配置...")
            baseline_key = recorder.build_sample_key(workload_name, 'baseline', baseline_config)
            if recorder.should_skip(baseline_key):
                logger.info("      baseline 已完成，resume 模式下跳过")
            else:
                performance = tool.test_config(
                    baseline_config,
                    sample_metadata={
                        'sample_key': baseline_key,
                        'workload_id': workload_name,
                        'sample_kind': 'baseline',
                        'feature_extraction_status': 'pending',
                    },
                )
                sample_count += 1
                recorder.record({
                    'sample_key': baseline_key,
                    'status': 'success',
                    'workload_id': workload_name,
                    'sample_kind': 'baseline',
                    'parameter_config': baseline_config,
                    'score': performance,
                    'feature_extraction_status': 'pending',
                    'safety_validation_result': parameter_subsystem.validate_config(baseline_config),
                })
                logger.info(f"      性能评分: {performance:.4f}")
        except Exception as e:
            logger.warning(f"      baseline 失败: {e}")
            recorder.record({
                'sample_key': recorder.build_sample_key(workload_name, 'baseline', baseline_config),
                'status': 'failed',
                'workload_id': workload_name,
                'sample_kind': 'baseline',
                'parameter_config': baseline_config,
                'score': 0.0,
                'feature_extraction_status': 'pending',
                'error': str(e),
                'safety_validation_result': parameter_subsystem.validate_config(baseline_config),
            })

        # 执行 N 次简单的参数采样
        for i in range(samples_per_workload):
            try:
                # 生成随机参数配置
                random_config = generate_safe_sample_config(
                    knobs_detail,
                    rng,
                    parameter_subsystem,
                )
                if not random_config:
                    logger.warning("      未能生成通过联合约束校验的随机配置，跳过本轮采样")
                    recorder.record({
                        'sample_key': f"{workload_name}:random:{i}",
                        'status': 'skipped',
                        'workload_id': workload_name,
                        'sample_kind': 'random',
                        'sample_index': i,
                        'reason': 'safety_validation_rejected',
                        'feature_extraction_status': 'pending',
                    })
                    continue
                
                # 测试这个配置
                logger.info(f"  [{i+1}/{samples_per_workload}] 测试参数配置...")
                sample_key = recorder.build_sample_key(workload_name, f'random-{i}', random_config)
                if recorder.should_skip(sample_key):
                    logger.info("      样本已完成，resume 模式下跳过")
                    continue

                safety_validation_result = parameter_subsystem.validate_config(random_config)
                performance = tool.test_config(
                    random_config,
                    sample_metadata={
                        'sample_key': sample_key,
                        'workload_id': workload_name,
                        'sample_kind': 'random',
                        'sample_index': i,
                        'feature_extraction_status': 'pending',
                        'safety_validation_result': safety_validation_result,
                    },
                )
                sample_count += 1
                logger.info(f"      性能评分: {performance:.4f}")
                recorder.record({
                    'sample_key': sample_key,
                    'status': 'success',
                    'workload_id': workload_name,
                    'sample_kind': 'random',
                    'sample_index': i,
                    'parameter_config': random_config,
                    'score': performance,
                    'feature_extraction_status': 'pending',
                    'safety_validation_result': safety_validation_result,
                })
                
            except Exception as e:
                logger.warning(f"      采样失败: {e}")
                recorder.record({
                    'sample_key': f"{workload_name}:random:{i}:failed",
                    'status': 'failed',
                    'workload_id': workload_name,
                    'sample_kind': 'random',
                    'sample_index': i,
                    'score': 0.0,
                    'feature_extraction_status': 'pending',
                    'error': str(e),
                })
                continue
    
    logger.info(f"\n✓ 离线采样完成，生成了 {sample_count} 个样本")
    
    # 检查生成的采样文件
    sample_jsonl = sample_file + '.jsonl'
    if os.path.exists(sample_jsonl):
        with open(sample_jsonl) as f:
            lines = f.readlines()
        logger.info(f"✓ 采样文件已保存: {sample_jsonl} ({len(lines)} 行)")
        return sample_jsonl
    else:
        logger.error(f"✗ 采样文件未生成: {sample_jsonl}")
        return None


def extract_features(sample_file, database_name, logger):
    """
    执行步骤3：提取工作负载特征
    """
    logger.info("\n[步骤 3/6] 提取工作负载特征向量...\n")
    
    if not os.path.exists(sample_file):
        logger.error(f"✗ 采样文件不存在: {sample_file}")
        return False
    
    logger.info(f"输入: {sample_file}")
    try:
        if extract_workload_features(sample_file, database_name):
            logger.info("✓ 特征向量提取成功")
            return True
        else:
            logger.error("✗ 特征向量提取失败")
            return False
    except Exception as e:
        logger.error(f"✗ 特征提取异常: {e}")
        return False


def build_sft_data(sample_file, database_name, logger):
    """
    执行步骤4：构建SFT训练数据
    """
    logger.info("\n[步骤 4/6] 构建SFT训练数据...\n")
    
    if not os.path.exists(sample_file):
        logger.error(f"✗ 采样文件不存在: {sample_file}")
        return False
    
    output_file = f'training_data/training_sft_data_{database_name}.jsonl'
    logger.info(f"输入: {sample_file}")
    logger.info(f"输出: {output_file}")
    
    try:
        if build_training_data(sample_file, output_file):
            logger.info("✓ SFT训练数据生成成功")
            
            # 显示数据文件统计
            if os.path.exists(output_file):
                with open(output_file) as f:
                    lines = f.readlines()
                file_size = os.path.getsize(output_file) / 1024 / 1024
                logger.info(f"✓ 数据统计: {len(lines)} 条样本, {file_size:.2f}MB")
            
            return True
        else:
            logger.error("✗ SFT训练数据生成失败")
            return False
    except Exception as e:
        logger.error(f"✗ SFT数据构建异常: {e}")
        return False


def generate_summary(logger):
    """
    生成Phase 1总结
    """
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1 完成总结")
    logger.info("=" * 70)
    
    logger.info("\n✓ 已生成的文件：")
    if os.path.exists('./offline_sample/offline_sample.jsonl'):
        size = os.path.getsize('./offline_sample/offline_sample.jsonl') / 1024
        logger.info(f"  1. offline_sample/offline_sample.jsonl ({size:.1f}KB)")
    
    if os.path.exists('./SuperWG/feature/tpch.json'):
        size = os.path.getsize('./SuperWG/feature/tpch.json') / 1024
        logger.info(f"  2. SuperWG/feature/tpch.json ({size:.1f}KB)")
    
    for f in os.listdir('./training_data') if os.path.exists('./training_data') else []:
        if f.endswith('.jsonl'):
            full_path = os.path.join('./training_data', f)
            size = os.path.getsize(full_path) / 1024 / 1024
            logger.info(f"  3. training_data/{f} ({size:.2f}MB)")
    
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1 已完成！")
    logger.info("=" * 70)
    logger.info("\n下一步：")
    logger.info("  使用生成的SFT训练数据微调Qwen2.5-7B模型")
    logger.info("  命令：python train_lora.py")
    logger.info("=" * 70 + "\n")


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="AutoDL云环境Phase 1快速启动")
    parser.add_argument('--config', type=str, default='config/config.ini', 
                       help='配置文件路径')
    parser.add_argument('--database', type=str, default='tpch',
                       help='数据库名称')
    parser.add_argument('--workload-dir', type=str, default='',
                       help='workload 目录，不传则优先使用 config 中的 workload_datapath')
    parser.add_argument('--test-static', action='store_true',
                       help='执行小规模静态参数 smoke test（会触发数据库重启）')
    parser.add_argument('--validate-all-knobs', action='store_true',
                       help='对所有参数做一次单项验证，覆盖动态与静态参数路径')
    parser.add_argument('--resume', action='store_true',
                       help='恢复已有采样任务，跳过已成功完成的样本')
    parser.add_argument('--workload-limit', type=int, default=3,
                       help='本轮最多处理多少个 workload')
    parser.add_argument('--samples-per-workload', type=int, default=5,
                       help='每个 workload 额外生成多少个随机样本')
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    try:
        # 1. 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = parse_args(args.config)
        workload_dir = args.workload_dir or config.get('database_config', {}).get('workload_datapath', '')
        config = build_phase1_config(
            base_config=config,
            workload_dir=workload_dir,
            database_name=args.database,
            sample_prefix=os.path.join('offline_sample', 'offline_sample'),
            workload_limit=args.workload_limit,
            samples_per_workload=args.samples_per_workload,
            resume_sampling=args.resume,
        )
        logger.info(f"✓ 配置加载成功")
        
        # 2. 验证环境
        if not verify_environment(config, logger):
            logger.error("\n✗ 环境验证失败，程序退出")
            return 1
        
        # 3. 发现workload
        workload_files = discover_workloads(config, logger)
        if not workload_files:
            logger.error("\n✗ 未发现workload文件")
            return 1
        
        # 4. 离线采样
        sample_file = run_offline_sampling(config, workload_files, logger)
        if not sample_file:
            logger.error("\n✗ 离线采样失败")
            return 1
        
        # 5. 特征提取
        if not extract_features(sample_file, args.database, logger):
            logger.warning("\n! 特征提取失败，继续...")
        
        # 6. 构建SFT数据
        if not build_sft_data(sample_file, args.database, logger):
            logger.warning("\n! SFT数据构建失败")
        
        # 7. 生成总结
        generate_summary(logger)

        if args.test_static:
            if not run_static_parameter_smoke_test(config, logger):
                logger.warning("\n! 静态参数 smoke test 失败，请检查日志")

        if args.validate_all_knobs:
            if not run_all_parameter_validation(config, workload_files, logger):
                logger.warning("\n! 全参数单项验证存在失败项，请检查报告")

        logger.info("\n✓ Phase 1 执行成功！\n")
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ 执行过程中发生异常: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
