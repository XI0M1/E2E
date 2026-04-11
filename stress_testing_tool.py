"""
压力测试工具模块（Stress Testing Tool）
功能：执行数据库工作负载并收集性能指标
"""

import time
import subprocess
import logging
import json
import re
import os
from typing import Dict, Any, List

from parameter_subsystem import ParameterExecutionSubsystem


class stress_testing_tool:
    """
    压力测试工具类
    
    职责：
        1. 执行数据库工作负载（如TPC-H、TPC-C、pgbench）
        2. 测试特定的参数配置
        3. 收集性能指标（TPS、响应时间、CPU、内存等）
        4. 返回性能评分用于优化
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        database,
        logger,
        sample_path: str,
        parameter_subsystem=None,
    ):
        """
        初始化压力测试工具
        
        参数：
            config (dict): 全局配置字典
            database: 数据库连接对象（Database实例）
            logger: 日志记录器
            sample_path (str): 采样数据保存路径
        """
        self.config = config
        self.database = database
        self.logger = logger
        self.sample_path = sample_path
        self.benchmark_config = config.get('benchmark_config', {})
        self.parameter_subsystem = parameter_subsystem
        self.timeout = int(self.benchmark_config.get('timeout', 300))
        self.fetch_result_rows = str(
            self.benchmark_config.get('fetch_result_rows', 'false')
        ).lower() == 'true'
        self.fresh_session_per_test = str(
            self.benchmark_config.get('fresh_session_per_test', 'true')
        ).lower() == 'true'
        
        # 新增：保存workload_file路径（用于后续提取SQL统计和执行计划）
        self.workload_file = self.benchmark_config.get('workload_path', '')
        if not os.path.exists(self.workload_file):
            self.logger.warning(f"Workload文件不存在: {self.workload_file}")
        if self.parameter_subsystem is None and self.database:
            self.parameter_subsystem = ParameterExecutionSubsystem.from_config(
                config,
                database,
                logger,
            )

    def test_config(
        self,
        config: Dict[str, Any],
        apply_static: bool = None,
        restart_if_static: bool = None,
        sample_metadata: Dict[str, Any] = None,
    ) -> float:
        """
        测试给定的参数配置
        
        流程：
            1. 应用参数配置到数据库（动态参数用 SET，静态参数可选 ALTER SYSTEM）
            2. 如需重启则重启数据库
            3. 预热缓存
            4. 运行工作负载
            5. 收集性能指标
            6. 保存结果到离线采样库
        
        参数：
            config (dict): 参数配置字典
                例如: {'shared_buffers': 1114632, 'work_mem': 716290752, ...}
            apply_static (bool): 是否应用静态参数（需要 ALTER SYSTEM）
            restart_if_static (bool): 如果应用了静态参数，是否自动重启数据库
        
        返回：
            float: 性能评分（通常是TPS，越高越好）
        """
        self.logger.info(f"开始测试配置 (apply_static={apply_static}, restart={restart_if_static})...")
        
        try:
            # 1. 应用参数配置
            config_stats = None
            if self.database:
                self.logger.info("正在应用参数配置...")
                config_stats = self.parameter_subsystem.apply(
                    config, 
                    apply_static=apply_static,
                    restart_if_static=restart_if_static,
                    force_new_session=self.fresh_session_per_test,
                )
                self.logger.info(
                    f"参数应用结果: 动态={config_stats['dynamic']}, "
                    f"静态={config_stats['static']}, reload={config_stats.get('reload', 0)}, "
                    f"跳过={config_stats['skipped']}, 失败={config_stats['failed']}, "
                    f"verified={config_stats.get('verified', 0)}, health_ok={config_stats.get('health_ok')}"
                )

                if config_stats.get('health_ok') is False:
                    self.logger.error(
                        f"参数应用后健康检查失败: rollback={config_stats.get('rollback')}"
                    )
                    return 0.0
                
                # 如果已重启，等待数据库完全恢复
                if config_stats['static'] > 0 and restart_if_static:
                    self.logger.info("数据库已重启，等待恢复...")
                    time.sleep(5)  # 给数据库一些额外时间来完全恢复
            
            # 2. 预热缓存
            self.logger.info("预热缓存中...")
            self._warmup()
            
            # 3. 执行测试工作负载
            self.logger.info("执行工作负载测试...")
            workload_path = self.benchmark_config.get('workload_path', '')
            performance = self._run_workload(workload_path)
            self.logger.info(f"工作负载完成，性能评分: {performance:.2f}")
            
            # 4. 收集系统指标
            metrics = self._collect_metrics(config, performance)
            
            # 5. 采集Query Plans（在当前参数配置下）
            query_plans = self._collect_query_plans()
            
            # 6. 保存结果
            sample_data = {
                'config': config,
                'performance': performance,
                'metrics': metrics,
                'query_plans': query_plans,
                'apply_static': apply_static,
                'restart_performed': config_stats.get('static', 0) > 0 and restart_if_static,
                'config_stats': config_stats,
                'sample_metadata': sample_metadata or {},
            }
            self._save_sample_enhanced(sample_data)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"测试失败: {e}", exc_info=True)
            return 0.0  # 失败返回0
    
    def _apply_config(self, config: Dict[str, Any]):
        """
        应用参数配置到数据库（已废弃，使用 database.apply_config() 代替）
        
        参数：
            config (dict): 参数配置字典
        """
        if not self.database:
            return
        
        try:
            # 使用新的智能参数设置方法
            stats = self.parameter_subsystem.apply(
                config,
                force_new_session=self.fresh_session_per_test,
            )
            self.logger.info(
                f"已应用参数: 动态={stats['dynamic']}, "
                f"静态={stats['static']}, reload={stats.get('reload', 0)}, 失败={stats['failed']}"
            )
            
        except Exception as e:
            self.logger.error(f"配置应用失败: {e}")
    
    def _warmup(self, iterations: int = 2):
        """
        预热缓存
        
        运行工作负载几次，使数据加载到缓存中，
        后续测试结果更稳定。
        
        参数：
            iterations (int): 预热迭代次数
        """
        try:
            workload_path = self.benchmark_config.get('workload_path', '')
            for i in range(iterations):
                self.logger.debug(f"预热迭代 {i+1}/{iterations}")
                self._run_workload(workload_path, timeout=60)
        except Exception as e:
            self.logger.warning(f"预热失败: {e}")
    
    def _run_workload(self, workload_path: str, timeout: int = None) -> float:
        """
        执行工作负载并返回性能评分
        
        支持的工作负载类型：
            - TPC-H: OLAP基准测试
            - TPC-C: OLTP基准测试
            - pgbench: PostgreSQL内置基准工具
            - 自定义SQL脚本
        
        参数：
            workload_path (str): 工作负载文件路径
            timeout (int): 执行超时时间（秒）
        
        返回：
            float: 工作负载执行的性能评分
        """
        if timeout is None:
            timeout = self.timeout
        
        try:
            self.logger.info(f"执行工作负载: {workload_path}")

            if not self.database:
                self.logger.warning("没有数据库连接，无法执行真实 workload")
                return 0.0

            sqls = self._load_sqls_from_file(workload_path)
            if not sqls:
                self.logger.warning(f"Workload 中没有可执行 SQL: {workload_path}")
                return 0.0

            start_time = time.time()
            executed_sqls = 0

            for sql in sqls:
                if time.time() - start_time > timeout:
                    raise subprocess.TimeoutExpired(cmd=workload_path, timeout=timeout)
                self._execute_sql(sql)
                executed_sqls += 1

            elapsed_time = time.time() - start_time
            if elapsed_time <= 0:
                elapsed_time = 1e-6

            tps = executed_sqls / elapsed_time
            self.logger.info(
                f"Workload 执行完成: sqls={executed_sqls}, elapsed={elapsed_time:.3f}s, score={tps:.3f}"
            )
            return max(tps, 1e-6)
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"工作负载执行超时（{timeout}s）")
            return 0.0
        except Exception as e:
            self.logger.error(f"工作负载执行失败: {e}")
            return 0.0

    def _execute_sql(self, sql: str):
        """执行单条 SQL，兼容返回结果和无返回结果的语句。"""
        statement = sql.strip().rstrip(';')
        if not statement:
            return

        cursor = self.database.cursor
        cursor.execute(statement)

        if cursor.description is not None:
            try:
                if self.fetch_result_rows:
                    cursor.fetchall()
                else:
                    cursor.fetchmany(1)
            except Exception:
                pass
    
    def _collect_query_plans(self) -> str:
        """
        采集Query Plans（EXPLAIN输出）
        
        流程：
            1. 从workload_file读取SQL语句
            2. 对每条SQL执行EXPLAIN (FORMAT TEXT)
            3. 汇总所有执行计划
        
        返回：
            str: 所有SQL的EXPLAIN结果（按顺序拼接）
        """
        if not self.workload_file or not os.path.exists(self.workload_file):
            self.logger.debug("Workload文件不存在，跳过Query Plans采集")
            return ""
        
        if not self.database:
            return ""
        
        try:
            plans = []
            sqls = self._load_sqls_from_file(self.workload_file)
            
            self.logger.debug(f"从{self.workload_file}中读取了{len(sqls)}条SQL")
            
            for i, sql in enumerate(sqls[:10], 1):  # 最多采集前10条SQL的执行计划
                try:
                    # 对每条SQL执行EXPLAIN
                    explain_sql = f"EXPLAIN (FORMAT TEXT) {sql}"
                    result = self.database.execute_query(explain_sql)
                    
                    # 格式化执行计划
                    plan_text = '\n'.join([row[0] for row in result])
                    plans.append(f"=== SQL {i} ===\n{sql}\n--- PLAN ---\n{plan_text}\n")
                    
                except Exception as e:
                    self.logger.debug(f"执行计划采集失败 (SQL{i}): {e}")
                    plans.append(f"=== SQL {i} ===\n{sql}\n--- ERROR: {str(e)} ---\n")
            
            query_plans_text = '\n'.join(plans)
            self.logger.debug(f"Query Plans采集完成，大小: {len(query_plans_text)} bytes")
            return query_plans_text
            
        except Exception as e:
            self.logger.warning(f"Query Plans采集失败: {e}")
            return ""
    
    def _load_sqls_from_file(self, workload_file: str) -> List[str]:
        """
        从workload文件中加载SQL语句
        
        支持的格式：
            - 每行一条SQL语句
            - 以分号;分隔的SQL语句
            - 注释行（以--或#开头）被忽略
        
        参数：
            workload_file (str): workload文件路径
        
        返回：
            list: SQL语句列表
        """
        sqls = []
        try:
            with open(workload_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 移除注释（以--或#开头的行）
            lines = content.split('\n')
            cleaned_lines = [
                line for line in lines 
                if not line.strip().startswith('--') and 
                   not line.strip().startswith('#')
            ]
            content = '\n'.join(cleaned_lines)
            
            # 按分号分割SQL语句
            raw_sqls = content.split(';')
            
            for sql in raw_sqls:
                sql = sql.strip()
                if sql:  # 去掉空语句
                    sqls.append(sql)
            
            self.logger.debug(f"从{workload_file}解析出{len(sqls)}条SQL")
            return sqls
            
        except Exception as e:
            self.logger.error(f"解析SQL文件失败: {e}")
            return []
    
    def _collect_metrics(self, config: Dict[str, Any], performance: float) -> Dict[str, Any]:
        """
        收集系统和查询的内部性能指标
        
        参数：
            config (dict): 当前参数配置
            performance (float): 工作负载性能评分
        
        返回：
            dict: 包含各种性能指标的字典
        """
        metrics = {
            'timestamp': time.time(),
            'performance_score': performance,
        }
        
        # 若数据库连接可用，收集额外指标
        if self.database:
            try:
                db_metrics = self.database.get_system_metrics()
                metrics.update(db_metrics)
            except Exception as e:
                self.logger.warning(f"无法收集数据库指标: {e}")
        
        return metrics
    
    def _save_sample(self, config: Dict[str, Any], performance: float, 
                      metrics: Dict[str, Any], query_plans: str = ""):
        """
        保存测试样本到离线采样库
        
        采样格式（JSONL，新增workload_file和query_plans字段）：
        {
            'workload': 'tpch_1',
            'workload_file': 'tpch_1.wg',
            'config': {...},
            'tps': 150.5,
            'inner_metrics': {...},
            'query_plans': '执行计划文本',
            'y': -150.5
        }
        
        参数：
            config (dict): 参数配置
            performance (float): 性能评分
            metrics (dict): 系统指标
            query_plans (str): SQL执行计划文本
        """
        try:
            sample_data = {
                'workload': self.benchmark_config.get('workload_path', 'unknown'),
                'workload_file': os.path.basename(self.workload_file),  # 新增
                'config': config,
                'tps': performance,
                'inner_metrics': metrics,
                'query_plans': query_plans,  # 新增
                'y': -performance  # 用于最小化优化
            }
            
            # 追加到JSONL文件
            sample_file = self.sample_path + '.jsonl'
            with open(sample_file, 'a') as f:
                f.write(json.dumps(sample_data) + '\n')
            
            self.logger.debug(f"样本已保存: {sample_file}")
            
        except Exception as e:
            self.logger.warning(f"保存采样数据失败: {e}")
    
    def _save_sample_enhanced(self, sample_data: Dict[str, Any]):
        """
        保存完整的测试样本（包括重启信息）
        
        采样格式（JSONL）：
        {
            'config': {...},
            'performance': 150.5,
            'metrics': {...},
            'query_plans': '执行计划',
            'apply_static': false,
            'restart_performed': false,
            'workload': 'tpch_1',
            'workload_file': 'tpch_1.wg',
            'y': -150.5,
            'tps': 150.5,
            'inner_metrics': {...}
        }
        
        参数：
            sample_data (dict): 完整的样本数据
        """
        try:
            # 统一格式，兼容旧版和新版
            config = sample_data.get('config', {})
            performance = sample_data.get('performance', 0)
            metrics = sample_data.get('metrics', {})
            query_plans = sample_data.get('query_plans', '')
            apply_static = sample_data.get('apply_static', False)
            restart_performed = sample_data.get('restart_performed', False)
            config_stats = sample_data.get('config_stats', {})
            sample_metadata = sample_data.get('sample_metadata', {})
            
            output_data = {
                'config': config,
                'performance': performance,
                'metrics': metrics,
                'query_plans': query_plans,
                'apply_static': apply_static,
                'restart_performed': restart_performed,
                'config_stats': config_stats,
                'sample_metadata': sample_metadata,
                'workload': self.benchmark_config.get('workload_path', 'unknown'),
                'workload_file': os.path.basename(self.workload_file),
                'y': -performance,  # 用于最小化优化
                'tps': performance,
                'inner_metrics': metrics
            }
            
            # 追加到JSONL文件
            sample_file = self.sample_path + '.jsonl'
            with open(sample_file, 'a') as f:
                f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            
            self.logger.debug(
                f"样本已保存: {sample_file} "
                f"(static={'是' if apply_static else '否'}, "
                f"restart={'是' if restart_performed else '否'})"
            )
            
        except Exception as e:
            self.logger.warning(f"保存采样数据失败: {e}")
