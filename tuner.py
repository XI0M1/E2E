"""
调优器模块（Tuner Module）
功能：核心的数据库参数优化算法实现

核心算法：SMAC (Sequential Model-Based Algorithm Configuration)
- 使用贝叶斯优化来搜索参数配置空间
- 通过学习代理模型(随机森林)逐步改进配置
- 在有限的评估次数内找到接近最优的参数

关键特性：
1. 工作负载映射：找到相似工作负载以重用历史数据
2. 预热策略：支持多种初始化方式（ours, pilot, workload_map等）
3. 安全约束：确保生成的配置在有效范围内
4. 实时采样：边优化边保存数据用于代理模型训练
"""

import os
import csv
import pickle
import sys
import json
import copy

# 超采样和实验设计库
from pyDOE import lhs

# 本地模块导入
from knob_config import parse_knob_config
import utils
import numpy as np
import pandas as pd
import jsonlines
import random

# 数据库和工具模块
from Database import Database
from Vectorlib import VectorLibrary
from stress_testing_tool import stress_testing_tool
from safe.subspace_adaptation import Safe

# SMAC优化框架相关导入
from poap.controller import BasicWorkerThread, ThreadController
from pySOT.experimental_design import LatinHypercube
from pySOT import strategy, surrogate
from smac.configspace import ConfigurationSpace
from smac.runhistory.runhistory import RunHistory
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter


# ==================== 常数定义 ====================

# TPC-H工作负载的默认参数配置
# 这是一个参考配置，用于pilot预热方法：基于这个配置添加噪声生成初始点
tpch_origin = {"max_wal_senders": 21, "autovacuum_max_workers": 126, "max_connections": 860, "wal_buffers": 86880, "shared_buffers": 1114632, "autovacuum_analyze_scale_factor": 78, "autovacuum_analyze_threshold": 1202647040, "autovacuum_naptime": 101527, "autovacuum_vacuum_cost_delay": 45, "autovacuum_vacuum_cost_limit": 1114, "autovacuum_vacuum_scale_factor": 31, "autovacuum_vacuum_threshold": 1280907392, "backend_flush_after": 172, "bgwriter_delay": 5313, "bgwriter_flush_after": 217, "bgwriter_lru_maxpages": 47, "bgwriter_lru_multiplier": 4, "checkpoint_completion_target": 1, "checkpoint_flush_after": 44, "checkpoint_timeout": 758, "commit_delay": 22825, "commit_siblings": 130, "cursor_tuple_fraction": 1, "deadlock_timeout": 885378880, "default_statistics_target": 5304, "effective_cache_size": 1581112576, "effective_io_concurrency": 556, "from_collapse_limit": 407846592, "geqo_effort": 3, "geqo_generations": 1279335040, "geqo_pool_size": 838207872, "geqo_seed": 0, "geqo_threshold": 1336191360, "join_collapse_limit": 1755487872, "maintenance_work_mem": 1634907776, "temp_buffers": 704544576, "temp_file_limit": -1, "vacuum_cost_delay": 46, "vacuum_cost_limit": 5084, "vacuum_cost_page_dirty": 6633, "vacuum_cost_page_hit": 6940, "vacuum_cost_page_miss": 9381, "wal_writer_delay": 4773, "work_mem": 716290752}


def add_noise(knobs_detail, origin_config, range):
    """
    向参数配置添加随机噪声
    
    用途：pilot预热策略使用此函数基于已知的好配置添加扰动以生成多样化的初始点
    
    参数：
        knobs_detail (dict): 参数详细信息，包含每个参数的min/max值
        origin_config (dict): 原始参数配置
        range (float): 噪声范围百分比（0-1），例如0.05表示±5%相对于参数范围
    
    返回：
        dict: 添加随机噪声后的新配置，所有值都在合法范围内
    
    原理：
        对于每个参数，在其范围的±range%内生成随机扰动，确保最终值不超出边界
        这样可以在已知的优势点周围进行局部探索，加快收敛
    """
    new_config = copy.deepcopy(origin_config)
    
    for knob in knobs_detail:
        detail = knobs_detail[knob]
        rb = detail['max']  # 参数上界
        lb = detail['min']  # 参数下界
        
        # 参数范围太小则跳过噪声添加
        if rb - lb <= 1:
            continue
        
        # 计算噪声幅度：范围的±range%
        length = int((rb - lb) * range * 0.5)
        noise = random.randint(-length, length)
        
        # 应用噪声并严格限制在合法范围内
        tmp = origin_config[knob] + noise 
        if tmp < lb: 
            tmp = lb
        elif tmp > rb: 
            tmp = rb
        new_config[knob] = tmp
    
    print(new_config)
    return new_config




class tuner:
    """
    数据库参数自动调优器核心类
    
    ==== 核心职责 ====
    1. 初始化调优环境（数据库连接、参数空间、日志）
    2. 管理参数空间和约束
    3. 运行参数优化算法（SMAC）
    4. 记录采样数据和历史结果
    5. 应用安全约束检查
    
    ==== 优化流程 ====
    初始化 → 参数空间定义 → 默认配置测试 → 工作负载映射 → 
    SMAC迭代优化 → 最优配置保存
    
    ==== 属性说明 ====
    database: PostgreSQL数据库连接对象（若tool='surrogate'则为None）
    method: 调优方法名称（如'SMAC'）
    warmup: 预热策略（决定初始点选择）
        - 'ours': 使用模型预生成的参数
        - 'pilot': 基于已知优秀配置的扰动
        - 'workload_map': 使用3个最相似工作负载的历史数据
        - 'rgpe': 使用10个最相似工作负载的历史数据
    knobs_detail: 所有参数的定义（包含type, min, max, default等）
    pre_safe: 前验安全模型（用于约束搜索空间）
    post_safe: 后验安全模型（用于验证生成的配置）
    veclib: 工作负载向量库
    stt: 压力测试工具实例
    """
    
    def __init__(self, config):
        """
        初始化调优器
        
        初始化流程：
        1. 建立数据库连接（非代理模式）
        2. 加载配置参数和参数定义
        3. 初始化压力测试工具
        4. 加载工作负载特征向量
        5. 根据预热策略准备初始数据
        6. 初始化安全约束框架
        
        参数：
            config (dict): 全局配置字典，包含以下关键sections:
                benchmark_config: 
                    - tool: 'direct'(数据库) 或 'surrogate'(代理模型)
                    - workload_path: 工作负载文件路径
                tuning_config:
                    - tuning_method: 'SMAC'
                    - warmup_method: 预热策略
                    - sample_num: 初始采样数量
                    - suggest_num: 优化迭代数
                    - knob_config: 参数定义文件路径
                    - log_path: 日志文件路径
                ssh_config, database_config: 连接信息
        
        异常：
            FileNotFoundError: 参数定义文件或特征文件不存在
            psycopg2.Error: 数据库连接失败
        """
        
        # ==================== 第一步：初始化数据库 ====================
        # 决定是否需要真实数据库连接
        # 使用代理模型时（tool='surrogate'）不需要真实DB，仅进行推理
        if config['benchmark_config']['tool'] != 'surrogate':
            # 直接方法：需要连接真实数据库
            self.database = Database(config, config['tuning_config']['knob_config'])
            print(f"已连接到数据库: {config['database_config']['database']}")
        else: 
            # 代理模型方法：不需要数据库连接
            self.database = None
            print(f"使用代理模型，无需数据库连接")
        
        # ==================== 第二步：加载配置参数 ====================
        # 调优配置参数
        self.method = config['tuning_config']['tuning_method']      # 通常为'SMAC'
        self.warmup = config['tuning_config']['warmup_method']      # 预热策略
        self.online = config['tuning_config']['online']             # 'true'或'false'
        self.online_sample = config['tuning_config']['online_sample']
        self.offline_sample = config['tuning_config']['offline_sample']
        self.finetune_sample = config['tuning_config']['finetune_sample']
        self.inner_metric_sample = config['tuning_config']['inner_metric_sample']
        
        # 优化参数
        self.sampling_number = int(config['tuning_config']['sample_num'])  # 初始采样数
        self.iteration = int(config['tuning_config']['suggest_num'])       # SMAC迭代数
        
        # ==================== 第三步：加载参数定义 ====================
        # 从knob_config.json加载所有可优化参数的定义
        # 包含每个参数的type(integer/float/enum), min, max, default, step等
        self.knobs_detail = parse_knob_config.get_knobs(
            config['tuning_config']['knob_config']
        )
        print(f"已加载{len(self.knobs_detail)}个可优化参数")
        
        # ==================== 第四步：初始化日志和连接信息 ====================
        # 创建日志文件用于记录优化过程
        self.logger = utils.get_logger(config['tuning_config']['log_path'])
        self.logger.info(f"开始调优: 方法={self.method}, 预热={self.warmup}")
        
        # SSH连接信息（用于远程执行命令）
        self.ssh_host = config['ssh_config']['host']
        self.last_point = []  # 上一个评估的配置
        
        # ==================== 第五步：初始化压力测试工具 ====================
        # 压力测试工具负责执行工作负载并收集性能指标
        if self.online == 'false':
            # 离线采样模式：保存到offline_sample_*.jsonl
            self.stt = stress_testing_tool(
                config, self.database, self.logger, self.offline_sample
            )
        else:
            # 在线（微调）模式：保存到finetune_sample
            self.stt = stress_testing_tool(
                config, self.database, self.logger, self.finetune_sample
            )

        # ==================== 第六步：初始化安全约束相关 ====================
        # 安全约束用于确保生成的参数配置在合法和安全的范围内
        self.pre_safe = None    # 前验安全模型：约束搜索空间
        self.post_safe = None   # 后验安全模型：验证生成配置
        
        # ==================== 第七步：加载工作负载向量库 ====================
        # 用于工作负载相似性计算和工作负载映射
        self.veclib = VectorLibrary(config['database_config']['database'])
        
        # 尝试加载工作负载特征向量（用于工作负载映射）
        feature_path = f"SuperWG/feature/{config['database_config']['database']}.json"
        try:
            with open(feature_path, 'r') as f:
                features = json.load(f)
            self.logger.info(f"已加载{len(features)}个工作负载的特征向量")
        except FileNotFoundError:
            self.logger.warning(f"特征文件不存在: {feature_path}")
            features = {}
        
        # 当前工作负载ID
        self.wl_id = config['benchmark_config']['workload_path']
        
        # ==================== 第八步：工作负载映射 ====================
        # 根据预热策略选择是否使用相似工作负载的历史数据
        # 这可以加速当前工作负载的优化
        
        if self.warmup == 'workload_map' and self.wl_id in features:
            # workload_map: 使用3个最相似的工作负载
            self.feature = features[self.wl_id]
            self.rh_data, self.matched_wl = self.workload_mapper(
                config['database_config']['database'], k=3
            )
            self.logger.info(f"找到{len(self.rh_data)}条相似工作负载的历史数据")
            
        elif self.warmup == 'rgpe' and self.wl_id in features:
            # rgpe: 使用10个最相似的工作负载（更多样化）
            self.feature = features[self.wl_id]
            self.rh_data, self.matched_wl = self.workload_mapper(
                config['database_config']['database'], k=10
            )
            self.logger.info(f"找到{len(self.rh_data)}条相似工作负载的历史数据(RGPE)")

        # ==================== 第九步：初始化安全模型 ====================
        # 评估默认参数并初始化安全约束框架
        self.init_safe()
        self.logger.info("调优器初始化完成")

    def workload_mapper(self, database, k):
        matched_wls = self.veclib.find_most_similar(self.feature, k)
        rh_data = []
        keys_to_remove = ["tps", "y", "inner_metrics", "workload"]
        for wl in matched_wls:
            if len(rh_data) > 50:
                break
            if wl == self.wl_id:
                continue
            with jsonlines.open(f'offline_sample/offline_sample_{database}.jsonl') as f:
                for line in f:
                    if line['workload'] == wl:
                        filtered_config = {key: line[key] for key in line if key not in keys_to_remove}
                        rh_data.append({'config': filtered_config, 'tps': line['tps']})
        for wl in matched_wls:
            if wl != self.wl_id:
                best_wl = wl
                break
        return rh_data, best_wl

    def init_safe(self):
        """
        初始化安全约束框架
        
        ==== 核心功能 ====
        1. 清理旧的采样数据文件
        2. 收集参数的搜索空间范围
        3. 选择和测试默认配置
        4. 初始化前验和后验安全模型
        5. 进行缓存预热
        
        ==== 工作流程 ====
        清理数据 → 收集边界 → 选择初始点 → 测试初始点 → 建立安全框架 → 缓存预热
        
        ==== 预热策略说明 ====
        - 'ours': 使用生成式模型预生成的参数（来自model_config.json）
        - 'pilot': 基于tpch_origin配置添加±5%的随机扰动
        - 其他: 使用参数的默认值
        
        ==== 输出 ====
        初始化两个安全模型：
        - pre_safe: 在优化前约束搜索空间
        - post_safe: 在参数生成后进行验证
        """
        # ==================== 第一步：清理旧数据 ====================
        # 删除上一次调优的临时文件，确保每次调优从干净状态开始
        
        if os.path.exists(self.inner_metric_sample):
            with open(self.inner_metric_sample, 'r+') as f:
                f.truncate(0)
        else:
            file = open(self.inner_metric_sample, 'w')
            file.close()
            
        if os.path.exists(self.offline_sample):
            with open(self.offline_sample, 'r+') as f:
                f.truncate(0)
        else:
            file = open(self.offline_sample, 'w')
            file.close()
            
        if not os.path.exists(self.offline_sample + '.jsonl'):
            file = open(self.offline_sample + '.jsonl', 'w')
            file.close()

        # ==================== 第二步：收集参数空间信息 ====================
        # 从参数定义中提取每个参数的上界、下界和步长
        # 这些信息用于SMAC优化器定义搜索空间
        
        step = []  # 参数步长列表
        lb, ub = [], []  # 下界、上界列表
        knob_default = {}  # 默认参数配置

        for index, knob in enumerate(self.knobs_detail):
            detail = self.knobs_detail[knob]
            
            if detail['type'] in ['integer', 'float']:
                # 数值参数：直接使用min/max
                lb.append(detail['min'])
                ub.append(detail['max'])
            elif detail['type'] == 'enum':
                # 枚举参数：将其转码为0-N的整数范围
                lb.append(0)
                ub.append(len(detail['enum_values']) - 1)
            
            # 记录默认值和步长
            knob_default[knob] = detail['default']
            step.append(detail['step'])

        # ==================== 第三步：选择默认配置或初始点 ====================
        # 根据预热策略选择不同的初始配置
        
        if self.warmup == 'ours':
            # 'ours'策略：使用模型预生成的参数
            # 这假设已有model_config.json文件包含每个工作负载的推荐参数
            try:
                model_config = json.load(open('model_config.json'))
                workload = self.wl_id.split('SuperWG/res/gpt_workloads/')[1]
                knob_default = model_config[workload]
                self.logger.info(f"使用模型预生成的参数作为初始点")
            except (FileNotFoundError, KeyError) as e:
                self.logger.warning(f"无法加载模型参数: {e}，使用默认值")
                
        elif self.warmup == 'pilot':
            # 'pilot'策略：基于已知优秀配置的随机扰动
            # 在tpch_origin周围添加±5%的噪声生成初始点
            origin_config = tpch_origin
            knob_default = add_noise(self.knobs_detail, origin_config, 0.05)
            self.logger.info(f"使用pilot策略生成初始点")

        # ==================== 第四步：测试初始配置 ====================
        # 评估选定的初始配置以确定基线性能
        
        print('测试初始参数配置性能...')
        print(knob_default)
        
        default_performance = self.stt.test_config(knob_default)
        
        print(f'初始性能评分: {default_performance}')
        self.logger.info(f"初始配置性能: {default_performance}")

        # ==================== 第五步：初始化安全约束框架 ====================
        # 创建前验安全模型，用于在优化过程中约束参数搜索空间
        
        self.pre_safe = Safe(
            default_performance,    # 默认配置的性能
            knob_default,          # 默认参数配置
            default_performance,   # 当前最佳性能
            lb,                    # 参数下界
            ub,                    # 参数上界
            step                   # 参数步长
        )
        
        # 加载后验安全模型（若存在）
        # 后验安全模型用于在参数生成后进行验证和约束
        try:
            with open('safe/predictor.pickle', 'rb') as f:
                self.post_safe = pickle.load(f)
                self.logger.info("已加载后验安全模型")
        except FileNotFoundError:
            self.logger.warning("后验安全模型不存在，将跳过后验检查")
            self.post_safe = None

        # ==================== 第六步：缓存预热 ====================
        # 运行4次初始配置以充分预热数据库缓存
        # 确保后续性能测试得到稳定的结果
        
        for i in range(4):
            self.logger.debug(f"缓存预热第{i+1}/4次")
            self.stt.test_config(knob_default)
        
        # 记录初始点用于参考
        self.last_point = list(knob_default.values())
        
        # 可选：训练后验安全模型（需要大量历史数据）
        # self.post_safe.train(data_path='./')

    def tune(self):
        """
        调优主入口
        
        根据配置的方法调用相应的优化算法
        
        参数：
            无
        
        返回：
            无（结果保存到文件）
        """
        if self.method == 'SMAC':
            self.SMAC()
        # 可以在这里添加其他优化方法
        # elif self.method == 'BOHB':
        #     self.BOHB()

    def SMAC(self):
        """
        SMAC (Sequential Model-Based Algorithm Configuration) 优化器
        
        ==== 算法原理 ====
        1. 贝叶斯优化：使用高斯过程或随机森林建模性能函数
        2. 获取函数：选择最有可能改进性能的下一个候选配置
        3. 迭代：评估候选配置，更新模型，重复直到收敛或迭代限制
        
        ==== 关键步骤 ====
        1. 定义参数配置空间（ConfigurationSpace）
        2. 初始化SMAC优化器
        3. 迭代运行：
           - SMAC生成候选配置
           - 测试配置的性能
           - SMAC更新内部模型
        4. 保存最优配置和历史记录
        
        ==== 输出 ====
        - incumbent: 找到的最佳配置
        - runhistory: 所有评估的历史记录
        - 结果保存到smac_his/{workload}_{warmup}.json
        
        ==== 性能指标 ====
        优化目标是最大化TPS（Transaction Per Second）
        SMAC内部最小化负TPS（-TPS）
        """
        
        print("开始SMAC优化过程...")
        self.logger.info(f"===== SMAC优化开始 =====")
        self.logger.info(f"样本数: {self.sampling_number}, 迭代数: {self.iteration}")
        
        # ==================== 第一步：定义性能目标函数 ====================
        # 这个函数会被SMAC反复调用来评估候选配置的性能
        
        def get_neg_result(point):
            """
            内部目标函数：测试一个参数配置
            
            参数：
                point: ConfigurationSpace中的一个配置对象
            
            返回：
                float: 负的性能评分（SMAC最小化此值等价于最大化性能）
            """
            # 测试这个配置
            y = self.stt.test_config(point)
            
            # SMAC最小化目标，所以如果我们要最大化TPS，返回-TPS
            result = -y
            
            self.logger.debug(f"评估配置，性能={y:.2f}")
            return result
        
        # ==================== 第二步：定义配置空间 ====================
        # ConfigurationSpace定义了所有参数及其范围，SMAC在此空间内搜索
        
        cs = ConfigurationSpace()
        self.logger.info(f"定义配置空间: {len(self.knobs_detail)}个参数")
        
        for name in self.knobs_detail.keys():
            detail = self.knobs_detail[name]
            
            if detail['type'] == 'integer':
                # 整数参数：UniformIntegerHyperparameter
                # 处理边界情况：min==max时添加1避免错误
                min_val = detail['min']
                max_val = detail['max']
                if max_val == min_val: 
                    max_val += 1
                
                knob = UniformIntegerHyperparameter(
                    name, min_val, max_val, 
                    default_value=detail['default']
                )
                
            elif detail['type'] == 'float':
                # 浮点参数：UniformFloatHyperparameter
                knob = UniformFloatHyperparameter(
                    name, detail['min'], detail['max'],
                    default_value=detail['default']
                )
            
            # 注：枚举参数在后续版本中可以通过CategoricalHyperparameter支持
            cs.add_hyperparameter(knob)

        # ==================== 第三步：初始化SMAC优化器 ====================
        # 配置SMAC的运行参数
        
        runhistory = RunHistory()
        
        # 根据预热策略加载历史数据
        # （当前代码中为空，可扩展为加载相似工作负载的数据）
        if self.warmup == 'workload_map' or self.warmup == 'rgpe':
            # 可以在这里添加历史数据以加速优化
            # for line in self.rh_data:
            #     config = cs.get_default_configuration()
            #     # 导入配置数据...
            pass
        
        # 生成输出目录名
        save_workload = self.wl_id.split('SuperWG/res/gpt_workloads/')[1]
        save_workload = save_workload.split('.wg')[0]
        
        # 对于RGPE策略，使用匹配的工作负载作为参考
        if self.warmup == 'rgpe':
            matched_workload = self.matched_wl.split('SuperWG/res/gpt_workloads/')[1]
            matched_workload = matched_workload.split('.wg')[0]
            output_dir = f"./{matched_workload}_smac_output"
            model_dir = f"./models/{save_workload}"
        else:
            output_dir = f"./{save_workload}_smac_output"
            model_dir = f"./models/{save_workload}"
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建SMAC场景（optimization scenario）
        scenario = Scenario({
            "run_obj": "quality",           # 优化目标质量（而不是时间）
            "runcount-limit": self.iteration,  # 最大评估次数
            "cs": cs,                       # 配置空间
            "deterministic": "true",        # 问题确定性（可重现）
            "output_dir": output_dir,       # 输出目录
            "save_model": "true",           # 保存优化模型
            "local_results_path": model_dir # 本地结果路径
        })
        
        # 创建SMAC优化器
        smac = SMAC4HPO(
            scenario=scenario,
            rng=np.random.RandomState(42),  # 随机种子保证可重现性
            tae_runner=get_neg_result,       # 目标函数
            runhistory=runhistory           # 运行历史
        )
        
        # ==================== 第四步：运行优化 ====================
        # SMAC迭代优化，在配置空间中搜索最优参数
        
        self.logger.info("开始迭代优化...")
        incumbent = smac.optimize()  # 运行优化，返回最佳配置
        self.logger.info("优化完成")
        
        print('SMAC优化完成')
        print(f'最优配置类型: {type(incumbent)}')
        print(f'最优配置: {incumbent}')
        
        # ==================== 第五步：提取和保存结果 ====================
        # 处理优化结果
        
        runhistory = smac.runhistory
        self.logger.info(f"总共评估了{len(runhistory.data)}个配置")
        
        # 将运行历史转换为JSON格式保存
        def runhistory_to_json(runhistory):
            """将SMAC运行历史转换为JSON格式"""
            data_to_save = {}
            for run_key in runhistory.data.keys():
                config_id, instance_id, seed, budget = run_key
                run_value = runhistory.data[run_key]
                data_to_save[str(run_key)] = {
                    "cost": run_value.cost,
                    "time": run_value.time,
                    "status": run_value.status.name if hasattr(run_value.status, 'name') else str(run_value.status),
                    "additional_info": run_value.additional_info
                }
            return json.dumps(data_to_save, indent=4)

        # 保存结果
        os.makedirs('smac_his', exist_ok=True)
        result_file = f"smac_his/{save_workload}_{self.warmup}.json"
        
        with open(result_file, "w") as f:
            f.write(runhistory_to_json(runhistory))
        
        self.logger.info(f"优化结果已保存到: {result_file}")
        
        self.logger.info(f"===== SMAC优化完成 =====")
        print(f"优化结果已保存到: {result_file}")

        def get_neg_result(point):
            y = self.stt.test_config(point)
            result = -y
            # evaluation_results.append([point, result])
            # print(result)
            return result
        
        cs = ConfigurationSpace()
        print('begin')
        for name in self.knobs_detail.keys():
            detail = self.knobs_detail[name]
            if detail['type'] == 'integer':
                if detail['max'] == detail['min']: detail['max'] += 1
                knob = UniformIntegerHyperparameter(name, detail['min'],\
                                                     detail['max'], default_value=detail['default'])
            elif detail['type'] == 'float':
                knob = UniformFloatHyperparameter(name, detail['min'],\
                                                     detail['max'], default_value=detail['default'])
            cs.add_hyperparameter(knob)

        runhistory = RunHistory()
        if self.warmup == 'workload_map' or self.warmup == 'rgpe':
            for line in self.rh_data:
                continue
                # empty_config = cs.sample_configuration()
                # config = empty_config.import_values(line['config'])
                # config = cs.get_default_configuration().new_configuration(line['config'])
                # runhistory.add(config=config, cost=line['tps'], time=line['tps']*10)
        
        save_workload = self.wl_id.split('SuperWG/res/gpt_workloads/')[1]
        save_workload = save_workload.split('.wg')[0]
        if self.warmup == 'rgpe':
            matched_workload = self.matched_wl.split('SuperWG/res/gpt_workloads/')[1]
            matched_workload = matched_workload.split('.wg')[0]
            scenario = Scenario({"run_obj": "quality",   # {runtime,quality}
                            "runcount-limit": 75,   # max. number of function evaluations; for this example set to a low number
                            "cs": cs,               # configuration space
                            "deterministic": "true",
                            "output_dir": f"./{matched_workload}_smac_output",  
                            "save_model": "true",
                            "local_results_path": f"./models/{save_workload}"
                            })
        else:
            scenario = Scenario({"run_obj": "quality",   # {runtime,quality}
                            "runcount-limit": 75,   # max. number of function evaluations; for this example set to a low number
                            "cs": cs,               # configuration space
                            "deterministic": "true",
                            "output_dir": f"./{save_workload}_smac_output",  
                            "save_model": "true",
                            "local_results_path": f"./models/{save_workload}"
                            })
        
        smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),tae_runner=get_neg_result, runhistory=runhistory)
        incumbent = smac.optimize()  
        print('finish')
        print(type(incumbent))
        print(incumbent)
        # print(get_neg_result(incumbent))
        runhistory = smac.runhistory
        print(runhistory.data)

        def runhistory_to_json(runhistory):
            data_to_save = {}
            for run_key in runhistory.data.keys():
                config_id, instance_id, seed, budget = run_key
                run_value = runhistory.data[run_key]
                data_to_save[str(run_key)] = {
                    "cost": run_value.cost,
                    "time": run_value.time,
                    "status": run_value.status.name,
                    "additional_info": run_value.additional_info
                }
            return json.dumps(data_to_save, indent=4)

        with open(f"smac_his/{save_workload}_{self.warmup}.json", "w") as f:
            f.write(runhistory_to_json(runhistory))

    # def RGPE(self):
    #     matched_workload = self.matched_wl.split('SuperWG/res/gpt_workloads/')[1]
    #     matched_workload = matched_workload.split('.wg')[0]
    #     return 