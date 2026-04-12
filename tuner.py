"""
璋冧紭鍣ㄦā鍧楋紙Tuner Module锛?
鍔熻兘锛氭牳蹇冪殑鏁版嵁搴撳弬鏁颁紭鍖栫畻娉曞疄鐜?

鏍稿績绠楁硶锛歋MAC (Sequential Model-Based Algorithm Configuration)
- 浣跨敤璐濆彾鏂紭鍖栨潵鎼滅储鍙傛暟閰嶇疆绌洪棿
- 閫氳繃瀛︿範浠ｇ悊妯″瀷(闅忔満妫灄)閫愭鏀硅繘閰嶇疆
- 鍦ㄦ湁闄愮殑璇勪及娆℃暟鍐呮壘鍒版帴杩戞渶浼樼殑鍙傛暟

鍏抽敭鐗规€э細
1. 宸ヤ綔璐熻浇鏄犲皠锛氭壘鍒扮浉浼煎伐浣滆礋杞戒互閲嶇敤鍘嗗彶鏁版嵁
2. 棰勭儹绛栫暐锛氭敮鎸佸绉嶅垵濮嬪寲鏂瑰紡锛坥urs, pilot, workload_map绛夛級
3. 瀹夊叏绾︽潫锛氱‘淇濈敓鎴愮殑閰嶇疆鍦ㄦ湁鏁堣寖鍥村唴
4. 瀹炴椂閲囨牱锛氳竟浼樺寲杈逛繚瀛樻暟鎹敤浜庝唬鐞嗘ā鍨嬭缁?
"""

import os
import csv
import pickle
import sys
import json
import copy

# 鏈湴妯″潡瀵煎叆
from knob_config import parse_knob_config
import utils
import numpy as np
import pandas as pd
import jsonlines
import random

# 鏁版嵁搴撳拰宸ュ叿妯″潡
from Database import Database
from Vectorlib import VectorLibrary
from stress_testing_tool import stress_testing_tool
from safe.subspace_adaptation import Safe
from proposal_generators.smac_generator import SMACProposalGenerator

# SMAC浼樺寲妗嗘灦鐩稿叧瀵煎叆
from poap.controller import BasicWorkerThread, ThreadController
from pySOT.experimental_design import LatinHypercube
from pySOT import strategy, surrogate


# ==================== 甯告暟瀹氫箟 ====================

# TPC-H宸ヤ綔璐熻浇鐨勯粯璁ゅ弬鏁伴厤缃?
# 杩欐槸涓€涓弬鑰冮厤缃紝鐢ㄤ簬pilot棰勭儹鏂规硶锛氬熀浜庤繖涓厤缃坊鍔犲櫔澹扮敓鎴愬垵濮嬬偣
tpch_origin = {"max_wal_senders": 21, "autovacuum_max_workers": 126, "max_connections": 860, "wal_buffers": 86880, "shared_buffers": 1114632, "autovacuum_analyze_scale_factor": 78, "autovacuum_analyze_threshold": 1202647040, "autovacuum_naptime": 101527, "autovacuum_vacuum_cost_delay": 45, "autovacuum_vacuum_cost_limit": 1114, "autovacuum_vacuum_scale_factor": 31, "autovacuum_vacuum_threshold": 1280907392, "backend_flush_after": 172, "bgwriter_delay": 5313, "bgwriter_flush_after": 217, "bgwriter_lru_maxpages": 47, "bgwriter_lru_multiplier": 4, "checkpoint_completion_target": 1, "checkpoint_flush_after": 44, "checkpoint_timeout": 758, "commit_delay": 22825, "commit_siblings": 130, "cursor_tuple_fraction": 1, "deadlock_timeout": 885378880, "default_statistics_target": 5304, "effective_cache_size": 1581112576, "effective_io_concurrency": 556, "from_collapse_limit": 407846592, "geqo_effort": 3, "geqo_generations": 1279335040, "geqo_pool_size": 838207872, "geqo_seed": 0, "geqo_threshold": 1336191360, "join_collapse_limit": 1755487872, "maintenance_work_mem": 1634907776, "temp_buffers": 704544576, "temp_file_limit": -1, "vacuum_cost_delay": 46, "vacuum_cost_limit": 5084, "vacuum_cost_page_dirty": 6633, "vacuum_cost_page_hit": 6940, "vacuum_cost_page_miss": 9381, "wal_writer_delay": 4773, "work_mem": 716290752}


def add_noise(knobs_detail, origin_config, range):
    """
    鍚戝弬鏁伴厤缃坊鍔犻殢鏈哄櫔澹?
    
    鐢ㄩ€旓細pilot棰勭儹绛栫暐浣跨敤姝ゅ嚱鏁板熀浜庡凡鐭ョ殑濂介厤缃坊鍔犳壈鍔ㄤ互鐢熸垚澶氭牱鍖栫殑鍒濆鐐?
    
    鍙傛暟锛?
        knobs_detail (dict): 鍙傛暟璇︾粏淇℃伅锛屽寘鍚瘡涓弬鏁扮殑min/max鍊?
        origin_config (dict): 鍘熷鍙傛暟閰嶇疆
        range (float): 鍣０鑼冨洿鐧惧垎姣旓紙0-1锛夛紝渚嬪0.05琛ㄧず卤5%鐩稿浜庡弬鏁拌寖鍥?
    
    杩斿洖锛?
        dict: 娣诲姞闅忔満鍣０鍚庣殑鏂伴厤缃紝鎵€鏈夊€奸兘鍦ㄥ悎娉曡寖鍥村唴
    
    鍘熺悊锛?
        瀵逛簬姣忎釜鍙傛暟锛屽湪鍏惰寖鍥寸殑卤range%鍐呯敓鎴愰殢鏈烘壈鍔紝纭繚鏈€缁堝€间笉瓒呭嚭杈圭晫
        杩欐牱鍙互鍦ㄥ凡鐭ョ殑浼樺娍鐐瑰懆鍥磋繘琛屽眬閮ㄦ帰绱紝鍔犲揩鏀舵暃
    """
    new_config = copy.deepcopy(origin_config)
    
    for knob in knobs_detail:
        detail = knobs_detail[knob]
        rb = detail['max']  # 鍙傛暟涓婄晫
        lb = detail['min']  # 鍙傛暟涓嬬晫
        
        # 鍙傛暟鑼冨洿澶皬鍒欒烦杩囧櫔澹版坊鍔?
        if rb - lb <= 1:
            continue
        
        # 璁＄畻鍣０骞呭害锛氳寖鍥寸殑卤range%
        length = int((rb - lb) * range * 0.5)
        noise = random.randint(-length, length)
        
        # 搴旂敤鍣０骞朵弗鏍奸檺鍒跺湪鍚堟硶鑼冨洿鍐?
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
    鏁版嵁搴撳弬鏁拌嚜鍔ㄨ皟浼樺櫒鏍稿績绫?
    
    ==== 鏍稿績鑱岃矗 ====
    1. 鍒濆鍖栬皟浼樼幆澧冿紙鏁版嵁搴撹繛鎺ャ€佸弬鏁扮┖闂淬€佹棩蹇楋級
    2. 绠＄悊鍙傛暟绌洪棿鍜岀害鏉?
    3. 杩愯鍙傛暟浼樺寲绠楁硶锛圫MAC锛?
    4. 璁板綍閲囨牱鏁版嵁鍜屽巻鍙茬粨鏋?
    5. 搴旂敤瀹夊叏绾︽潫妫€鏌?
    
    ==== 浼樺寲娴佺▼ ====
    鍒濆鍖?鈫?鍙傛暟绌洪棿瀹氫箟 鈫?榛樿閰嶇疆娴嬭瘯 鈫?宸ヤ綔璐熻浇鏄犲皠 鈫?
    SMAC杩唬浼樺寲 鈫?鏈€浼橀厤缃繚瀛?
    
    ==== 灞炴€ц鏄?====
    database: PostgreSQL鏁版嵁搴撹繛鎺ュ璞★紙鑻ool='surrogate'鍒欎负None锛?
    method: 璋冧紭鏂规硶鍚嶇О锛堝'SMAC'锛?
    warmup: 棰勭儹绛栫暐锛堝喅瀹氬垵濮嬬偣閫夋嫨锛?
        - 'ours': 浣跨敤妯″瀷棰勭敓鎴愮殑鍙傛暟
        - 'pilot': 鍩轰簬宸茬煡浼樼閰嶇疆鐨勬壈鍔?
        - 'workload_map': 浣跨敤3涓渶鐩镐技宸ヤ綔璐熻浇鐨勫巻鍙叉暟鎹?
        - 'rgpe': 浣跨敤10涓渶鐩镐技宸ヤ綔璐熻浇鐨勫巻鍙叉暟鎹?
    knobs_detail: 鎵€鏈夊弬鏁扮殑瀹氫箟锛堝寘鍚玹ype, min, max, default绛夛級
    pre_safe: 鍓嶉獙瀹夊叏妯″瀷锛堢敤浜庣害鏉熸悳绱㈢┖闂达級
    post_safe: 鍚庨獙瀹夊叏妯″瀷锛堢敤浜庨獙璇佺敓鎴愮殑閰嶇疆锛?
    veclib: 宸ヤ綔璐熻浇鍚戦噺搴?
    stt: 鍘嬪姏娴嬭瘯宸ュ叿瀹炰緥
    """
    
    def __init__(self, config):
        """
        鍒濆鍖栬皟浼樺櫒
        
        鍒濆鍖栨祦绋嬶細
        1. 寤虹珛鏁版嵁搴撹繛鎺ワ紙闈炰唬鐞嗘ā寮忥級
        2. 鍔犺浇閰嶇疆鍙傛暟鍜屽弬鏁板畾涔?
        3. 鍒濆鍖栧帇鍔涙祴璇曞伐鍏?
        4. 鍔犺浇宸ヤ綔璐熻浇鐗瑰緛鍚戦噺
        5. 鏍规嵁棰勭儹绛栫暐鍑嗗鍒濆鏁版嵁
        6. 鍒濆鍖栧畨鍏ㄧ害鏉熸鏋?
        
        鍙傛暟锛?
            config (dict): 鍏ㄥ眬閰嶇疆瀛楀吀锛屽寘鍚互涓嬪叧閿畇ections:
                benchmark_config: 
                    - tool: 'direct'(鏁版嵁搴? 鎴?'surrogate'(浠ｇ悊妯″瀷)
                    - workload_path: 宸ヤ綔璐熻浇鏂囦欢璺緞
                tuning_config:
                    - tuning_method: 'SMAC'
                    - warmup_method: 棰勭儹绛栫暐
                    - sample_num: 鍒濆閲囨牱鏁伴噺
                    - suggest_num: 浼樺寲杩唬鏁?
                    - knob_config: 鍙傛暟瀹氫箟鏂囦欢璺緞
                    - log_path: 鏃ュ織鏂囦欢璺緞
                ssh_config, database_config: 杩炴帴淇℃伅
        
        寮傚父锛?
            FileNotFoundError: 鍙傛暟瀹氫箟鏂囦欢鎴栫壒寰佹枃浠朵笉瀛樺湪
            psycopg2.Error: 鏁版嵁搴撹繛鎺ュけ璐?
        """
        
        # ==================== 绗竴姝ワ細鍒濆鍖栨暟鎹簱 ====================
        # 鍐冲畾鏄惁闇€瑕佺湡瀹炴暟鎹簱杩炴帴
        # 浣跨敤浠ｇ悊妯″瀷鏃讹紙tool='surrogate'锛変笉闇€瑕佺湡瀹濪B锛屼粎杩涜鎺ㄧ悊
        if config['benchmark_config']['tool'] != 'surrogate':
            # 鐩存帴鏂规硶锛氶渶瑕佽繛鎺ョ湡瀹炴暟鎹簱
            self.database = Database(config, config['tuning_config']['knob_config'])
            print(f"宸茶繛鎺ュ埌鏁版嵁搴? {config['database_config']['database']}")
        else: 
            # 浠ｇ悊妯″瀷鏂规硶锛氫笉闇€瑕佹暟鎹簱杩炴帴
            self.database = None
            print("使用代理模型，无需数据库连接")
        
        # ==================== 绗簩姝ワ細鍔犺浇閰嶇疆鍙傛暟 ====================
        # 璋冧紭閰嶇疆鍙傛暟
        self.method = config['tuning_config']['tuning_method']      # 閫氬父涓?SMAC'
        self.warmup = config['tuning_config']['warmup_method']      # 棰勭儹绛栫暐
        self.online = config['tuning_config']['online']             # 'true'鎴?false'
        self.online_sample = config['tuning_config']['online_sample']
        self.offline_sample = config['tuning_config']['offline_sample']
        self.finetune_sample = config['tuning_config']['finetune_sample']
        self.inner_metric_sample = config['tuning_config']['inner_metric_sample']
        
        # 浼樺寲鍙傛暟
        self.sampling_number = int(config['tuning_config']['sample_num'])  # 鍒濆閲囨牱鏁?
        self.iteration = int(config['tuning_config']['suggest_num'])       # SMAC杩唬鏁?
        
        # ==================== 绗笁姝ワ細鍔犺浇鍙傛暟瀹氫箟 ====================
        # 浠巏nob_config.json鍔犺浇鎵€鏈夊彲浼樺寲鍙傛暟鐨勫畾涔?
        # 鍖呭惈姣忎釜鍙傛暟鐨則ype(integer/float/enum), min, max, default, step绛?
        self.knobs_detail = parse_knob_config.get_knobs(
            config['tuning_config']['knob_config']
        )
        print(f"已加载 {len(self.knobs_detail)} 个可优化参数")
        
        # ==================== 绗洓姝ワ細鍒濆鍖栨棩蹇楀拰杩炴帴淇℃伅 ====================
        # 鍒涘缓鏃ュ織鏂囦欢鐢ㄤ簬璁板綍浼樺寲杩囩▼
        self.logger = utils.get_logger(config['tuning_config']['log_path'])
        self.logger.info(f"寮€濮嬭皟浼? 鏂规硶={self.method}, 棰勭儹={self.warmup}")
        
        # SSH杩炴帴淇℃伅锛堢敤浜庤繙绋嬫墽琛屽懡浠わ級
        self.ssh_host = config['ssh_config']['host']
        self.last_point = []  # 涓婁竴涓瘎浼扮殑閰嶇疆
        
        # ==================== 绗簲姝ワ細鍒濆鍖栧帇鍔涙祴璇曞伐鍏?====================
        # 鍘嬪姏娴嬭瘯宸ュ叿璐熻矗鎵ц宸ヤ綔璐熻浇骞舵敹闆嗘€ц兘鎸囨爣
        if self.online == 'false':
            # 绂荤嚎閲囨牱妯″紡锛氫繚瀛樺埌offline_sample_*.jsonl
            self.stt = stress_testing_tool(
                config, self.database, self.logger, self.offline_sample
            )
        else:
            # 鍦ㄧ嚎锛堝井璋冿級妯″紡锛氫繚瀛樺埌finetune_sample
            self.stt = stress_testing_tool(
                config, self.database, self.logger, self.finetune_sample
            )

        # ==================== 绗叚姝ワ細鍒濆鍖栧畨鍏ㄧ害鏉熺浉鍏?====================
        # 瀹夊叏绾︽潫鐢ㄤ簬纭繚鐢熸垚鐨勫弬鏁伴厤缃湪鍚堟硶鍜屽畨鍏ㄧ殑鑼冨洿鍐?
        self.pre_safe = None    # 鍓嶉獙瀹夊叏妯″瀷锛氱害鏉熸悳绱㈢┖闂?
        self.post_safe = None   # 鍚庨獙瀹夊叏妯″瀷锛氶獙璇佺敓鎴愰厤缃?
        
        # ==================== 绗竷姝ワ細鍔犺浇宸ヤ綔璐熻浇鍚戦噺搴?====================
        # 鐢ㄤ簬宸ヤ綔璐熻浇鐩镐技鎬ц绠楀拰宸ヤ綔璐熻浇鏄犲皠
        self.veclib = VectorLibrary(config['database_config']['database'])
        
        # 灏濊瘯鍔犺浇宸ヤ綔璐熻浇鐗瑰緛鍚戦噺锛堢敤浜庡伐浣滆礋杞芥槧灏勶級
        feature_path = f"SuperWG/feature/{config['database_config']['database']}.json"
        try:
            with open(feature_path, 'r') as f:
                features = json.load(f)
            self.logger.info("已加载 %s 个工作负载的特征向量", len(features))
        except FileNotFoundError:
            self.logger.warning("特征文件不存在: %s", feature_path)
            features = {}
        
        # 褰撳墠宸ヤ綔璐熻浇ID
        self.wl_id = config['benchmark_config']['workload_path']
        
        # ==================== 绗叓姝ワ細宸ヤ綔璐熻浇鏄犲皠 ====================
        # 鏍规嵁棰勭儹绛栫暐閫夋嫨鏄惁浣跨敤鐩镐技宸ヤ綔璐熻浇鐨勫巻鍙叉暟鎹?
        # 杩欏彲浠ュ姞閫熷綋鍓嶅伐浣滆礋杞界殑浼樺寲
        
        if self.warmup == 'workload_map' and self.wl_id in features:
            # workload_map: 浣跨敤3涓渶鐩镐技鐨勫伐浣滆礋杞?
            self.feature = features[self.wl_id]
            self.rh_data, self.matched_wl = self.workload_mapper(
                config['database_config']['database'], k=3
            )
            self.logger.info(f"鎵惧埌{len(self.rh_data)}鏉＄浉浼煎伐浣滆礋杞界殑鍘嗗彶鏁版嵁")
            
        elif self.warmup == 'rgpe' and self.wl_id in features:
            # rgpe: 浣跨敤10涓渶鐩镐技鐨勫伐浣滆礋杞斤紙鏇村鏍峰寲锛?
            self.feature = features[self.wl_id]
            self.rh_data, self.matched_wl = self.workload_mapper(
                config['database_config']['database'], k=10
            )
            self.logger.info(f"鎵惧埌{len(self.rh_data)}鏉＄浉浼煎伐浣滆礋杞界殑鍘嗗彶鏁版嵁(RGPE)")

        # ==================== 绗節姝ワ細鍒濆鍖栧畨鍏ㄦā鍨?====================
        # 璇勪及榛樿鍙傛暟骞跺垵濮嬪寲瀹夊叏绾︽潫妗嗘灦
        self.init_safe()
        self.logger.info("璋冧紭鍣ㄥ垵濮嬪寲瀹屾垚")

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
        鍒濆鍖栧畨鍏ㄧ害鏉熸鏋?
        
        ==== 鏍稿績鍔熻兘 ====
        1. 娓呯悊鏃х殑閲囨牱鏁版嵁鏂囦欢
        2. 鏀堕泦鍙傛暟鐨勬悳绱㈢┖闂磋寖鍥?
        3. 閫夋嫨鍜屾祴璇曢粯璁ら厤缃?
        4. 鍒濆鍖栧墠楠屽拰鍚庨獙瀹夊叏妯″瀷
        5. 杩涜缂撳瓨棰勭儹
        
        ==== 宸ヤ綔娴佺▼ ====
        娓呯悊鏁版嵁 鈫?鏀堕泦杈圭晫 鈫?閫夋嫨鍒濆鐐?鈫?娴嬭瘯鍒濆鐐?鈫?寤虹珛瀹夊叏妗嗘灦 鈫?缂撳瓨棰勭儹
        
        ==== 棰勭儹绛栫暐璇存槑 ====
        - 'ours': 浣跨敤鐢熸垚寮忔ā鍨嬮鐢熸垚鐨勫弬鏁帮紙鏉ヨ嚜model_config.json锛?
        - 'pilot': 鍩轰簬tpch_origin閰嶇疆娣诲姞卤5%鐨勯殢鏈烘壈鍔?
        - 鍏朵粬: 浣跨敤鍙傛暟鐨勯粯璁ゅ€?
        
        ==== 杈撳嚭 ====
        鍒濆鍖栦袱涓畨鍏ㄦā鍨嬶細
        - pre_safe: 鍦ㄤ紭鍖栧墠绾︽潫鎼滅储绌洪棿
        - post_safe: 鍦ㄥ弬鏁扮敓鎴愬悗杩涜楠岃瘉
        """
        # ==================== 绗竴姝ワ細娓呯悊鏃ф暟鎹?====================
        # 鍒犻櫎涓婁竴娆¤皟浼樼殑涓存椂鏂囦欢锛岀‘淇濇瘡娆¤皟浼樹粠骞插噣鐘舵€佸紑濮?
        
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

        # ==================== 绗簩姝ワ細鏀堕泦鍙傛暟绌洪棿淇℃伅 ====================
        # 浠庡弬鏁板畾涔変腑鎻愬彇姣忎釜鍙傛暟鐨勪笂鐣屻€佷笅鐣屽拰姝ラ暱
        # 杩欎簺淇℃伅鐢ㄤ簬SMAC浼樺寲鍣ㄥ畾涔夋悳绱㈢┖闂?
        
        step = []  # 鍙傛暟姝ラ暱鍒楄〃
        lb, ub = [], []  # 涓嬬晫銆佷笂鐣屽垪琛?
        knob_default = {}  # 榛樿鍙傛暟閰嶇疆

        for index, knob in enumerate(self.knobs_detail):
            detail = self.knobs_detail[knob]
            
            if detail['type'] in ['integer', 'float']:
                # 鏁板€煎弬鏁帮細鐩存帴浣跨敤min/max
                lb.append(detail['min'])
                ub.append(detail['max'])
            elif detail['type'] == 'enum':
                # 鏋氫妇鍙傛暟锛氬皢鍏惰浆鐮佷负0-N鐨勬暣鏁拌寖鍥?
                lb.append(0)
                ub.append(len(detail['enum_values']) - 1)
            
            # 璁板綍榛樿鍊煎拰姝ラ暱
            knob_default[knob] = detail['default']
            step.append(detail['step'])

        # ==================== 绗笁姝ワ細閫夋嫨榛樿閰嶇疆鎴栧垵濮嬬偣 ====================
        # 鏍规嵁棰勭儹绛栫暐閫夋嫨涓嶅悓鐨勫垵濮嬮厤缃?
        
        if self.warmup == 'ours':
            # 'ours'绛栫暐锛氫娇鐢ㄦā鍨嬮鐢熸垚鐨勫弬鏁?
            # 杩欏亣璁惧凡鏈塵odel_config.json鏂囦欢鍖呭惈姣忎釜宸ヤ綔璐熻浇鐨勬帹鑽愬弬鏁?
            try:
                model_config = json.load(open('model_config.json'))
                workload = self.wl_id.split('SuperWG/res/gpt_workloads/')[1]
                knob_default = model_config[workload]
                self.logger.info("使用模型预生成的参数作为初始点")
            except (FileNotFoundError, KeyError) as e:
                self.logger.warning("无法加载模型参数: %s，使用默认值", e)
                
        elif self.warmup == 'pilot':
            # 'pilot'绛栫暐锛氬熀浜庡凡鐭ヤ紭绉€閰嶇疆鐨勯殢鏈烘壈鍔?
            # 鍦╰pch_origin鍛ㄥ洿娣诲姞卤5%鐨勫櫔澹扮敓鎴愬垵濮嬬偣
            origin_config = tpch_origin
            knob_default = add_noise(self.knobs_detail, origin_config, 0.05)
            self.logger.info("使用 pilot 策略生成初始点")

        # ==================== 绗洓姝ワ細娴嬭瘯鍒濆閰嶇疆 ====================
        # 璇勪及閫夊畾鐨勫垵濮嬮厤缃互纭畾鍩虹嚎鎬ц兘
        
        print('娴嬭瘯鍒濆鍙傛暟閰嶇疆鎬ц兘...')
        print(knob_default)
        
        default_performance = self.stt.test_config(knob_default)
        
        print(f'鍒濆鎬ц兘璇勫垎: {default_performance}')
        self.logger.info(f"鍒濆閰嶇疆鎬ц兘: {default_performance}")

        # ==================== 绗簲姝ワ細鍒濆鍖栧畨鍏ㄧ害鏉熸鏋?====================
        # 鍒涘缓鍓嶉獙瀹夊叏妯″瀷锛岀敤浜庡湪浼樺寲杩囩▼涓害鏉熷弬鏁版悳绱㈢┖闂?
        
        self.pre_safe = Safe(
            default_performance,    # 榛樿閰嶇疆鐨勬€ц兘
            knob_default,          # 榛樿鍙傛暟閰嶇疆
            default_performance,   # 褰撳墠鏈€浣虫€ц兘
            lb,                    # 鍙傛暟涓嬬晫
            ub,                    # 鍙傛暟涓婄晫
            step                   # 鍙傛暟姝ラ暱
        )
        
        # 鍔犺浇鍚庨獙瀹夊叏妯″瀷锛堣嫢瀛樺湪锛?
        # 鍚庨獙瀹夊叏妯″瀷鐢ㄤ簬鍦ㄥ弬鏁扮敓鎴愬悗杩涜楠岃瘉鍜岀害鏉?
        try:
            with open('safe/predictor.pickle', 'rb') as f:
                self.post_safe = pickle.load(f)
                self.logger.info("已加载后验安全模型")
        except FileNotFoundError:
            self.logger.warning("后验安全模型不存在，将跳过后验检查")
            self.post_safe = None

        # ==================== 绗叚姝ワ細缂撳瓨棰勭儹 ====================
        # 杩愯4娆″垵濮嬮厤缃互鍏呭垎棰勭儹鏁版嵁搴撶紦瀛?
        # 纭繚鍚庣画鎬ц兘娴嬭瘯寰楀埌绋冲畾鐨勭粨鏋?
        
        for i in range(4):
            self.logger.debug("缓存预热第 %s/4 次", i + 1)
            self.stt.test_config(knob_default)
        
        # 璁板綍鍒濆鐐圭敤浜庡弬鑰?
        self.last_point = list(knob_default.values())
        
        # 鍙€夛細璁粌鍚庨獙瀹夊叏妯″瀷锛堥渶瑕佸ぇ閲忓巻鍙叉暟鎹級
        # self.post_safe.train(data_path='./')

    def tune(self) -> float | None:
        """
        璋冧紭涓诲叆鍙?
        
        鏍规嵁閰嶇疆鐨勬柟娉曡皟鐢ㄧ浉搴旂殑浼樺寲绠楁硶
        
        鍙傛暟锛?
            鏃?
        
        杩斿洖锛?
            鏃狅紙缁撴灉淇濆瓨鍒版枃浠讹級
        """
        if self.method == 'SMAC':
            return self.SMAC()
        # 鍙互鍦ㄨ繖閲屾坊鍔犲叾浠栦紭鍖栨柟娉?        # elif self.method == 'BOHB':
        #     self.BOHB()

    def SMAC(self) -> float:
        """
        Run SMAC through the modern proposal generator plugin.

        This keeps the legacy tuner entry point available while delegating the
        actual optimization loop to the NumPy-2-compatible SMAC 2.x plugin.
        """

        print("开始 SMAC 优化过程...")
        self.logger.info("===== SMAC 优化开始 =====")
        self.logger.info("样本数: %s, 迭代数: %s", self.sampling_number, self.iteration)

        workload_name = os.path.splitext(os.path.basename(self.wl_id))[0] or "unknown_workload"
        state_dir = os.path.join("smac_state", workload_name)
        os.makedirs(state_dir, exist_ok=True)
        os.makedirs("smac_his", exist_ok=True)

        generator = SMACProposalGenerator(
            knobs_detail=self.knobs_detail,
            output_dir=state_dir,
            runcount_limit=self.iteration,
            seed=42,
            logger=self.logger,
        )

        history = []
        trial_records = []
        best_tps = 0.0
        workload_features = {}

        for iteration_index in range(self.iteration):
            proposals = generator.generate(
                workload_features=workload_features,
                history=history,
                constraints=self.knobs_detail,
                n=1,
            )
            if not proposals:
                self.logger.warning("SMAC 在第 %s 轮没有返回候选配置", iteration_index + 1)
                continue

            config = proposals[0]
            tps = float(self.stt.test_config(config))
            generator.tell(config, tps)
            history.append({"config": config, "tps": tps})
            trial_records.append(
                {
                    "iteration": iteration_index + 1,
                    "config": config,
                    "tps": tps,
                    "cost": -tps,
                }
            )
            best_tps = max(best_tps, tps)
            self.logger.debug("第 %s 轮评估完成，性能=%.4f", iteration_index + 1, tps)

        generator.save_state(generator.state_path)

        result_file = os.path.join("smac_his", f"{workload_name}_{self.warmup}.json")
        with open(result_file, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "workload": workload_name,
                    "warmup_method": self.warmup,
                    "iterations": self.iteration,
                    "best_tps": best_tps,
                    "trials": trial_records,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        self.logger.info("总共评估了 %s 个配置", len(trial_records))
        self.logger.info("优化结果已保存到: %s", result_file)
        self.logger.info("===== SMAC 优化完成 =====")
        print("SMAC 优化完成")
        print(f"优化结果已保存到: {result_file}")
        return best_tps
