"""
Core tuner module for PostgreSQL auto-tuning.

This module keeps the legacy `tuner` entry point alive while delegating the
actual SMAC optimization loop to the modern SMACProposalGenerator plugin.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import random

import jsonlines

from knob_config import parse_knob_config
import utils
from Database import Database
from Vectorlib import VectorLibrary
from stress_testing_tool import stress_testing_tool
from safe.subspace_adaptation import Safe
from proposal_generators.smac_generator import SMACProposalGenerator


# Reference TPCH configuration used by the pilot warmup strategy.
tpch_origin = {
    "max_wal_senders": 21,
    "autovacuum_max_workers": 126,
    "max_connections": 860,
    "wal_buffers": 86880,
    "shared_buffers": 1114632,
    "autovacuum_analyze_scale_factor": 78,
    "autovacuum_analyze_threshold": 1202647040,
    "autovacuum_naptime": 101527,
    "autovacuum_vacuum_cost_delay": 45,
    "autovacuum_vacuum_cost_limit": 1114,
    "autovacuum_vacuum_scale_factor": 31,
    "autovacuum_vacuum_threshold": 1280907392,
    "backend_flush_after": 172,
    "bgwriter_delay": 5313,
    "bgwriter_flush_after": 217,
    "bgwriter_lru_maxpages": 47,
    "bgwriter_lru_multiplier": 4,
    "checkpoint_completion_target": 1,
    "checkpoint_flush_after": 44,
    "checkpoint_timeout": 758,
    "commit_delay": 22825,
    "commit_siblings": 130,
    "cursor_tuple_fraction": 1,
    "deadlock_timeout": 885378880,
    "default_statistics_target": 5304,
    "effective_cache_size": 1581112576,
    "effective_io_concurrency": 556,
    "from_collapse_limit": 407846592,
    "geqo_effort": 3,
    "geqo_generations": 1279335040,
    "geqo_pool_size": 838207872,
    "geqo_seed": 0,
    "geqo_threshold": 1336191360,
    "join_collapse_limit": 1755487872,
    "maintenance_work_mem": 1634907776,
    "temp_buffers": 704544576,
    "temp_file_limit": -1,
    "vacuum_cost_delay": 46,
    "vacuum_cost_limit": 5084,
    "vacuum_cost_page_dirty": 6633,
    "vacuum_cost_page_hit": 6940,
    "vacuum_cost_page_miss": 9381,
    "wal_writer_delay": 4773,
    "work_mem": 716290752,
}


def add_noise(knobs_detail, origin_config, ratio):
    """
    Add bounded random noise to a reference configuration.

    This is used by the `pilot` warmup strategy to generate an initial point
    around a known good TPCH configuration.
    """
    new_config = copy.deepcopy(origin_config)

    for knob_name, detail in knobs_detail.items():
        upper_bound = detail["max"]
        lower_bound = detail["min"]

        if upper_bound - lower_bound <= 1:
            continue

        noise_span = int((upper_bound - lower_bound) * ratio * 0.5)
        noise = random.randint(-noise_span, noise_span)

        candidate = origin_config[knob_name] + noise
        if candidate < lower_bound:
            candidate = lower_bound
        elif candidate > upper_bound:
            candidate = upper_bound
        new_config[knob_name] = candidate

    print(new_config)
    return new_config


class tuner:
    """Legacy tuner facade used by the controller layer."""

    def __init__(self, config):
        # Direct mode needs a live database connection; surrogate mode does not.
        if config["benchmark_config"]["tool"] != "surrogate":
            self.database = Database(
                config["database_config"]
            )  # Database now accepts only the DB config mapping.
            print(f"Connected to database: {config['database_config']['database']}")
        else:
            self.database = None
            print("Using surrogate mode without a database connection")

        self.method = config["tuning_config"]["tuning_method"]
        self.warmup = config["tuning_config"]["warmup_method"]
        self.online = config["tuning_config"]["online"]
        self.online_sample = config["tuning_config"]["online_sample"]
        self.offline_sample = config["tuning_config"]["offline_sample"]
        self.finetune_sample = config["tuning_config"]["finetune_sample"]
        self.inner_metric_sample = config["tuning_config"]["inner_metric_sample"]

        self.sampling_number = int(config["tuning_config"]["sample_num"])
        self.iteration = int(config["tuning_config"]["suggest_num"])

        self.knobs_detail = parse_knob_config.get_knobs(
            config["tuning_config"]["knob_config"]
        )
        print(f"Loaded {len(self.knobs_detail)} tunable knobs")

        self.logger = utils.get_logger(config["tuning_config"]["log_path"])
        self.logger.info(
            "Initializing tuner: method=%s, warmup=%s",
            self.method,
            self.warmup,
        )

        self.ssh_host = config["ssh_config"]["host"]
        self.last_point = []

        if self.online == "false":
            self.stt = stress_testing_tool(
                config, self.database, self.logger, self.offline_sample
            )
        else:
            self.stt = stress_testing_tool(
                config, self.database, self.logger, self.finetune_sample
            )

        self.pre_safe = None
        self.post_safe = None

        self.veclib = VectorLibrary(config["database_config"]["database"])
        feature_path = f"SuperWG/feature/{config['database_config']['database']}.json"
        try:
            with open(feature_path, "r", encoding="utf-8") as handle:
                features = json.load(handle)
            self.logger.info("Loaded %s workload feature vectors", len(features))
        except FileNotFoundError:
            self.logger.warning("Feature file not found: %s", feature_path)
            features = {}

        self.wl_id = config["benchmark_config"]["workload_path"]

        if self.warmup == "workload_map" and self.wl_id in features:
            self.feature = features[self.wl_id]
            self.rh_data, self.matched_wl = self.workload_mapper(
                config["database_config"]["database"], k=3
            )
            self.logger.info(
                "Loaded %s warmup history samples from workload_map",
                len(self.rh_data),
            )
        elif self.warmup == "rgpe" and self.wl_id in features:
            self.feature = features[self.wl_id]
            self.rh_data, self.matched_wl = self.workload_mapper(
                config["database_config"]["database"], k=10
            )
            self.logger.info(
                "Loaded %s warmup history samples from RGPE",
                len(self.rh_data),
            )
        else:
            self.rh_data = []
            self.matched_wl = None

        self.init_safe()
        self.logger.info("Tuner initialization completed")

    def workload_mapper(self, database, k):
        matched_wls = self.veclib.find_most_similar(self.feature, k)
        rh_data = []
        best_wl = None
        keys_to_remove = ["tps", "y", "inner_metrics", "workload"]

        for wl in matched_wls:
            if len(rh_data) > 50:
                break
            if wl == self.wl_id:
                continue

            with jsonlines.open(f"offline_sample/offline_sample_{database}.jsonl") as reader:
                for line in reader:
                    if line["workload"] == wl:
                        filtered_config = {
                            key: line[key] for key in line if key not in keys_to_remove
                        }
                        rh_data.append({"config": filtered_config, "tps": line["tps"]})

        for wl in matched_wls:
            if wl != self.wl_id:
                best_wl = wl
                break

        return rh_data, best_wl

    def init_safe(self):
        """
        Initialize safety-related state and evaluate the default configuration.
        """
        if os.path.exists(self.inner_metric_sample):
            with open(self.inner_metric_sample, "r+", encoding="utf-8") as handle:
                handle.truncate(0)
        else:
            open(self.inner_metric_sample, "w", encoding="utf-8").close()

        if os.path.exists(self.offline_sample):
            with open(self.offline_sample, "r+", encoding="utf-8") as handle:
                handle.truncate(0)
        else:
            open(self.offline_sample, "w", encoding="utf-8").close()

        if not os.path.exists(self.offline_sample + ".jsonl"):
            open(self.offline_sample + ".jsonl", "w", encoding="utf-8").close()

        step = []
        lb, ub = [], []
        knob_default = {}

        for knob_name, detail in self.knobs_detail.items():
            if detail["type"] in ["integer", "float"]:
                lb.append(detail["min"])
                ub.append(detail["max"])
            elif detail["type"] == "enum":
                lb.append(0)
                ub.append(len(detail["enum_values"]) - 1)

            knob_default[knob_name] = detail["default"]
            step.append(detail["step"])

        if self.warmup == "ours":
            try:
                with open("model_config.json", "r", encoding="utf-8") as handle:
                    model_config = json.load(handle)
                workload = self.wl_id.split("SuperWG/res/gpt_workloads/")[1]
                knob_default = model_config[workload]
                self.logger.info("Using model-generated parameters as the initial point")
            except (FileNotFoundError, KeyError) as exc:
                self.logger.warning(
                    "Failed to load model-generated parameters: %s; using defaults",
                    exc,
                )
        elif self.warmup == "pilot":
            knob_default = add_noise(self.knobs_detail, tpch_origin, 0.05)
            self.logger.info("Using pilot warmup to build the initial point")

        print("Testing initial parameter configuration...")
        print(knob_default)
        default_performance = self.stt.test_config(knob_default)

        print(f"Initial performance score: {default_performance}")
        self.logger.info(
            "Initial configuration performance: %s",
            default_performance,
        )

        self.pre_safe = Safe(
            default_performance,
            knob_default,
            default_performance,
            lb,
            ub,
            step,
        )

        try:
            with open("safe/predictor.pickle", "rb") as handle:
                self.post_safe = pickle.load(handle)
                self.logger.info("Loaded posterior safety model")
        except FileNotFoundError:
            self.logger.warning("Posterior safety model not found; skipping it")
            self.post_safe = None

        for i in range(4):
            self.logger.debug("Cache warmup pass %s/4", i + 1)
            self.stt.test_config(knob_default)

        self.last_point = list(knob_default.values())

    def tune(self) -> float | None:
        """Dispatch to the configured tuning method."""
        if self.method == "SMAC":
            return self.SMAC()
        return None

    def SMAC(self) -> float:
        """
        Run SMAC through the modern proposal generator plugin.

        This keeps the legacy tuner entry point available while delegating the
        actual optimization loop to the NumPy-2-compatible SMAC 2.x plugin.
        """

        print("Starting SMAC optimization...")
        self.logger.info("===== SMAC optimization start =====")
        self.logger.info(
            "Sampling number: %s, iterations: %s",
            self.sampling_number,
            self.iteration,
        )

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
                self.logger.warning(
                    "SMAC returned no candidate at iteration %s",
                    iteration_index + 1,
                )
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
            self.logger.debug(
                "Completed iteration %s with tps=%.4f",
                iteration_index + 1,
                tps,
            )

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

        self.logger.info("Evaluated %s configurations in total", len(trial_records))
        self.logger.info("Saved optimization result to: %s", result_file)
        self.logger.info("===== SMAC optimization complete =====")
        print("SMAC optimization complete")
        print(f"Saved optimization result to: {result_file}")
        return best_tps
