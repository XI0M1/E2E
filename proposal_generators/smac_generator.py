"""
Modern SMAC-backed proposal generator implemented with ask-and-tell.

The outer orchestration loop remains responsible for:
- choosing when to ask for a proposal
- evaluating the proposal on the real system
- feeding the result back through `tell()`

This module only manages SMAC state and candidate generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from proposal_generators.base import (
    ConfigDict,
    HistorySamples,
    KnobConstraints,
    ProposalGenerator,
    WorkloadFeatures,
)


class SMACProposalGenerator(ProposalGenerator):
    """SMAC proposal generator that exposes an external ask-tell interface."""

    def __init__(
        self,
        knobs_detail: dict[str, dict[str, Any]],
        output_dir: str = "./smac_state",
        runcount_limit: int = 75,
        seed: int = 42,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.knobs_detail = knobs_detail
        self.output_dir = output_dir
        self.runcount_limit = int(runcount_limit)
        self.seed = int(seed)
        self.state_path = os.path.join(self.output_dir, "smac_state.json")

        self._history_keys: set[str] = set()
        self._pending_trials: dict[str, list[TrialInfo]] = {}
        self._trial_records: list[dict[str, Any]] = []
        self._knobs_hash = self._compute_knobs_hash(self.knobs_detail)

        os.makedirs(self.output_dir, exist_ok=True)
        self.configspace = self._build_configspace(self.knobs_detail)
        self.scenario = self._build_scenario(self.configspace)
        self.smac = self._build_smac()
        self.runhistory = self.smac.runhistory

        if self.load_state(self.state_path):
            self.logger.info("Loaded SMAC state from %s", self.state_path)

    @property
    def name(self) -> str:
        return "smac"

    def generate(
        self,
        workload_features: WorkloadFeatures,
        history: HistorySamples,
        constraints: KnobConstraints,
        n: int = 1,
    ) -> list[ConfigDict]:
        """
        Ask SMAC for up to `n` next candidate configurations.

        Existing completed history is replayed into the modern SMAC facade so
        the surrogate can be reconstructed without letting SMAC own the loop.
        """
        if n <= 0:
            return []

        self._sync_history(history)

        proposals: list[ConfigDict] = []
        for _ in range(n):
            trial_info = self.smac.ask()
            proposal = self._configuration_to_dict(trial_info.config)
            validated = self.validate_proposal(proposal, constraints)
            proposal_key = self._history_key(validated)
            # Keep the original TrialInfo returned by ask() so tell() updates
            # the same pending trial instead of creating a second runhistory row.
            self._pending_trials.setdefault(proposal_key, []).append(trial_info)
            proposals.append(validated)

        self.runhistory = self.smac.runhistory
        self.logger.info("Generated %s proposals using SMAC", len(proposals))
        return proposals

    def tell(self, config: dict[str, Any], tps: float) -> None:
        """
        Feed one evaluated result back into SMAC's RunHistory.

        SMAC minimizes cost, so TPS is converted to negative cost.
        """
        validated_config = self.validate_proposal(config, self.knobs_detail)
        config_key = self._history_key(validated_config)
        if config_key in self._history_keys:
            return

        trial_info = self._pop_pending_trial(config_key)
        if trial_info is None:
            trial_info = self._build_trial_info(validated_config)

        trial_value = TrialValue(cost=-float(tps), time=0.0)
        self.smac.tell(trial_info, trial_value, save=False)
        self.runhistory = self.smac.runhistory

        self._history_keys.add(config_key)
        self._trial_records.append(
            self._serialize_trial_record(
                config=validated_config,
                tps=float(tps),
                trial_info=trial_info,
                trial_value=trial_value,
            )
        )
        self.logger.debug(
            "Recorded SMAC feedback: cost=%s for config=%s",
            trial_value.cost,
            validated_config,
        )

    def save_state(self, path: str) -> None:
        """
        Persist a minimal JSON state for safe resume.

        We intentionally persist completed trial records instead of pickling the
        full SMAC facade. The optimizer, model, and runhistory are reconstructed
        by replaying these trials on load.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "version": 1,
            "seed": self.seed,
            "runcount_limit": self.runcount_limit,
            "knobs_hash": self._knobs_hash,
            "trial_records": self._trial_records,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self.logger.info("Saved SMAC state to %s", path)

    def load_state(self, path: str) -> bool:
        """
        Restore SMAC state by replaying completed trial history.

        Returns False when the state file does not exist or is unreadable.
        A malformed or mismatched state file is ignored with a warning so the
        pipeline can continue from a clean state.
        """
        if not os.path.exists(path):
            return False

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if payload.get("knobs_hash") not in {None, self._knobs_hash}:
                self.logger.warning(
                    "SMAC state at %s does not match current knob space; ignoring it",
                    path,
                )
                return False

            trial_records = payload.get("trial_records", [])
            if not isinstance(trial_records, list):
                self.logger.warning("SMAC state at %s is malformed; ignoring it", path)
                return False

            self._reset_optimizer_state()
            for record in trial_records:
                self._replay_trial_record(record)

            self.runhistory = self.smac.runhistory
            return True
        except Exception as exc:
            self.logger.warning("Failed to load SMAC state from %s: %s", path, exc)
            return False

    def _build_configspace(self, constraints: KnobConstraints) -> ConfigurationSpace:
        configspace = ConfigurationSpace(seed=self.seed)
        hyperparameters: list[Any] = []

        for knob_name, knob_detail in constraints.items():
            knob_type = knob_detail.get("type")
            if knob_type == "integer":
                hyperparameters.append(
                    Integer(
                        knob_name,
                        (int(knob_detail["min"]), int(knob_detail["max"])),
                        default=int(knob_detail["default"]),
                    )
                )
            elif knob_type == "float":
                hyperparameters.append(
                    Float(
                        knob_name,
                        (float(knob_detail["min"]), float(knob_detail["max"])),
                        default=float(knob_detail["default"]),
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported knob type for SMACProposalGenerator: {knob_type}"
                )

        configspace.add(hyperparameters)
        return configspace

    def _build_scenario(self, configspace: ConfigurationSpace) -> Scenario:
        return Scenario(
            configspace=configspace,
            deterministic=True,
            n_trials=self.runcount_limit,
            seed=self.seed,
            output_directory=Path(self.output_dir) / "smac_internal",  # Isolate SMAC-managed files from our JSON state file.
            name="smac_proposal_generator",
        )

    def _build_smac(self) -> HyperparameterOptimizationFacade:
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            self.scenario,
            max_config_calls=1,
        )
        return HyperparameterOptimizationFacade(
            scenario=self.scenario,
            target_function=self._dummy_target_function,
            intensifier=intensifier,
            overwrite=True,
        )

    def _dummy_target_function(self, config: Configuration, seed: int = 0) -> float:
        """Dummy target function required by the modern SMAC facade."""
        _ = (config, seed)
        return 0.0

    def _sync_history(self, history: HistorySamples) -> None:
        for sample in history:
            config = sample.get("config")
            tps = sample.get("tps")
            if not isinstance(config, dict) or tps is None:
                continue
            self.tell(config, float(tps))

    def _pop_pending_trial(self, config_key: str) -> TrialInfo | None:
        queue = self._pending_trials.get(config_key)
        if not queue:
            return None
        trial_info = queue.pop(0)
        if not queue:
            self._pending_trials.pop(config_key, None)
        return trial_info

    def _build_trial_info(self, config: ConfigDict) -> TrialInfo:
        return TrialInfo(
            config=self._dict_to_configuration(config),
            seed=self.seed,
        )

    def _dict_to_configuration(self, config: ConfigDict) -> Configuration:
        return Configuration(self.configspace, values=config)

    def _configuration_to_dict(self, configuration: Any) -> ConfigDict:
        if isinstance(configuration, dict):
            config_dict = dict(configuration)
        elif hasattr(configuration, "items"):
            config_dict = dict(configuration.items())
        else:
            try:
                config_dict = dict(configuration)
            except Exception as exc:  # pragma: no cover - defensive fallback
                raise TypeError(
                    f"Unsupported SMAC configuration object: {type(configuration)!r}"
                ) from exc

        normalized: ConfigDict = {}
        for knob_name, knob_value in config_dict.items():
            knob_detail = self.knobs_detail.get(knob_name, {})
            if knob_detail.get("type") in {"integer", "float"}:
                normalized[knob_name] = self._snap_to_step(float(knob_value), knob_detail)
            else:
                normalized[knob_name] = knob_value
        return normalized

    def _reset_optimizer_state(self) -> None:
        self._history_keys.clear()
        self._pending_trials.clear()
        self._trial_records = []
        self.smac = self._build_smac()
        self.runhistory = self.smac.runhistory

    def _replay_trial_record(self, record: dict[str, Any]) -> None:
        config = record.get("config")
        tps = record.get("tps")
        if not isinstance(config, dict) or tps is None:
            # Skip malformed records so one bad line does not discard all replayable history.
            self.logger.warning(
                "Skipping malformed trial record (index in file unknown): %r",
                record,
            )
            return

        validated_config = self.validate_proposal(config, self.knobs_detail)
        trial_info = TrialInfo(
            config=self._dict_to_configuration(validated_config),
            instance=record.get("instance"),
            seed=record.get("seed", self.seed),
            budget=record.get("budget"),
        )
        trial_value = TrialValue(
            cost=float(record.get("cost", -float(tps))),
            time=float(record.get("time", 0.0)),
        )
        self.smac.tell(trial_info, trial_value, save=False)

        config_key = self._history_key(validated_config)
        self._history_keys.add(config_key)
        self._trial_records.append(
            self._serialize_trial_record(
                config=validated_config,
                tps=float(tps),
                trial_info=trial_info,
                trial_value=trial_value,
            )
        )

    def _serialize_trial_record(
        self,
        config: ConfigDict,
        tps: float,
        trial_info: TrialInfo,
        trial_value: TrialValue,
    ) -> dict[str, Any]:
        return {
            "config": config,
            "tps": float(tps),
            "cost": float(trial_value.cost),
            "time": float(trial_value.time),
            "seed": trial_info.seed,
            "instance": trial_info.instance,
            "budget": trial_info.budget,
        }

    def _compute_knobs_hash(self, knobs_detail: KnobConstraints) -> str:
        canonical = json.dumps(knobs_detail, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _history_key(config: dict[str, Any]) -> str:
        return json.dumps(
            config,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
