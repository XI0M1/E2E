"""
SMAC-backed proposal generator implemented in ask-tell style.

The outer orchestration loop remains responsible for:
- choosing when to ask for a proposal
- evaluating the proposal on the real system
- feeding the result back through `tell()`

This module only manages SMAC state and candidate generation.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from collections.abc import Iterable
from typing import Any, Dict, List, Set

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario

try:
    from smac.tae.execute_ta_run import StatusType
except Exception:  # pragma: no cover - compatibility fallback for older layouts
    from smac.tae import StatusType  # type: ignore

from proposal_generators.base import ConfigDict, HistorySamples, KnobConstraints, ProposalGenerator, WorkloadFeatures


class SMACProposalGenerator(ProposalGenerator):
    """SMAC proposal generator that exposes an external ask-tell interface."""

    def __init__(
        self,
        knobs_detail: Dict[str, Dict[str, Any]],
        output_dir: str = "./smac_state",
        runcount_limit: int = 75,
        seed: int = 42,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.knobs_detail = knobs_detail
        self.output_dir = output_dir
        self.runcount_limit = runcount_limit
        self.seed = seed
        self.state_path = os.path.join(self.output_dir, "smac_state.pkl")
        self._history_keys: Set[str] = set()

        os.makedirs(self.output_dir, exist_ok=True)
        self.configspace = self._build_configspace(self.knobs_detail)
        self.runhistory = RunHistory()
        self.scenario = self._build_scenario(self.configspace)
        self.smac = self._build_smac(self.scenario, self.runhistory)

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
    ) -> List[ConfigDict]:
        """
        Ask SMAC for up to `n` next candidate configurations.

        Existing history is synced into the shared RunHistory before asking for
        new challengers so the proposal model reflects all completed feedback.
        """
        if n <= 0:
            return []

        self._sync_history(history)
        raw_candidates = self._ask_solver(n)

        proposals: List[ConfigDict] = []
        for candidate in raw_candidates[:n]:
            proposal = self._configuration_to_dict(candidate)
            proposals.append(self.validate_proposal(proposal, constraints))

        self.logger.info("Generated %s proposals using SMAC", len(proposals))
        return proposals

    def tell(self, config: Dict[str, Any], tps: float) -> None:
        """
        Feed one evaluated result back into SMAC's RunHistory.

        SMAC minimizes cost, so TPS is converted to negative cost.
        """
        validated_config = self.validate_proposal(config, self.knobs_detail)
        key = self._history_key(validated_config)
        if key in self._history_keys:
            return

        configuration = self.configspace.get_default_configuration()
        configuration = configuration.copy()
        for knob_name, knob_value in validated_config.items():
            configuration[knob_name] = knob_value

        self.runhistory.add(
            config=configuration,
            cost=-float(tps),
            time=0.0,
            status=StatusType.SUCCESS,
        )
        self._history_keys.add(key)
        self.logger.debug("Recorded SMAC feedback: cost=%s for config=%s", -float(tps), validated_config)

    def save_state(self, path: str) -> None:
        """Persist SMAC RunHistory state to disk for resume."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "runhistory": self.runhistory,
            "history_keys": list(self._history_keys),
            "seed": self.seed,
            "runcount_limit": self.runcount_limit,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        self.logger.info("Saved SMAC state to %s", path)

    def load_state(self, path: str) -> bool:
        """
        Restore SMAC RunHistory state from disk.

        Returns False when the state file does not exist or is unreadable.
        Corrupted state files are ignored with a warning so the pipeline can
        continue from a clean state.
        """
        if not os.path.exists(path):
            return False

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)

            loaded_runhistory = payload.get("runhistory")
            if loaded_runhistory is None:
                return False

            self.runhistory = loaded_runhistory
            self._history_keys = set(payload.get("history_keys", []))
            self.smac = self._build_smac(self.scenario, self.runhistory)
            return True
        except Exception as exc:
            self.logger.warning("Failed to load SMAC state from %s: %s", path, exc)
            return False

    def _build_configspace(self, constraints: KnobConstraints) -> ConfigurationSpace:
        configspace = ConfigurationSpace()
        for knob_name, knob_detail in constraints.items():
            knob_type = knob_detail.get("type")
            if knob_type == "integer":
                minimum = int(knob_detail["min"])
                maximum = int(knob_detail["max"])
                if minimum == maximum:
                    maximum += 1
                hyperparameter = UniformIntegerHyperparameter(
                    knob_name,
                    minimum,
                    maximum,
                    default_value=int(knob_detail["default"]),
                )
            elif knob_type == "float":
                minimum = float(knob_detail["min"])
                maximum = float(knob_detail["max"])
                hyperparameter = UniformFloatHyperparameter(
                    knob_name,
                    minimum,
                    maximum,
                    default_value=float(knob_detail["default"]),
                )
            else:
                raise ValueError(f"Unsupported knob type for SMACProposalGenerator: {knob_type}")

            configspace.add_hyperparameter(hyperparameter)

        return configspace

    def _build_scenario(self, configspace: ConfigurationSpace) -> Scenario:
        return Scenario(
            {
                "run_obj": "quality",
                "runcount-limit": self.runcount_limit,
                "cs": configspace,
                "deterministic": "true",
                "output_dir": self.output_dir,
            }
        )

    def _build_smac(self, scenario: Scenario, runhistory: RunHistory) -> SMAC4HPO:
        return SMAC4HPO(
            scenario=scenario,
            rng=np.random.RandomState(self.seed),
            tae_runner=lambda cfg: 0.0,
            runhistory=runhistory,
        )

    def _sync_history(self, history: HistorySamples) -> None:
        for sample in history:
            config = sample.get("config")
            tps = sample.get("tps")
            if not isinstance(config, dict) or tps is None:
                continue
            self.tell(config, float(tps))

    def _ask_solver(self, n: int) -> List[Any]:
        solver = getattr(self.smac, "solver", None)
        if solver is None:
            return [self.configspace.sample_configuration() for _ in range(n)]

        for attr_name in ("choose_next",):
            chooser = getattr(solver, attr_name, None)
            if chooser is None:
                continue
            raw = chooser()
            candidates = self._normalize_candidates(raw, n)
            if candidates:
                return candidates

        intensifier = getattr(solver, "intensifier", None)
        get_next_challenger = getattr(intensifier, "get_next_challenger", None)
        if callable(get_next_challenger):
            raw = get_next_challenger()
            candidates = self._normalize_candidates(raw, n)
            if candidates:
                return candidates

        return [self.configspace.sample_configuration() for _ in range(n)]

    def _normalize_candidates(self, raw: Any, n: int) -> List[Any]:
        if raw is None:
            return []

        if hasattr(raw, "challenger"):
            return [raw.challenger]

        if hasattr(raw, "get_dictionary"):
            return [raw]

        if isinstance(raw, Iterable) and not isinstance(raw, (dict, str, bytes)):
            candidates: List[Any] = []
            for item in raw:
                if hasattr(item, "challenger"):
                    candidates.append(item.challenger)
                else:
                    candidates.append(item)
                if len(candidates) >= n:
                    break
            return candidates

        return [raw]

    def _configuration_to_dict(self, configuration: Any) -> ConfigDict:
        if hasattr(configuration, "get_dictionary"):
            config_dict = dict(configuration.get_dictionary())
        elif isinstance(configuration, dict):
            config_dict = dict(configuration)
        else:
            try:
                config_dict = dict(configuration)
            except Exception:
                raise TypeError(f"Unsupported SMAC configuration object: {type(configuration)!r}") from None

        normalized: ConfigDict = {}
        for knob_name, knob_value in config_dict.items():
            knob_detail = self.knobs_detail.get(knob_name, {})
            if knob_detail.get("type") in {"integer", "float"}:
                normalized[knob_name] = self._snap_to_step(float(knob_value), knob_detail)
            else:
                normalized[knob_name] = knob_value
        return normalized

    @staticmethod
    def _history_key(config: Dict[str, Any]) -> str:
        return json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
