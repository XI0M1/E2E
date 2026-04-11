"""
Reusable PostgreSQL safety rule library.

The goal of this module is to validate parameter configurations before they are
applied to a live PostgreSQL instance. Rules are intentionally modular so new
startup, dependency, memory, or workload-sensitive constraints can be added
without changing the parameter execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


POSTMASTER_PROCESS_LIMIT = 262142


@dataclass
class SafetyRuleIssue:
    severity: str
    rule: str
    message: str
    parameter: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
            "parameter": self.parameter,
        }


class SafetyRuleContext:
    """Runtime context shared by all safety rules."""

    def __init__(self, database, overrides: Dict[str, Any], workload_metadata: Optional[Dict[str, Any]] = None):
        self.database = database
        self.overrides = overrides
        self.workload_metadata = workload_metadata or {}
        self._settings_cache: Dict[str, Any] = {}

    def get_effective_setting(self, param_name: str) -> Optional[Any]:
        if param_name in self._settings_cache:
            return self._settings_cache[param_name]

        metadata = self.database.get_parameter_info(param_name)
        if metadata is None:
            return None

        raw_value = self.overrides.get(param_name, metadata["setting"])
        normalized_value, _ = self.database.normalize_parameter_value(metadata, raw_value)
        self._settings_cache[param_name] = normalized_value
        return normalized_value

    def get_effective_settings(self, param_names: List[str]) -> Dict[str, Any]:
        return {param_name: self.get_effective_setting(param_name) for param_name in param_names}


class PostgresSafetyRule:
    rule_name = "base"

    def evaluate(self, context: SafetyRuleContext) -> List[SafetyRuleIssue]:
        raise NotImplementedError


class PostmasterProcessLimitRule(PostgresSafetyRule):
    rule_name = "joint.postmaster_process_limit"

    def __init__(self, process_limit: int = POSTMASTER_PROCESS_LIMIT):
        self.process_limit = process_limit

    def evaluate(self, context: SafetyRuleContext) -> List[SafetyRuleIssue]:
        effective = context.get_effective_settings(
            [
                "max_connections",
                "autovacuum_worker_slots",
                "max_worker_processes",
                "max_wal_senders",
            ]
        )
        if any(value is None for value in effective.values()):
            return []

        max_connections = int(effective["max_connections"] or 0)
        autovacuum_worker_slots = int(effective["autovacuum_worker_slots"] or 0)
        max_worker_processes = int(effective["max_worker_processes"] or 0)
        max_wal_senders = int(effective["max_wal_senders"] or 0)
        total = (
            max_connections
            + autovacuum_worker_slots
            + max_worker_processes
            + max_wal_senders
        )

        if total >= self.process_limit:
            return [
                SafetyRuleIssue(
                    severity="error",
                    rule=self.rule_name,
                    message=(
                        "Restart-required process parameters exceed PostgreSQL startup limit: "
                        f"max_connections({max_connections}) + "
                        f"autovacuum_worker_slots({autovacuum_worker_slots}) + "
                        f"max_worker_processes({max_worker_processes}) + "
                        f"max_wal_senders({max_wal_senders}) = {total}, "
                        f"must be less than {self.process_limit}."
                    ),
                )
            ]
        return []


class WorkerProcessConsistencyRule(PostgresSafetyRule):
    rule_name = "dependency.worker_process_capacity"

    def evaluate(self, context: SafetyRuleContext) -> List[SafetyRuleIssue]:
        max_worker_processes = context.get_effective_setting("max_worker_processes")
        autovacuum_max_workers = context.get_effective_setting("autovacuum_max_workers")
        if max_worker_processes is None or autovacuum_max_workers is None:
            return []

        max_worker_processes = int(max_worker_processes)
        autovacuum_max_workers = int(autovacuum_max_workers)
        if autovacuum_max_workers > max_worker_processes:
            return [
                SafetyRuleIssue(
                    severity="warning",
                    rule=self.rule_name,
                    message=(
                        f"autovacuum_max_workers({autovacuum_max_workers}) exceeds "
                        f"max_worker_processes({max_worker_processes}); this may reduce worker availability."
                    ),
                )
            ]
        return []


class PostgresSafetyRuleEngine:
    """Extensible rule engine for PostgreSQL parameter safety checks."""

    def __init__(self, rules: Optional[List[PostgresSafetyRule]] = None):
        self.rules = rules or self.default_rules()

    @classmethod
    def default_rules(cls) -> List[PostgresSafetyRule]:
        return [
            PostmasterProcessLimitRule(),
            WorkerProcessConsistencyRule(),
        ]

    def validate(
        self,
        database,
        overrides: Dict[str, Any],
        workload_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = SafetyRuleContext(database, overrides, workload_metadata=workload_metadata)
        issues: List[SafetyRuleIssue] = []
        for rule in self.rules:
            issues.extend(rule.evaluate(context))

        valid = not any(issue.severity == "error" for issue in issues)
        return {
            "valid": valid,
            "issues": [issue.to_dict() for issue in issues],
            "effective_settings": context._settings_cache.copy(),
            "rule_count": len(self.rules),
        }

