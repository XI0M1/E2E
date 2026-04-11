"""
Unified PostgreSQL parameter execution subsystem.

This module isolates parameter application logic from workload execution so the
rest of the tuning stack can depend on a stable interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from parameter_validation import ParameterConstraintValidator


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ParameterExecutionPolicy:
    apply_reload: bool = True
    apply_restart: bool = True
    reload_if_needed: bool = True
    verify: bool = True
    health_check: bool = True
    rollback_on_failure: bool = True
    session_mode: str = "always"

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "ParameterExecutionPolicy":
        section = config or {}
        session_mode = str(section.get("session_mode", "always")).strip().lower()
        if session_mode not in {"always", "auto", "never"}:
            session_mode = "always"

        return cls(
            apply_reload=_parse_bool(section.get("apply_reload"), True),
            apply_restart=_parse_bool(section.get("apply_restart"), True),
            reload_if_needed=_parse_bool(section.get("reload_if_needed"), True),
            verify=_parse_bool(section.get("verify"), True),
            health_check=_parse_bool(section.get("health_check"), True),
            rollback_on_failure=_parse_bool(section.get("rollback_on_failure"), True),
            session_mode=session_mode,
        )


class ParameterExecutionSubsystem:
    """
    Stable entry point for applying PostgreSQL parameter batches.

    The subsystem builds a simple execution plan up front, prepares the session
    if needed, and then delegates the actual SQL/application details to
    `Database.apply_config()`.
    """

    def __init__(
        self,
        database,
        logger: Optional[logging.Logger] = None,
        policy: Optional[ParameterExecutionPolicy] = None,
    ):
        self.database = database
        self.logger = logger or logging.getLogger("ParameterExecutionSubsystem")
        self.policy = policy or ParameterExecutionPolicy()
        self.validator = ParameterConstraintValidator(database, self.logger)

    @classmethod
    def from_config(cls, config: Dict[str, Any], database, logger=None) -> "ParameterExecutionSubsystem":
        policy = ParameterExecutionPolicy.from_config(config.get("parameter_execution"))
        return cls(database=database, logger=logger, policy=policy)

    def inspect_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        plan = {
            "total": len(config_dict),
            "dynamic": 0,
            "reload": 0,
            "restart": 0,
            "unknown": 0,
            "session_sensitive": [],
            "contexts": {},
            "requires_reload": False,
            "requires_restart": False,
        }

        for param_name in config_dict:
            metadata = self.database.get_parameter_info(param_name)
            if metadata is None:
                plan["unknown"] += 1
                continue

            context = metadata["context"]
            plan["contexts"][param_name] = context

            if self.database.requires_fresh_session({param_name: config_dict[param_name]}):
                plan["session_sensitive"].append(param_name)

            if context in self.database.DYNAMIC_CONTEXTS:
                plan["dynamic"] += 1
            elif context in self.database.RELOAD_CONTEXTS:
                plan["reload"] += 1
                plan["requires_reload"] = True
            elif context in self.database.RESTART_CONTEXTS:
                plan["restart"] += 1
                plan["requires_restart"] = True
            else:
                plan["unknown"] += 1

        return plan

    def validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        return self.validator.validate_payload(config_dict).to_dict()

    def validate_json_config(self, payload_text: str) -> Dict[str, Any]:
        return self.validator.validate_json_text(payload_text).to_dict()

    def _resolve_force_new_session(
        self,
        config_dict: Dict[str, Any],
        force_new_session: Optional[bool],
    ) -> bool:
        if force_new_session is not None:
            return force_new_session
        if self.policy.session_mode == "always":
            return True
        if self.policy.session_mode == "never":
            return False
        return self.database.requires_fresh_session(config_dict)

    def apply(
        self,
        config_dict: Dict[str, Any],
        apply_static: Optional[bool] = None,
        restart_if_static: Optional[bool] = None,
        force_new_session: Optional[bool] = None,
    ) -> Dict[str, Any]:
        validation = self.validator.validate_payload(config_dict)
        plan = self.inspect_config(validation.normalized_config)
        effective_apply_static = (
            apply_static
            if apply_static is not None
            else (self.policy.apply_reload or self.policy.apply_restart)
        )
        effective_restart_if_static = (
            restart_if_static
            if restart_if_static is not None
            else self.policy.apply_restart
        )
        effective_force_new_session = self._resolve_force_new_session(
            validation.normalized_config,
            force_new_session=force_new_session,
        )

        if not validation.valid:
            self.logger.error(
                "Parameter batch rejected by validator: %s",
                "; ".join(issue.message for issue in validation.issues if issue.severity == "error"),
            )
            return {
                "dynamic": 0,
                "static": 0,
                "reload": 0,
                "skipped": len(validation.normalized_config),
                "failed": len([issue for issue in validation.issues if issue.severity == "error"]),
                "clamped": len([issue for issue in validation.issues if issue.rule == "parameter.clamped"]),
                "verified": 0,
                "restarted": False,
                "reloaded": False,
                "health_ok": False,
                "health_report": None,
                "rollback": None,
                "requires_manual_recovery": False,
                "details": [
                    (
                        issue.parameter or "config",
                        f"{issue.rule}:{issue.message}",
                    )
                    for issue in validation.issues
                ],
                "execution_plan": plan,
                "policy": {
                    "apply_static": effective_apply_static,
                    "restart_if_static": effective_restart_if_static,
                    "force_new_session": effective_force_new_session,
                    "verify": self.policy.verify,
                    "health_check": self.policy.health_check,
                    "rollback_on_failure": self.policy.rollback_on_failure,
                },
                "validation": validation.to_dict(),
            }

        self.database.prepare_session_for_config(
            validation.normalized_config,
            force_new_session=effective_force_new_session,
        )

        self.logger.info(
            "Parameter batch plan: total=%s dynamic=%s reload=%s restart=%s fresh_session=%s",
            plan["total"],
            plan["dynamic"],
            plan["reload"],
            plan["restart"],
            effective_force_new_session,
        )

        stats = self.database.apply_config(
            validation.normalized_config,
            apply_static=effective_apply_static,
            restart_if_static=effective_restart_if_static,
            reload_if_needed=self.policy.reload_if_needed,
            verify=self.policy.verify,
            health_check=self.policy.health_check,
            rollback_on_failure=self.policy.rollback_on_failure,
        )
        stats["execution_plan"] = plan
        stats["validation"] = validation.to_dict()
        stats["policy"] = {
            "apply_static": effective_apply_static,
            "restart_if_static": effective_restart_if_static,
            "force_new_session": effective_force_new_session,
            "verify": self.policy.verify,
            "health_check": self.policy.health_check,
            "rollback_on_failure": self.policy.rollback_on_failure,
        }
        return stats
