"""
Parameter schema and safety validation for PostgreSQL tuning configurations.

This module validates:
1. basic payload structure
2. single-parameter normalization/range checks
3. multi-parameter joint constraints that PostgreSQL enforces only at restart
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from postgres_safety_rules import PostgresSafetyRuleEngine


@dataclass
class ValidationIssue:
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


class ParameterValidationResult:
    def __init__(
        self,
        valid: bool,
        normalized_config: Dict[str, Any],
        issues: List[ValidationIssue],
        derived: Optional[Dict[str, Any]] = None,
    ):
        self.valid = valid
        self.normalized_config = normalized_config
        self.issues = issues
        self.derived = derived or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "normalized_config": self.normalized_config,
            "issues": [issue.to_dict() for issue in self.issues],
            "derived": self.derived,
        }


class ParameterConstraintValidator:
    """
    Central validation layer for generator output and batch parameter execution.
    """

    def __init__(self, database, logger=None):
        self.database = database
        self.logger = logger
        self.rule_engine = PostgresSafetyRuleEngine()

    def validate_payload(self, payload: Any) -> ParameterValidationResult:
        if not isinstance(payload, dict):
            return ParameterValidationResult(
                valid=False,
                normalized_config={},
                issues=[
                    ValidationIssue(
                        severity="error",
                        rule="payload.type",
                        message="Parameter payload must be a JSON object / Python dict.",
                    )
                ],
            )

        normalized: Dict[str, Any] = {}
        issues: List[ValidationIssue] = []

        for param_name, raw_value in payload.items():
            if not isinstance(param_name, str):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        rule="payload.key_type",
                        message="Parameter names must be strings.",
                    )
                )
                continue

            metadata = self.database.get_parameter_info(param_name)
            if metadata is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        rule="parameter.unknown",
                        message=f"Unknown PostgreSQL parameter: {param_name}",
                        parameter=param_name,
                    )
                )
                continue

            try:
                normalized_value, was_clamped = self.database.normalize_parameter_value(
                    metadata,
                    raw_value,
                )
                normalized[param_name] = normalized_value
                if was_clamped:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            rule="parameter.clamped",
                            message=(
                                f"{param_name} exceeded pg_settings bounds and was normalized "
                                f"to {normalized_value}"
                            ),
                            parameter=param_name,
                        )
                    )
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        rule="parameter.normalize_failed",
                        message=f"Failed to normalize {param_name}: {exc}",
                        parameter=param_name,
                    )
                )

        safety_report = self.rule_engine.validate(self.database, normalized)
        issues.extend(
            ValidationIssue(
                severity=issue["severity"],
                rule=issue["rule"],
                message=issue["message"],
                parameter=issue.get("parameter"),
            )
            for issue in safety_report["issues"]
        )

        has_error = any(issue.severity == "error" for issue in issues)
        return ParameterValidationResult(
            valid=not has_error,
            normalized_config=normalized,
            issues=issues,
            derived={
                "effective_settings": safety_report.get("effective_settings", {}),
                "safety_report": safety_report,
            },
        )

    def validate_json_text(self, payload_text: str) -> ParameterValidationResult:
        try:
            payload = json.loads(payload_text)
        except Exception as exc:
            return ParameterValidationResult(
                valid=False,
                normalized_config={},
                issues=[
                    ValidationIssue(
                        severity="error",
                        rule="payload.json_decode",
                        message=f"Invalid JSON parameter payload: {exc}",
                    )
                ],
            )
        return self.validate_payload(payload)
