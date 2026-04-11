"""
Database connection and PostgreSQL parameter management utilities.

This module is intentionally conservative because it sits on the critical path
of automated tuning. The implementation separates read queries from command
execution, distinguishes dynamic vs reloadable vs restart-required parameters,
and provides verification helpers for post-restart smoke tests.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import psycopg2


class Database:
    """PostgreSQL connection wrapper used by the tuning system."""

    DYNAMIC_CONTEXTS = {"backend", "user", "superuser"}
    RELOAD_CONTEXTS = {"sighup"}
    RESTART_CONTEXTS = {"postmaster"}
    SESSION_INIT_PARAMS = {"temp_buffers"}

    def __init__(self, config: Dict[str, Any]):
        if "database_config" in config:
            self.config = config["database_config"]
        else:
            self.config = config

        self.conn = None
        self.cursor = None
        self.logger = logging.getLogger("Database")
        self._connect()

    def _connect(self):
        """Create a new autocommit connection."""
        self.conn = psycopg2.connect(
            host=self.config.get("host", "localhost"),
            port=int(self.config.get("port", 5432)),
            database=self.config.get("database", "postgres"),
            user=self.config.get("username", "postgres"),
            password=self.config.get("password", "postgres"),
        )
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        self.logger.info(
            "Connected to PostgreSQL %s@%s",
            self.config.get("database"),
            self.config.get("host"),
        )

    def reconnect(self):
        self.close()
        self._connect()

    def requires_fresh_session(self, config_dict: Dict[str, Any]) -> bool:
        """
        Some parameters are technically user-context/session-settable, but only
        before specific session activity has happened. temp_buffers is the most
        important example for this project.
        """
        return any(param_name in self.SESSION_INIT_PARAMS for param_name in config_dict)

    def prepare_session_for_config(
        self,
        config_dict: Dict[str, Any],
        force_new_session: bool = False,
    ) -> bool:
        """
        Ensure the current session is suitable for parameter application.

        We reconnect when:
        - the caller explicitly requests a fresh session
        - the config contains session-initialization-sensitive parameters such
          as temp_buffers
        """
        if force_new_session or self.requires_fresh_session(config_dict):
            self.logger.info("Reconnecting PostgreSQL session before applying parameters")
            self.reconnect()
            return True
        return False

    def execute_query(self, query: str) -> List[tuple]:
        """
        Execute a SQL statement that may return rows.

        psycopg2 exposes `cursor.description is None` for statements without a
        result set, so we return an empty list in that case instead of calling
        fetchall() and raising `no results to fetch`.
        """
        try:
            if self.cursor is None:
                raise RuntimeError("Database session is not connected")
            self.cursor.execute(query)
            if self.cursor.description is None:
                return []

            result = self.cursor.fetchall()
            converted_result = []
            for row in result:
                converted_row = tuple(
                    float(val) if hasattr(val, "__float__") and not isinstance(val, bool) else val
                    for val in row
                )
                converted_result.append(converted_row)
            return converted_result
        except Exception as exc:
            self.logger.error("Query failed: %s", exc)
            raise

    def execute_command(self, query: str) -> None:
        """Execute a SQL statement that does not need a fetched result set."""
        try:
            if self.cursor is None:
                raise RuntimeError("Database session is not connected")
            self.cursor.execute(query)
        except Exception as exc:
            self.logger.error("Command failed: %s", exc)
            raise

    def normalize_parameter_value(
        self,
        metadata: Dict[str, Any],
        value: Any,
    ) -> Tuple[Any, bool]:
        """Public wrapper used by the parameter validation subsystem."""
        return self._normalize_parameter_value(metadata, value)

    def _escape_literal(self, value: str) -> str:
        return value.replace("'", "''")

    def _to_sql_literal(self, value: Any) -> str:
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, str):
            return f"'{self._escape_literal(value)}'"
        return str(value)

    def _parse_numeric(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def get_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        rows = self.execute_query(
            "SELECT name, setting, unit, context, vartype, min_val, max_val, "
            "boot_val, reset_val, pending_restart "
            f"FROM pg_settings WHERE name = '{self._escape_literal(param_name)}' LIMIT 1"
        )
        if not rows:
            return None

        (
            name,
            setting,
            unit,
            context,
            vartype,
            min_val,
            max_val,
            boot_val,
            reset_val,
            pending_restart,
        ) = rows[0]

        return {
            "name": name,
            "setting": setting,
            "unit": unit or "",
            "context": context,
            "vartype": vartype,
            "min_val": min_val,
            "max_val": max_val,
            "boot_val": boot_val,
            "reset_val": reset_val,
            "pending_restart": bool(pending_restart),
        }

    def _normalize_parameter_value(
        self, metadata: Dict[str, Any], value: Any
    ) -> Tuple[Any, bool]:
        """
        Normalize a parameter value to a PostgreSQL-friendly representation.

        Returns `(normalized_value, was_clamped)`.
        """
        vartype = metadata["vartype"]
        was_clamped = False

        if vartype == "bool":
            if isinstance(value, str):
                lowered = value.strip().lower()
                normalized = lowered in {"1", "true", "on", "yes"}
            else:
                normalized = bool(value)
            return normalized, False

        numeric_value = self._parse_numeric(value)
        min_numeric = self._parse_numeric(metadata.get("min_val"))
        max_numeric = self._parse_numeric(metadata.get("max_val"))

        if vartype in {"integer", "real"} and numeric_value is not None:
            if min_numeric is not None and numeric_value < min_numeric:
                numeric_value = min_numeric
                was_clamped = True
            if max_numeric is not None and numeric_value > max_numeric:
                numeric_value = max_numeric
                was_clamped = True

            if vartype == "integer":
                return int(round(numeric_value)), was_clamped
            return float(numeric_value), was_clamped

        return value, False

    def _verify_parameter_state(self, param_name: str, expected_value: Any) -> bool:
        metadata = self.get_parameter_info(param_name)
        if metadata is None:
            return False

        actual = metadata["setting"]
        expected_numeric = self._parse_numeric(expected_value)
        actual_numeric = self._parse_numeric(actual)

        if expected_numeric is not None and actual_numeric is not None:
            if isinstance(expected_value, int):
                return int(round(actual_numeric)) == int(round(expected_numeric))
            return abs(actual_numeric - expected_numeric) < 1e-9

        if isinstance(expected_value, bool):
            return str(actual).lower() in {"on", "true", "1"} if expected_value else str(actual).lower() in {
                "off",
                "false",
                "0",
            }

        return str(actual) == str(expected_value)

    def run_health_checks(self) -> Dict[str, Any]:
        """
        Run a lightweight but practical PostgreSQL health check.

        This is intentionally more than a `SELECT 1`: it also checks whether
        configuration parsing is healthy and whether simple read/write database
        functionality still works after a reload or restart.
        """
        report: Dict[str, Any] = {
            "healthy": False,
            "checks": {},
            "errors": [],
        }

        def mark(name: str, ok: bool, detail: Any = None):
            report["checks"][name] = {"ok": ok, "detail": detail}
            if not ok:
                report["errors"].append(f"{name}:{detail}")

        if self.cursor is None:
            mark("connectivity", False, "Database session is not connected")
            report["healthy"] = False
            return report

        try:
            self.execute_query("SELECT 1")
            mark("connectivity", True, "SELECT 1 succeeded")
        except Exception as exc:
            mark("connectivity", False, str(exc))
            report["healthy"] = False
            return report

        try:
            current_db = self.execute_query("SELECT current_database()")[0][0]
            mark("current_database", bool(current_db), current_db)
        except Exception as exc:
            mark("current_database", False, str(exc))

        try:
            read_only = self.execute_query("SHOW transaction_read_only")[0][0]
            mark("transaction_read_only", str(read_only).lower() == "off", read_only)
        except Exception as exc:
            mark("transaction_read_only", False, str(exc))

        try:
            self.execute_command("CREATE TEMP TABLE codex_healthcheck(id int)")
            self.execute_command("INSERT INTO codex_healthcheck VALUES (1)")
            count = self.execute_query("SELECT count(*) FROM codex_healthcheck")[0][0]
            self.execute_command("DROP TABLE codex_healthcheck")
            mark("temp_table_rw", int(count) == 1, count)
        except Exception as exc:
            mark("temp_table_rw", False, str(exc))

        try:
            rows = self.execute_query(
                "SELECT name, error FROM pg_file_settings WHERE error IS NOT NULL LIMIT 10"
            )
            mark("pg_file_settings", len(rows) == 0, rows)
        except Exception as exc:
            # Some environments may restrict this view; record but don't explode.
            mark("pg_file_settings", False, str(exc))

        try:
            pending = self.execute_query(
                "SELECT count(*) FROM pg_settings WHERE pending_restart"
            )[0][0]
            mark("pending_restart_count", True, int(pending))
        except Exception as exc:
            mark("pending_restart_count", False, str(exc))

        report["healthy"] = all(item["ok"] for item in report["checks"].values())
        return report

    def _build_restore_plan(
        self, original_values: Dict[str, Any]
    ) -> Dict[str, List[Tuple[str, Any]]]:
        restore_plan = {"dynamic": [], "reload": [], "restart": []}
        for param_name, original_value in original_values.items():
            metadata = self.get_parameter_info(param_name)
            if metadata is None:
                continue
            context = metadata["context"]
            if context in self.DYNAMIC_CONTEXTS:
                restore_plan["dynamic"].append((param_name, original_value))
            elif context in self.RELOAD_CONTEXTS:
                restore_plan["reload"].append((param_name, original_value))
            elif context in self.RESTART_CONTEXTS:
                restore_plan["restart"].append((param_name, original_value))
        return restore_plan

    def restore_original_values(
        self,
        original_values: Dict[str, Any],
        restart_if_needed: bool = True,
    ) -> Dict[str, Any]:
        """
        Best-effort rollback used after a failed post-apply health check.
        """
        restore_stats = {
            "restored": 0,
            "failed": 0,
            "reloaded": False,
            "restarted": False,
        }
        restore_plan = self._build_restore_plan(original_values)

        try:
            for param_name, value in restore_plan["dynamic"]:
                self.execute_command(f"SET {param_name} = {self._to_sql_literal(value)}")
                restore_stats["restored"] += 1

            for param_name, value in restore_plan["reload"]:
                self.execute_command(f"ALTER SYSTEM SET {param_name} = {self._to_sql_literal(value)}")
                restore_stats["restored"] += 1

            for param_name, value in restore_plan["restart"]:
                self.execute_command(f"ALTER SYSTEM SET {param_name} = {self._to_sql_literal(value)}")
                restore_stats["restored"] += 1

            if restore_plan["reload"]:
                restore_stats["reloaded"] = self.reload_config()

            if restore_plan["restart"] and restart_if_needed:
                restore_stats["restarted"] = self.restart()

        except Exception as exc:
            self.logger.error("Rollback failed: %s", exc)
            restore_stats["failed"] += 1

        return restore_stats

    def apply_config(
        self,
        config_dict: Dict[str, Any],
        apply_static: bool = False,
        restart_if_static: bool = False,
        reload_if_needed: bool = True,
        verify: bool = True,
        health_check: bool = True,
        rollback_on_failure: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply a parameter dictionary with explicit handling for each context.

        Dynamic parameters use `SET`.
        Reload-required parameters use `ALTER SYSTEM` + `pg_reload_conf()`.
        Restart-required parameters use `ALTER SYSTEM` and optionally restart.
        """
        stats: Dict[str, Any] = {
            "dynamic": 0,
            "static": 0,
            "reload": 0,
            "skipped": 0,
            "failed": 0,
            "clamped": 0,
            "verified": 0,
            "restarted": False,
            "reloaded": False,
            "health_ok": None,
            "health_report": None,
            "rollback": None,
            "requires_manual_recovery": False,
            "details": [],
        }

        reload_params: List[Tuple[str, Any]] = []
        restart_params: List[Tuple[str, Any]] = []
        dynamic_params: List[Tuple[str, Any]] = []
        original_values: Dict[str, Any] = {}

        for param_name, raw_value in config_dict.items():
            try:
                metadata = self.get_parameter_info(param_name)
                if metadata is None:
                    stats["failed"] += 1
                    stats["details"].append((param_name, "unknown"))
                    continue

                context = metadata["context"]
                original_values[param_name] = metadata["setting"]
                normalized_value, was_clamped = self._normalize_parameter_value(metadata, raw_value)
                sql_literal = self._to_sql_literal(normalized_value)

                if was_clamped:
                    stats["clamped"] += 1
                    self.logger.warning(
                        "Parameter %s exceeded pg_settings bounds and was clamped to %s",
                        param_name,
                        normalized_value,
                    )

                if context in self.DYNAMIC_CONTEXTS:
                    self.execute_command(f"SET {param_name} = {sql_literal}")
                    stats["dynamic"] += 1
                    dynamic_params.append((param_name, normalized_value))
                    continue

                if context in self.RELOAD_CONTEXTS:
                    if not apply_static:
                        stats["skipped"] += 1
                        stats["details"].append((param_name, f"reload_required:{context}"))
                        continue
                    self.execute_command(f"ALTER SYSTEM SET {param_name} = {sql_literal}")
                    stats["reload"] += 1
                    reload_params.append((param_name, normalized_value))
                    continue

                if context in self.RESTART_CONTEXTS:
                    if not apply_static:
                        stats["skipped"] += 1
                        stats["details"].append((param_name, f"restart_required:{context}"))
                        continue
                    self.execute_command(f"ALTER SYSTEM SET {param_name} = {sql_literal}")
                    stats["static"] += 1
                    restart_params.append((param_name, normalized_value))
                    continue

                stats["skipped"] += 1
                stats["details"].append((param_name, f"unsupported_context:{context}"))

            except Exception as exc:
                stats["failed"] += 1
                stats["details"].append((param_name, f"error:{exc}"))
                self.logger.warning("Failed to apply %s=%s: %s", param_name, raw_value, exc)

        if reload_params and reload_if_needed:
            self.reload_config()
            stats["reloaded"] = True

        if restart_params and restart_if_static:
            stats["restarted"] = self.restart()
            if not stats["restarted"]:
                self.logger.error("Restart-required parameters were written but PostgreSQL restart failed")
                stats["health_ok"] = False
                stats["details"].append(("restart", "restart_failed"))
                stats["requires_manual_recovery"] = True
                return stats

        if verify:
            params_to_verify: List[Tuple[str, Any]] = []
            params_to_verify.extend(dynamic_params)
            params_to_verify.extend(reload_params if stats["reloaded"] else [])
            params_to_verify.extend(restart_params if stats["restarted"] else [])

            for param_name, expected in params_to_verify:
                try:
                    if self._verify_parameter_state(param_name, expected):
                        stats["verified"] += 1
                    else:
                        stats["details"].append((param_name, "verify_failed"))
                except Exception as exc:
                    stats["details"].append((param_name, f"verify_error:{exc}"))

        if health_check:
            health_report = self.run_health_checks()
            stats["health_report"] = health_report
            stats["health_ok"] = health_report["healthy"]
            if not health_report["healthy"] and rollback_on_failure:
                self.logger.error("Post-apply health check failed, starting rollback")
                stats["rollback"] = self.restore_original_values(
                    original_values,
                    restart_if_needed=restart_if_static,
                )

        return stats

    def reset_system_parameter(self, param_name: str, restart_if_static: bool = False) -> bool:
        """
        Reset a parameter previously written through ALTER SYSTEM.
        """
        metadata = self.get_parameter_info(param_name)
        if metadata is None:
            return False

        self.execute_command(f"ALTER SYSTEM RESET {param_name}")

        if metadata["context"] in self.RELOAD_CONTEXTS:
            self.reload_config()
            return True

        if metadata["context"] in self.RESTART_CONTEXTS and restart_if_static:
            return self.restart()

        return True

    def restart(self, max_wait_seconds: int = 90) -> bool:
        """
        Restart PostgreSQL and wait until a new connection succeeds.

        The method tries a configurable restart command first, then a small set
        of common Linux service commands.
        """
        self.close()

        restart_command = self.config.get("restart_command")
        service_name = self.config.get("service_name", "postgresql")
        pg_version = str(self.config.get("pg_version", "18"))

        command_attempts: List[Tuple[List[str], bool]] = []
        if restart_command:
            command_attempts.append(([restart_command], True))

        command_attempts.extend(
            [
                (["systemctl", "restart", service_name], False),
                (["service", service_name, "restart"], False),
                (["pg_ctlcluster", pg_version, "main", "restart"], False),
            ]
        )

        for command, use_shell in command_attempts:
            try:
                if use_shell:
                    subprocess.run(command[0], shell=True, check=True, timeout=60)
                else:
                    subprocess.run(command, check=True, timeout=60)
                self.logger.info("Restart command succeeded: %s", command[0] if use_shell else " ".join(command))
                return self.wait_for_restart(max_wait_seconds=max_wait_seconds)
            except Exception as exc:
                self.logger.warning(
                    "Restart command failed: %s (%s)",
                    command[0] if use_shell else " ".join(command),
                    exc,
                )

        self.logger.error("All PostgreSQL restart attempts failed")
        return False

    def wait_for_restart(self, max_wait_seconds: int = 90) -> bool:
        start_time = time.time()
        last_error = None

        while time.time() - start_time < max_wait_seconds:
            try:
                self._connect()
                self.execute_query("SELECT 1")
                self.logger.info("PostgreSQL restart completed and connection is healthy")
                return True
            except Exception as exc:
                last_error = exc
                time.sleep(2)

        self.logger.error("Timed out waiting for PostgreSQL restart: %s", last_error)
        return False

    def reload_config(self) -> bool:
        try:
            self.execute_query("SELECT pg_reload_conf()")
            self.logger.info("PostgreSQL configuration reloaded")
            return True
        except Exception as exc:
            self.logger.error("Failed to reload PostgreSQL config: %s", exc)
            return False

    def get_system_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        try:
            result = self.execute_query(
                "SELECT sum(heap_blks_read), sum(heap_blks_hit) FROM pg_statio_user_tables"
            )
            if result and result[0][0] is not None and result[0][1] is not None:
                heap_read = float(result[0][0] or 0)
                heap_hit = float(result[0][1] or 0)
                total_accesses = heap_read + heap_hit
                metrics["heap_blks_read"] = heap_read
                metrics["heap_blks_hit"] = heap_hit
                metrics["cache_hit_ratio"] = round(heap_hit / total_accesses, 4) if total_accesses > 0 else 0.5

            result = self.execute_query(
                f"SELECT pg_database_size('{self.config.get('database')}')"
            )
            if result and result[0][0] is not None:
                metrics["database_size"] = float(result[0][0])

            result = self.execute_query("SELECT count(*) FROM pg_stat_activity")
            if result:
                metrics["active_connections"] = float(result[0][0] or 0)

            result = self.execute_query(
                f"SELECT xact_commit, xact_rollback FROM pg_stat_database "
                f"WHERE datname = '{self.config.get('database')}'"
            )
            if result:
                metrics["xact_commit"] = float(result[0][0] or 0)
                metrics["xact_rollback"] = float(result[0][1] or 0)

            result = self.execute_query(
                "SELECT sum(seq_tup_read), sum(idx_tup_fetch), sum(n_tup_ins), "
                "sum(n_tup_upd), sum(n_tup_del), sum(n_live_tup), sum(n_dead_tup) "
                "FROM pg_stat_user_tables"
            )
            if result:
                metrics["seq_tup_read"] = float(result[0][0] or 0)
                metrics["idx_tup_fetch"] = float(result[0][1] or 0)
                metrics["tup_inserted"] = float(result[0][2] or 0)
                metrics["tup_updated"] = float(result[0][3] or 0)
                metrics["tup_deleted"] = float(result[0][4] or 0)
                metrics["live_tuples"] = float(result[0][5] or 0)
                metrics["dead_tuples"] = float(result[0][6] or 0)

            try:
                result = self.execute_query("SELECT sum(idx_blks_read) FROM pg_statio_user_indexes")
                metrics["disk_read_count"] = float(result[0][0] or 0) if result else 0.0
            except Exception:
                metrics["disk_read_count"] = 0.0

            metrics["cpu_usage"] = 50.0
            try:
                check_result = self.execute_query(
                    "SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements' LIMIT 1"
                )
                if check_result:
                    result = self.execute_query(
                        "SELECT sum(mean_exec_time) FROM pg_stat_statements "
                        "WHERE query NOT LIKE '%pg_stat%' LIMIT 10"
                    )
                    if result and result[0][0] is not None:
                        metrics["cpu_usage"] = min(float(result[0][0]) / 100, 100.0)
            except Exception as exc:
                self.logger.debug("pg_stat_statements not available: %s", exc)

        except Exception as exc:
            self.logger.warning("Failed to collect system metrics: %s", exc)

        return metrics

    def get_parameters(self) -> Dict[str, Any]:
        try:
            result = self.execute_query("SELECT name, setting FROM pg_settings")
            return {row[0]: row[1] for row in result}
        except Exception as exc:
            self.logger.error("Failed to fetch parameters: %s", exc)
            return {}

    def close(self):
        if self.cursor is not None:
            try:
                self.cursor.close()
            except Exception:
                pass
            self.cursor = None

        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
