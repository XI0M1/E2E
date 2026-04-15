"""
explain_cache_builder.py – Standalone post-processing script.

Connects to PostgreSQL, scans .wg workload files, runs EXPLAIN (FORMAT JSON)
on each SQL, and stores structured results in
explain_cache/{database}/{workload_id}.json.

Usage example
-------------
python explain_cache_builder.py \\
    --database tpch \\
    --datapath data/olap/tpch \\
    --host 127.0.0.1 \\
    --port 5432 \\
    --username dbuser \\
    --password YOUR_PASSWORD \\
    --config config/cloud.ini \\
    --output-dir explain_cache \\
    --max-sqls-per-workload 5 \\
    --skip-existing \\
    --timeout 30
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import psycopg2

from plan_feature_extractor import extract_plan_summary, parse_plan_node


# Maximum characters to store for SQL text in plan entries
_SQL_TEXT_MAX_CHARS = 200

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build EXPLAIN plan cache from .wg workload files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database", required=True, help="Database name (also used as workload prefix filter)")
    parser.add_argument("--datapath", required=True, help="Directory containing .wg workload files")
    parser.add_argument("--host", default=None, help="PostgreSQL host (overrides --config)")
    parser.add_argument("--port", type=int, default=None, help="PostgreSQL port (overrides --config)")
    parser.add_argument("--username", default=None, help="PostgreSQL username (overrides --config)")
    parser.add_argument("--password", default=None, help="PostgreSQL password (overrides --config)")
    parser.add_argument("--config", default="config/cloud.ini", help="INI config file path")
    parser.add_argument("--output-dir", default="explain_cache", help="Root directory for explain cache JSON files")
    parser.add_argument("--max-sqls-per-workload", type=int, default=5, help="Max SQL statements per workload")
    parser.add_argument("--skip-existing", action="store_true", help="Skip workloads whose cache file already exists and is valid JSON")
    parser.add_argument("--timeout", type=int, default=30, help="Per-statement timeout in seconds")
    return parser


def _load_db_params_from_config(config_path: str) -> dict:
    """Load database connection parameters from INI config file."""
    params: dict = {}
    if not os.path.exists(config_path):
        return params
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    section = "database_config"
    if cfg.has_section(section):
        for key, ini_key in [
            ("host", "host"),
            ("port", "port"),
            ("dbname", "database"),
            ("user", "username"),
            ("password", "password"),
        ]:
            if cfg.has_option(section, ini_key):
                params[key] = cfg.get(section, ini_key)
    return params


def _build_db_params(args: argparse.Namespace) -> dict:
    """Merge config-file defaults with explicit CLI overrides."""
    params = _load_db_params_from_config(args.config)
    # CLI overrides
    if args.host is not None:
        params["host"] = args.host
    if args.port is not None:
        params["port"] = str(args.port)
    if args.username is not None:
        params["user"] = args.username
    if args.password is not None:
        params["password"] = args.password
    # Always use the --database argument as the target DB
    params["dbname"] = args.database
    return params


# ---------------------------------------------------------------------------
# SQL parsing helpers
# ---------------------------------------------------------------------------

def _parse_sqls_from_wg(wg_path: str, max_sqls: int) -> list[str]:
    """
    Parse SQL statements from a .wg file.

    Rules:
    - Split on ';'
    - Strip whitespace
    - Skip lines starting with '--' or '#'
    - Discard empty results
    - Return at most max_sqls statements
    """
    try:
        with open(wg_path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError as exc:
        raise RuntimeError(f"Cannot read workload file {wg_path}: {exc}") from exc

    statements: list[str] = []
    for chunk in raw.split(";"):
        lines = chunk.splitlines()
        filtered = [
            line for line in lines
            if line.strip() and not line.strip().startswith("--") and not line.strip().startswith("#")
        ]
        sql = "\n".join(filtered).strip()
        if sql:
            statements.append(sql)
        if len(statements) >= max_sqls:
            break
    return statements


# ---------------------------------------------------------------------------
# EXPLAIN execution
# ---------------------------------------------------------------------------

def _get_pg_version(conn) -> str:
    """Return the PostgreSQL major.minor version string."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            row = cur.fetchone()
            if row:
                # e.g. "PostgreSQL 14.5 on x86_64-pc-linux-gnu ..."
                parts = str(row[0]).split()
                if len(parts) >= 2:
                    return parts[1]
        return "unknown"
    except Exception:
        return "unknown"


def _run_explain(conn, sql: str, timeout_ms: int) -> tuple[Optional[dict], Optional[str]]:
    """
    Run EXPLAIN (FORMAT JSON) for *sql*.

    Returns (root_plan_node, None) on success, or (None, error_message) on failure.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
            cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
            row = cur.fetchone()
            if not row:
                return None, "EXPLAIN returned no rows"
            plan_json = json.loads(row[0])
            root_node = plan_json[0]["Plan"]
            return root_node, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Cost bucketing helpers
# ---------------------------------------------------------------------------

def _assign_cost_buckets(plans: list[dict]) -> None:
    """
    Assign cost_bucket to every plan's plan_summary in-place.

    Uses percentile thresholds across all plans in this workload.
    """
    costs = [
        p["plan_summary"]["total_cost"]
        for p in plans
        if p.get("plan_summary") is not None
    ]
    for plan in plans:
        if plan.get("plan_summary") is None:
            continue
        from plan_feature_extractor import _compute_cost_bucket
        plan["plan_summary"]["cost_bucket"] = _compute_cost_bucket(
            plan["plan_summary"]["total_cost"],
            costs if len(costs) >= 2 else None,
        )


# ---------------------------------------------------------------------------
# Per-workload processing
# ---------------------------------------------------------------------------

def _process_workload(
    wg_path: str,
    workload_id: str,
    database: str,
    conn,
    max_sqls: int,
    timeout_sec: int,
    output_dir: str,
    skip_existing: bool,
) -> tuple[str, str]:
    """
    Process a single workload file and write the explain cache.

    Returns (status, message) where status is 'ok', 'skip', or 'error'.
    """
    cache_path = os.path.join(output_dir, database, f"{workload_id}.json")

    if skip_existing and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                json.load(fh)
            return "skip", f"[SKIP] {workload_id} (cached)"
        except Exception:
            pass  # File is corrupt; re-generate

    t0 = time.monotonic()

    sqls = _parse_sqls_from_wg(wg_path, max_sqls)
    pg_version = _get_pg_version(conn)
    timeout_ms = timeout_sec * 1000

    plans: list[dict] = []
    for i, sql in enumerate(sqls, 1):
        root_node, err = _run_explain(conn, sql, timeout_ms)
        if err is not None or root_node is None:
            plans.append(
                {
                    "sql_index": i,
                    "sql_text": sql[:_SQL_TEXT_MAX_CHARS],
                    "error": err or "No root node",
                    "plan_summary": None,
                    "plan_nodes": [],
                }
            )
            continue

        plan_nodes = parse_plan_node(root_node, max_depth=8)
        # Compute summary without cross-workload costs for now; buckets patched later
        plan_summary = extract_plan_summary(plan_nodes, all_workload_costs=None)
        plans.append(
            {
                "sql_index": i,
                "sql_text": sql[:_SQL_TEXT_MAX_CHARS],
                "error": None,
                "plan_summary": plan_summary,
                "plan_nodes": plan_nodes,
            }
        )

    # Assign workload-relative cost buckets now that all plans are processed
    _assign_cost_buckets(plans)

    total_nodes = sum(len(p["plan_nodes"]) for p in plans)
    elapsed = time.monotonic() - t0

    output: dict = {
        "workload_id": workload_id,
        "workload_file": os.path.basename(wg_path),
        "database": database,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pg_version": pg_version,
        "sql_count": len(plans),
        "plans": plans,
    }

    # Atomic write
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, cache_path)

    msg = f"[OK] {workload_id}: {len(sqls)} SQLs, {total_nodes} nodes, elapsed={elapsed:.2f}s"
    return "ok", msg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    db_params = _build_db_params(args)

    # Discover .wg files
    datapath = args.datapath
    if not os.path.isdir(datapath):
        print(f"[ERROR] --datapath '{datapath}' is not a directory.", file=sys.stderr)
        return 1

    wg_files: list[tuple[str, str]] = []  # (workload_id, full_path)
    for entry in os.scandir(datapath):
        if entry.is_file() and entry.name.endswith(".wg") and entry.name.startswith(args.database):
            workload_id = os.path.splitext(entry.name)[0]
            wg_files.append((workload_id, entry.path))

    wg_files.sort(key=lambda x: x[0])

    if not wg_files:
        print(f"[WARN] No .wg files found in '{datapath}' matching prefix '{args.database}'.")
        return 0

    print(f"Found {len(wg_files)} workload file(s) in '{datapath}'.")

    # Connect once
    try:
        conn = psycopg2.connect(**{k: v for k, v in db_params.items() if v is not None})
        conn.autocommit = True
    except Exception as exc:
        print(f"[ERROR] Cannot connect to PostgreSQL: {exc}", file=sys.stderr)
        return 1

    ok_count = skip_count = error_count = 0
    try:
        for workload_id, wg_path in wg_files:
            try:
                status, msg = _process_workload(
                    wg_path=wg_path,
                    workload_id=workload_id,
                    database=args.database,
                    conn=conn,
                    max_sqls=args.max_sqls_per_workload,
                    timeout_sec=args.timeout,
                    output_dir=args.output_dir,
                    skip_existing=args.skip_existing,
                )
                print(msg)
                if status == "ok":
                    ok_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    error_count += 1
            except Exception as exc:
                print(f"[ERROR] {workload_id}: {exc}")
                error_count += 1
    finally:
        conn.close()

    print(
        f"\nDone. ok={ok_count}, skipped={skip_count}, errors={error_count}"
    )
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
