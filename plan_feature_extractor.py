"""
plan_feature_extractor.py – Pure-utility module for PostgreSQL EXPLAIN plan parsing.

No database connections, no file I/O, no global state mutation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def parse_plan_node(
    node: dict,
    depth: int = 0,
    max_depth: int = 8,
) -> list[dict]:
    """
    Recursively traverse a PostgreSQL JSON-format plan node dict.

    Returns a flat list of node dicts, each with keys:
        node_type, total_cost, plan_rows, table, depth
    """
    if depth >= max_depth:
        return [
            {
                "node_type": "...(depth_limit)",
                "total_cost": 0.0,
                "plan_rows": 0,
                "table": None,
                "depth": depth,
            }
        ]

    node_type = node.get("Node Type", "Unknown")
    total_cost = float(node.get("Total Cost", 0.0))
    plan_rows = int(node.get("Plan Rows", 0))

    raw_table = node.get("Relation Name") or node.get("Index Name") or None
    table: Optional[str] = raw_table if raw_table else None

    result: list[dict] = [
        {
            "node_type": node_type,
            "total_cost": total_cost,
            "plan_rows": plan_rows,
            "table": table,
            "depth": depth,
        }
    ]

    for child in node.get("Plans", []):
        result.extend(parse_plan_node(child, depth=depth + 1, max_depth=max_depth))

    return result


def extract_plan_summary(
    plan_nodes: list[dict],
    all_workload_costs: Optional[list[float]] = None,
) -> dict:
    """
    Summarise a flat node list returned by parse_plan_node.

    Parameters
    ----------
    plan_nodes:
        Flat list produced by parse_plan_node for a single SQL statement.
    all_workload_costs:
        All total_cost values from the root node of every plan in the workload.
        Used to compute cost_bucket via percentile thresholds.  When None or
        fewer than 2 values, cost_bucket defaults to "M".

    Returns
    -------
    dict with keys: top_node, total_cost, cost_bucket, plan_depth, node_count,
                    scan_types, join_types, top_tables
    """
    if not plan_nodes:
        return {
            "top_node": "Unknown",
            "total_cost": 0.0,
            "cost_bucket": "M",
            "plan_depth": 0,
            "node_count": 0,
            "scan_types": [],
            "join_types": [],
            "top_tables": [],
        }

    top_node = plan_nodes[0]["node_type"]
    total_cost = plan_nodes[0]["total_cost"]
    plan_depth = max(n["depth"] for n in plan_nodes)
    node_count = len(plan_nodes)

    # Deduplicated scan/join types (insertion order preserved)
    scan_types: list[str] = []
    join_types: list[str] = []
    seen_scan: set[str] = set()
    seen_join: set[str] = set()
    for n in plan_nodes:
        nt = n["node_type"]
        if "Scan" in nt and nt not in seen_scan:
            seen_scan.add(nt)
            scan_types.append(nt)
        if "Join" in nt and nt not in seen_join:
            seen_join.add(nt)
            join_types.append(nt)

    # Top tables ordered by descending total_cost, max 5
    tables_with_cost: list[tuple[str, float]] = [
        (n["table"], n["total_cost"])
        for n in plan_nodes
        if n["table"] is not None
    ]
    tables_with_cost.sort(key=lambda x: x[1], reverse=True)
    seen_tables: set[str] = set()
    top_tables: list[str] = []
    for tbl, _ in tables_with_cost:
        if tbl not in seen_tables:
            seen_tables.add(tbl)
            top_tables.append(tbl)
            if len(top_tables) >= 5:
                break

    # Cost bucket
    cost_bucket = _compute_cost_bucket(total_cost, all_workload_costs)

    return {
        "top_node": top_node,
        "total_cost": total_cost,
        "cost_bucket": cost_bucket,
        "plan_depth": plan_depth,
        "node_count": node_count,
        "scan_types": scan_types,
        "join_types": join_types,
        "top_tables": top_tables,
    }


def _compute_cost_bucket(
    cost: float,
    all_costs: Optional[list[float]],
) -> str:
    """Return 'S', 'M', or 'L' based on workload-relative percentile thresholds."""
    if all_costs is None or len(all_costs) < 2:
        return "M"
    arr = np.array(all_costs, dtype=float)
    p33, p66 = np.percentile(arr, [33, 66])
    if cost <= p33:
        return "S"
    if cost <= p66:
        return "M"
    return "L"


def format_plan_compact(
    plans: list[dict],
    max_nodes: int = 12,
) -> str:
    """
    Produce a compact single-line representation of the most expensive nodes
    across all plans in a workload.

    Parameters
    ----------
    plans:
        List of per-SQL dicts, each having 'plan_summary' and 'plan_nodes'.
    max_nodes:
        Maximum number of nodes to include in the output.

    Returns
    -------
    A ` | `-joined string of node tokens, e.g.:
        [Aggregate:L] | [Seq Scan:L lineitem] | [Hash Join:M]
    Returns "N/A" if no usable nodes are found.
    """
    # Build sql_index → cost_bucket lookup
    bucket_by_index: dict[int, str] = {}
    for plan in plans:
        summary = plan.get("plan_summary")
        if summary is not None:
            bucket_by_index[plan.get("sql_index", 0)] = summary.get("cost_bucket", "M")

    # Collect all nodes from valid plans
    all_nodes: list[tuple[dict, int]] = []  # (node_dict, sql_index)
    for plan in plans:
        if plan.get("plan_summary") is None:
            continue
        nodes = plan.get("plan_nodes")
        if not nodes:
            continue
        sql_index = plan.get("sql_index", 0)
        for node in nodes:
            all_nodes.append((node, sql_index))

    if not all_nodes:
        return "N/A"

    # Sort by total_cost descending, take top max_nodes
    all_nodes.sort(key=lambda x: x[0]["total_cost"], reverse=True)
    top = all_nodes[:max_nodes]

    tokens: list[str] = []
    for node, sql_index in top:
        nt = node["node_type"]
        bucket = bucket_by_index.get(sql_index, "M")
        table = node.get("table")
        if table:
            tokens.append(f"[{nt}:{bucket} {table}]")
        else:
            tokens.append(f"[{nt}:{bucket}]")

    result = " | ".join(tokens)
    return result if result else "N/A"
