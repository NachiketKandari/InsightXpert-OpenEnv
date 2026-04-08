"""Reward computation for BIRD Text-to-SQL environment.

Tiered reward scheme:
  Tier 0 (0.00-0.15): Syntax/runtime errors
  Tier 1 (0.20-0.40): Structural match (columns, row count)
  Tier 2 (0.50-0.90): Partial value match (Soft-F1)
  Tier 3 (1.00):      Perfect execution match
"""

from __future__ import annotations

import re


def _normalize_value(v: object) -> str:
    """Normalize a single cell value for comparison."""
    if v is None:
        return "NULL"
    s = str(v).strip()
    try:
        f = float(s)
        return f"{f:.2f}"
    except (ValueError, OverflowError):
        return s.lower()


def _rows_to_set(rows: list[tuple]) -> set[tuple[str, ...]]:
    """Convert result rows to a normalized set of string tuples."""
    return {tuple(_normalize_value(v) for v in row) for row in rows}


def _rows_to_list(rows: list[tuple]) -> list[tuple[str, ...]]:
    """Convert result rows to a normalized ordered list of string tuples."""
    return [tuple(_normalize_value(v) for v in row) for row in rows]


def _has_order_by(sql: str) -> bool:
    """Check if SQL contains a top-level ORDER BY (not inside a subquery)."""
    depth = 0
    outer: list[str] = []
    for ch in sql:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            outer.append(ch)
    outer_sql = "".join(outer)
    return bool(re.search(r"\bORDER\s+BY\b", outer_sql, re.IGNORECASE))


def compute_reward(
    gold_results: list[tuple] | None,
    agent_results: list[tuple] | None,
    agent_error: str | None,
    gold_columns: list[str],
    agent_columns: list[str],
    gold_sql: str = "",
) -> tuple[float, str]:
    """Compute reward and feedback for an agent's SQL attempt.

    Returns:
        (reward, feedback) where reward is 0.0-1.0 and feedback is a string.
    """
    # Tier 0: Error cases
    if agent_error is not None:
        error_lower = agent_error.lower()
        if "syntax" in error_lower or "parse" in error_lower:
            return 0.05, f"Syntax error: {agent_error}"
        return 0.10, f"Runtime error: {agent_error}"

    if agent_results is None:
        return 0.00, "No valid SQL query provided"

    if not agent_results and gold_results:
        return 0.15, "Query returned no results (expected rows)"

    if not gold_results and not agent_results:
        return 1.0, "Perfect match! (both empty)"

    if not gold_results:
        return 0.15, "Gold query returned no results but yours did"

    # Tier 1: Structural match
    col_score = 0.0
    col_feedback = ""
    if len(agent_columns) == len(gold_columns):
        col_score = 0.20
        col_feedback = "Column count matches"
        agent_col_set = {c.lower() for c in agent_columns}
        gold_col_set = {c.lower() for c in gold_columns}
        if agent_col_set == gold_col_set:
            col_score = 0.30
            col_feedback = "Column names match"
            if len(agent_results) == len(gold_results):
                col_score = 0.40
                col_feedback = "Structure matches, checking values"

    # Tier 2/3: Value matching
    order_sensitive = _has_order_by(gold_sql)

    if order_sensitive:
        gold_norm = _rows_to_list(gold_results)
        agent_norm = _rows_to_list(agent_results)
        if gold_norm == agent_norm:
            return 1.0, "Perfect match!"
    else:
        gold_set = _rows_to_set(gold_results)
        agent_set = _rows_to_set(agent_results)
        if gold_set == agent_set:
            return 1.0, "Perfect match!"

    # Soft-F1 on result sets (always set-based for partial credit)
    gold_set = _rows_to_set(gold_results)
    agent_set = _rows_to_set(agent_results)

    if not agent_set or not gold_set:
        return max(col_score, 0.15), "No value overlap"

    intersection = gold_set & agent_set
    if not intersection:
        # Check single-value numeric closeness
        if (
            len(gold_results) == 1
            and len(gold_results[0]) == 1
            and len(agent_results) == 1
            and len(agent_results[0]) == 1
        ):
            try:
                gold_val = float(str(gold_results[0][0]))
                agent_val = float(str(agent_results[0][0]))
                denom = max(abs(gold_val), 1.0)
                closeness = max(0.0, 1.0 - abs(gold_val - agent_val) / denom)
                numeric_reward = 0.40 + 0.50 * closeness
                return max(col_score, numeric_reward), f"Numeric closeness: {closeness:.2f}"
            except (ValueError, TypeError):
                pass
        return max(col_score, 0.20), "No value overlap"

    precision = len(intersection) / len(agent_set)
    recall = len(intersection) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if f1 >= 0.999:
        return 1.0, "Perfect match!"

    value_reward = 0.40 + (f1 * 0.50)
    return max(col_score, value_reward), f"Soft-F1: {f1:.2f} (P={precision:.2f}, R={recall:.2f})"
