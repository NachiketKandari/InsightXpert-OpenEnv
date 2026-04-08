"""Reward computation for BIRD Text-to-SQL environment.

Uses BIRD-standard evaluation metrics:
  - Strict EX (execution accuracy): exact result-set match
  - Relaxed EX: tolerates extra columns in predicted results
  - BIRD Soft-F1: row-element level precision/recall

Scoring:
  Error cases:            0.00 - 0.15
  Soft-F1 partial match:  0.20 - 0.90
  Relaxed EX match:       0.95
  Strict EX match:        1.00
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# BIRD EX metrics (from InsightXpert executor.py)
# ---------------------------------------------------------------------------

def _execution_match(
    gold_rows: list[tuple],
    agent_rows: list[tuple],
) -> bool:
    """Strict EX: unordered set equality of result tuples.

    Matches BIRD official evaluate_ex.py and InsightXpert executor.py.
    No normalization — raw tuple comparison.
    """
    return set(tuple(r) for r in gold_rows) == set(tuple(r) for r in agent_rows)


def _execution_match_ordered(
    gold_rows: list[tuple],
    agent_rows: list[tuple],
) -> bool:
    """Order-sensitive EX: list equality of result tuples."""
    return [tuple(r) for r in gold_rows] == [tuple(r) for r in agent_rows]


def _execution_match_relaxed(
    gold_rows: list[tuple],
    agent_rows: list[tuple],
) -> bool:
    """Relaxed EX: tolerates extra columns in predicted results.

    From InsightXpert executor.py lines 156-193.
    Returns True if the gold result columns appear as a contiguous
    subsequence within the predicted result columns.
    """
    if not gold_rows:
        return not agent_rows

    gold_ncols = len(gold_rows[0])
    agent_ncols = len(agent_rows[0]) if agent_rows else 0

    if agent_ncols < gold_ncols:
        return False

    if agent_ncols == gold_ncols:
        return _execution_match(gold_rows, agent_rows)

    gold_set = set(tuple(r) for r in gold_rows)
    # Slide a window of gold_ncols across predicted columns
    for start in range(agent_ncols - gold_ncols + 1):
        projected = set(tuple(r[start : start + gold_ncols]) for r in agent_rows)
        if projected == gold_set:
            return True
    return False


# ---------------------------------------------------------------------------
# BIRD Soft-F1 (from official evaluation_f1.py)
# ---------------------------------------------------------------------------

def _calculate_row_match(
    predicted_row: tuple, ground_truth_row: tuple
) -> tuple[float, float, float]:
    """Calculate element-level matching for a single row.

    From BIRD official evaluation_f1.py lines 14-40.
    Returns (match_pct, pred_only_pct, truth_only_pct).
    """
    total_columns = len(ground_truth_row)
    if total_columns == 0:
        return (1.0, 0.0, 0.0) if not predicted_row else (0.0, 1.0, 0.0)

    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0

    for pred_val in predicted_row:
        if pred_val in ground_truth_row:
            matches += 1
        else:
            element_in_pred_only += 1

    for truth_val in ground_truth_row:
        if truth_val not in predicted_row:
            element_in_truth_only += 1

    match_pct = matches / total_columns
    pred_only_pct = element_in_pred_only / total_columns
    truth_only_pct = element_in_truth_only / total_columns
    return match_pct, pred_only_pct, truth_only_pct


def _bird_soft_f1(
    gold_rows: list[tuple],
    agent_rows: list[tuple],
) -> float:
    """BIRD official Soft-F1: row-element level precision/recall.

    From BIRD official evaluation_f1.py lines 43-101.
    """
    if not gold_rows and not agent_rows:
        return 1.0

    # Deduplicate preserving order
    gold_deduped = list(dict.fromkeys(gold_rows))
    agent_deduped = list(dict.fromkeys(agent_rows))

    match_scores: list[float] = []
    pred_only_scores: list[float] = []
    truth_only_scores: list[float] = []

    for i, gt_row in enumerate(gold_deduped):
        if i >= len(agent_deduped):
            match_scores.append(0.0)
            truth_only_scores.append(1.0)
            continue
        pred_row = agent_deduped[i]
        match_score, pred_only_score, truth_only_score = _calculate_row_match(
            pred_row, gt_row
        )
        match_scores.append(match_score)
        pred_only_scores.append(pred_only_score)
        truth_only_scores.append(truth_only_score)

    # Extra rows in predicted results
    for _ in range(len(agent_deduped) - len(gold_deduped)):
        match_scores.append(0.0)
        pred_only_scores.append(1.0)
        truth_only_scores.append(0.0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_reward(
    gold_results: list[tuple] | None,
    agent_results: list[tuple] | None,
    agent_error: str | None,
    gold_columns: list[str],
    agent_columns: list[str],
    gold_sql: str = "",
) -> tuple[float, str]:
    """Compute reward and feedback for an agent's SQL attempt.

    Scoring:
      1. Strict EX (1.0) — exact result-set match
      2. Relaxed EX (0.95) — tolerates extra columns
      3. BIRD Soft-F1 (0.20-0.90) — row-element partial credit
      4. Error cases (0.00-0.15) — syntax/runtime/empty results

    Returns:
        (reward, feedback) where reward is 0.0-1.0.
    """
    # ---- Error cases: 0.00 - 0.15 ----
    if agent_error is not None:
        error_lower = agent_error.lower()
        if "syntax" in error_lower or "parse" in error_lower:
            return 0.05, f"Syntax error: {agent_error}"
        return 0.10, f"Runtime error: {agent_error}"

    if agent_results is None:
        return 0.00, "No valid SQL query provided"

    if not agent_results and gold_results:
        return 0.15, (
            "Query returned 0 rows but the correct answer has rows. "
            "Check your WHERE conditions and JOINs."
        )

    if not gold_results and not agent_results:
        return 1.0, "Perfect match! (both empty)"

    if not gold_results:
        return 0.15, "Gold query returned no results but yours did"

    # ---- Strict EX: 1.0 ----
    order_sensitive = _has_order_by(gold_sql)

    if order_sensitive:
        if _execution_match_ordered(gold_results, agent_results):
            return 1.0, "Perfect match!"
    else:
        if _execution_match(gold_results, agent_results):
            return 1.0, "Perfect match!"

    # ---- Relaxed EX: 0.95 ----
    if _execution_match_relaxed(gold_results, agent_results):
        return 0.95, (
            "Correct values but extra columns returned. "
            "Simplify your SELECT clause."
        )

    # ---- BIRD Soft-F1: 0.20 - 0.90 ----
    f1 = _bird_soft_f1(gold_results, agent_results)

    if f1 >= 0.999:
        return 1.0, "Perfect match!"

    if f1 > 0.0:
        # Map F1 (0.0-1.0) to reward (0.20-0.90)
        value_reward = 0.20 + (f1 * 0.70)

        # Compute precision/recall for actionable feedback
        gold_deduped = list(dict.fromkeys(gold_results))
        agent_deduped = list(dict.fromkeys(agent_results))
        match_scores = []
        pred_only_total = []
        truth_only_total = []
        for i, gt_row in enumerate(gold_deduped):
            if i < len(agent_deduped):
                m, p, t = _calculate_row_match(agent_deduped[i], gt_row)
                match_scores.append(m)
                pred_only_total.append(p)
                truth_only_total.append(t)
        tp = sum(match_scores)
        fp = sum(pred_only_total)
        fn = sum(truth_only_total)
        prec = round(tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        rec = round(tp / (tp + fn) * 100) if (tp + fn) > 0 else 0

        return value_reward, (
            f"Partial match: {prec}% of your rows are correct, "
            f"you found {rec}% of expected rows. Soft-F1={f1:.2f}"
        )

    # F1 = 0 — SQL ran but results have no overlap with gold
    return 0.20, "Query executed but results don't overlap with the expected answer."
