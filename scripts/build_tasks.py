#!/usr/bin/env python3
"""Build tasks.json and schema_linking.json from InsightXpert perfect_linking data.

Selects 150 tasks (50 simple, 50 moderate, 50 challenging) spread across 5 DBs.
Excludes european_football_2 (570MB, too heavy for HF Spaces deployment).

Usage:
    python3 scripts/build_tasks.py

Reads:
    ../InsightXpert/perfect_linking/perfect_linking_bird_dev.json

Writes:
    data/tasks.json
    data/schema_linking.json
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

# --- Configuration ---
LINKING_FILE = Path(__file__).resolve().parent.parent.parent / "InsightXpert" / "perfect_linking" / "perfect_linking_bird_dev.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

INCLUDE_DBS = {"formula_1", "toxicology", "financial", "california_schools", "debit_card_specializing"}
TARGET_PER_DIFFICULTY = 50
DIFFICULTIES = ["simple", "moderate", "challenging"]


def load_entries() -> list[dict]:
    """Load and filter entries from perfect_linking_bird_dev.json."""
    with open(LINKING_FILE) as f:
        raw = json.load(f)
    entries = []
    for val in raw.values():
        if val["db_id"] in INCLUDE_DBS:
            entries.append(val)
    return entries


def select_tasks(entries: list[dict]) -> list[dict]:
    """Select 50 tasks per difficulty, spread proportionally across DBs."""
    # Group by (difficulty, db_id)
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        groups[e["difficulty"]][e["db_id"]].append(e)

    selected = []
    for diff in DIFFICULTIES:
        db_pools = groups[diff]
        # Sort each pool by question_id for determinism
        for db_id in db_pools:
            db_pools[db_id].sort(key=lambda x: x["question_id"])

        total_available = sum(len(pool) for pool in db_pools.values())
        target = TARGET_PER_DIFFICULTY

        # Allocate proportionally, ensuring at least 1 per DB if available
        db_ids = sorted(db_pools.keys())
        allocation: dict[str, int] = {}
        remaining = target

        for i, db_id in enumerate(db_ids):
            pool_size = len(db_pools[db_id])
            if i == len(db_ids) - 1:
                # Last DB gets the remainder
                alloc = min(remaining, pool_size)
            else:
                # Proportional allocation
                alloc = min(
                    math.ceil(target * pool_size / total_available),
                    pool_size,
                    remaining,
                )
            allocation[db_id] = max(1, alloc) if pool_size > 0 else 0
            remaining -= allocation[db_id]

        # If we over-allocated (rounding), trim from largest pools
        total_alloc = sum(allocation.values())
        while total_alloc > target:
            # Find DB with most excess
            for db_id in sorted(db_ids, key=lambda d: allocation[d], reverse=True):
                if allocation[db_id] > 1:
                    allocation[db_id] -= 1
                    total_alloc -= 1
                    break

        # If we under-allocated, add to DBs with room
        while total_alloc < target:
            for db_id in sorted(db_ids, key=lambda d: len(db_pools[d]) - allocation[d], reverse=True):
                if allocation[db_id] < len(db_pools[db_id]):
                    allocation[db_id] += 1
                    total_alloc += 1
                    break
            else:
                break  # No more room

        # Select entries
        for db_id in db_ids:
            count = allocation[db_id]
            selected.extend(db_pools[db_id][:count])

    return selected


def build_outputs(selected: list[dict]) -> tuple[dict, dict]:
    """Build tasks.json and schema_linking.json from selected entries."""
    tasks = {}
    schema_linking = {}

    # Group by difficulty to assign sequential IDs
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for entry in selected:
        by_diff[entry["difficulty"]].append(entry)

    for diff in DIFFICULTIES:
        entries = sorted(by_diff[diff], key=lambda x: (x["db_id"], x["question_id"]))
        for i, entry in enumerate(entries, 1):
            task_id = f"{diff}_{i}"

            tasks[task_id] = {
                "question_id": entry["question_id"],
                "task_id": task_id,
                "db_id": entry["db_id"],
                "question": entry["question"],
                "evidence": entry.get("evidence", ""),
                "SQL": entry["gold_sql"],
                "difficulty": entry["difficulty"],
            }

            schema_linking[task_id] = {
                "question_id": entry["question_id"],
                "question": entry["question"],
                "db_id": entry["db_id"],
                "difficulty": entry["difficulty"],
                "schema_text": entry.get("schema_text", ""),
            }

    return tasks, schema_linking


def main():
    print(f"Loading entries from {LINKING_FILE}")
    entries = load_entries()
    print(f"  Total entries (5 DBs): {len(entries)}")

    # Print pool stats
    from collections import Counter
    diff_counts = Counter(e["difficulty"] for e in entries)
    db_counts = Counter(e["db_id"] for e in entries)
    print(f"  By difficulty: {dict(diff_counts)}")
    print(f"  By database: {dict(db_counts)}")

    print(f"\nSelecting {TARGET_PER_DIFFICULTY * len(DIFFICULTIES)} tasks ({TARGET_PER_DIFFICULTY} per difficulty)...")
    selected = select_tasks(entries)

    # Print selection stats
    sel_diff = Counter(e["difficulty"] for e in selected)
    sel_db = Counter(e["db_id"] for e in selected)
    print(f"  Selected by difficulty: {dict(sel_diff)}")
    print(f"  Selected by database: {dict(sel_db)}")

    tasks, schema_linking = build_outputs(selected)

    tasks_file = OUTPUT_DIR / "tasks.json"
    linking_file = OUTPUT_DIR / "schema_linking.json"

    with open(tasks_file, "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(tasks)} tasks to {tasks_file}")

    with open(linking_file, "w") as f:
        json.dump(schema_linking, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(schema_linking)} entries to {linking_file}")

    # Verify
    assert len(tasks) == TARGET_PER_DIFFICULTY * len(DIFFICULTIES), f"Expected {TARGET_PER_DIFFICULTY * len(DIFFICULTIES)} tasks, got {len(tasks)}"
    assert len(schema_linking) == len(tasks), "Task count mismatch"
    for tid in tasks:
        assert tid in schema_linking, f"Missing schema linking for {tid}"
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
