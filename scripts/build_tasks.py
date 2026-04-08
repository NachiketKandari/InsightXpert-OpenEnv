#!/usr/bin/env python3
"""Build tasks.json and schema_linking.json from InsightXpert perfect_linking data.

Merges two data sources:
  - perfect_linking_bird_dev.json (707 entries, 6 DBs)
  - perfect_linking_mini_dev_evidence.json (498 entries, 11 DBs, enriched schema)

For overlapping questions, the evidence-enriched schema_text is preferred.
Selects 200 tasks (85 simple, 65 moderate, 50 challenging) across 5 DBs.
Excludes european_football_2 (570MB, too heavy for HF Spaces deployment).

Usage:
    python3 scripts/build_tasks.py

Writes:
    data/tasks.json
    data/schema_linking.json
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# --- Configuration ---
LINKING_DIR = Path(__file__).resolve().parent.parent.parent / "InsightXpert" / "perfect_linking"
BIRD_DEV_FILE = LINKING_DIR / "perfect_linking_bird_dev.json"
EVIDENCE_FILE = LINKING_DIR / "perfect_linking_mini_dev_evidence.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

INCLUDE_DBS = {"formula_1", "toxicology", "financial", "california_schools", "debit_card_specializing"}
TARGETS = {"simple": 85, "moderate": 65, "challenging": 50}
DIFFICULTIES = ["simple", "moderate", "challenging"]


def load_and_merge() -> list[dict]:
    """Load entries from both sources, merge by question_id.

    For questions present in both files, the evidence-enriched schema_text
    is used while gold_sql comes from bird_dev (authoritative source).
    """
    # Load bird_dev (primary source for gold SQL and questions)
    with open(BIRD_DEV_FILE) as f:
        bird_raw = json.load(f)

    # Load mini_dev_evidence (enriched schema_text)
    with open(EVIDENCE_FILE) as f:
        evidence_raw = json.load(f)

    # Index evidence entries by question_id for fast lookup
    evidence_by_qid: dict[int, dict] = {}
    for val in evidence_raw.values():
        evidence_by_qid[val["question_id"]] = val

    # Merge: start with bird_dev, enrich schema_text from evidence where available
    merged: dict[int, dict] = {}
    enriched_count = 0

    for val in bird_raw.values():
        if val["db_id"] not in INCLUDE_DBS:
            continue
        qid = val["question_id"]
        entry = dict(val)
        # Prefer evidence-enriched schema if available for this question
        if qid in evidence_by_qid:
            entry["schema_text"] = evidence_by_qid[qid]["schema_text"]
            enriched_count += 1
        merged[qid] = entry

    # Add questions from evidence file that are NOT in bird_dev (new questions)
    new_from_evidence = 0
    for val in evidence_raw.values():
        qid = val["question_id"]
        if qid not in merged and val["db_id"] in INCLUDE_DBS:
            merged[qid] = dict(val)
            new_from_evidence += 1

    print(f"  Bird-dev entries (5 DBs): {len(merged) - new_from_evidence}")
    print(f"  Enriched with evidence schema: {enriched_count}")
    print(f"  New from evidence file: {new_from_evidence}")

    return list(merged.values())


def select_tasks(entries: list[dict]) -> list[dict]:
    """Select tasks per difficulty, spread proportionally across DBs."""
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
        target = TARGETS[diff]

        if total_available < target:
            print(f"  WARNING: Only {total_available} available for {diff} (need {target}), using all")
            target = total_available

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
    print("Loading and merging entries from bird_dev + mini_dev_evidence...")
    entries = load_and_merge()
    print(f"  Total merged entries (5 DBs): {len(entries)}")

    # Print pool stats
    diff_counts = Counter(e["difficulty"] for e in entries)
    db_counts = Counter(e["db_id"] for e in entries)
    print(f"  By difficulty: {dict(diff_counts)}")
    print(f"  By database: {dict(db_counts)}")

    total_target = sum(TARGETS.values())
    target_str = ", ".join(f"{TARGETS[d]} {d}" for d in DIFFICULTIES)
    print(f"\nSelecting {total_target} tasks ({target_str})...")
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
    expected = sum(min(TARGETS[d], sel_diff.get(d, 0)) for d in DIFFICULTIES)
    assert len(tasks) == expected, f"Expected {expected} tasks, got {len(tasks)}"
    assert len(schema_linking) == len(tasks), "Task count mismatch"
    for tid in tasks:
        assert tid in schema_linking, f"Missing schema linking for {tid}"
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
