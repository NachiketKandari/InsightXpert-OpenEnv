#!/usr/bin/env python3
"""Build perfect schema linking JSON for all tasks in this OpenEnv environment.

Reads data/tasks.json, profiles each database, parses gold SQL to identify
relevant tables/columns, and writes data/schema_linking.json.

Usage:
    python -m schema_linking.build_linking
    python -m schema_linking.build_linking --tasks data/tasks.json --db-dir data/databases --output data/schema_linking.json

Requirements:
    pip install sqlglot pydantic
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema_linking.db import Database
from schema_linking.models import DatabaseProfile, DatabaseSchema
from schema_linking.perfect_linker import build_perfect_schema
from schema_linking.schema_extractor import SchemaExtractor
from schema_linking.stats_collector import StatsCollector

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_db_path(db_dir: Path, db_id: str) -> Path:
    """Locate the .sqlite file for a given db_id under db_dir.

    Supports both flat layout (db_dir/db_id.sqlite) and nested layout
    (db_dir/db_id/db_id.sqlite).
    """
    flat = db_dir / f"{db_id}.sqlite"
    if flat.exists():
        return flat
    nested = db_dir / db_id / f"{db_id}.sqlite"
    if nested.exists():
        return nested
    raise FileNotFoundError(
        f"Database '{db_id}' not found at {flat} or {nested}"
    )


def build_linking(
    tasks_path: Path,
    db_dir: Path,
    profiles_dir: Path | None = None,
) -> dict[str, dict]:
    """Build perfect schema linking for all tasks.

    If profiles_dir is provided and contains pre-computed profile.json/schema.json
    files (from InsightXpert), those are loaded instead of re-profiling. This
    preserves LLM-generated column summaries from the full profiling pipeline.

    Returns a dict mapping task_id -> {task_id, question_id, question, db_id,
    difficulty, schema_text}.
    """
    tasks = json.loads(tasks_path.read_text())

    # Cache schemas and profiles per db_id
    schema_cache: dict[str, DatabaseSchema] = {}
    profile_cache: dict[str, DatabaseProfile] = {}

    result: dict[str, dict] = {}
    failed = 0

    for task_id, task in tasks.items():
        db_id = task["db_id"]

        if db_id not in schema_cache:
            # Try loading pre-computed profiles first
            loaded = False
            if profiles_dir:
                schema_path = profiles_dir / db_id / "schema.json"
                profile_path = profiles_dir / db_id / "profile.json"
                if schema_path.exists() and profile_path.exists():
                    schema_cache[db_id] = DatabaseSchema.model_validate_json(
                        schema_path.read_text()
                    )
                    profile_cache[db_id] = DatabaseProfile.model_validate_json(
                        profile_path.read_text()
                    )
                    loaded = True
                    logger.info("Loaded pre-computed profile for '%s'", db_id)

            if not loaded:
                # Profile from scratch (stats only, no LLM summaries)
                logger.info("Profiling database '%s' from scratch...", db_id)
                db_path = find_db_path(db_dir, db_id)
                with Database(db_path) as db:
                    schema = SchemaExtractor().extract(db)
                    profile = StatsCollector().collect(db, schema)
                schema_cache[db_id] = schema
                profile_cache[db_id] = profile

        schema = schema_cache[db_id]
        profile = profile_cache[db_id]

        try:
            schema_text = build_perfect_schema(
                gold_sql=task["SQL"],
                schema=schema,
                profile=profile,
            )
            result[task_id] = {
                "task_id": task_id,
                "question_id": task["question_id"],
                "question": task["question"],
                "db_id": db_id,
                "difficulty": task["difficulty"],
                "schema_text": schema_text,
            }
        except Exception as e:
            logger.error(
                "Perfect linking failed for %s (%s): %s", task_id, db_id, e,
            )
            failed += 1

    logger.info(
        "Perfect linking complete: %d/%d succeeded, %d failed",
        len(result), len(tasks), failed,
    )
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build perfect schema linking from gold SQL queries"
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=PROJECT_ROOT / "data" / "tasks.json",
        help="Path to tasks.json (default: data/tasks.json)",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "databases",
        help="Path to databases directory (default: data/databases)",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=None,
        help="Path to pre-computed profiles (optional, from InsightXpert). "
             "Each db should have <profiles-dir>/<db_id>/schema.json and profile.json.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "schema_linking.json",
        help="Output JSON file (default: data/schema_linking.json)",
    )
    args = parser.parse_args()

    linking = build_linking(
        tasks_path=args.tasks,
        db_dir=args.db_dir,
        profiles_dir=args.profiles_dir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(linking, f, indent=2)

    print(f"Wrote {len(linking)} perfect schemas to {args.output}")


if __name__ == "__main__":
    main()
