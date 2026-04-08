"""BIRD Text-to-SQL OpenEnv Environment -- core logic."""

from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from openenv.core.env_server import Environment

try:
    from ..models import BirdSQLAction, BirdSQLObservation, BirdSQLState
except ImportError:
    from models import BirdSQLAction, BirdSQLObservation, BirdSQLState

logger = logging.getLogger(__name__)

# Paths relative to package root
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _PACKAGE_DIR / "data"
_TASKS_FILE = _DATA_DIR / "tasks.json"
_SCHEMA_LINKING_FILE = _DATA_DIR / "schema_linking.json"
_DB_DIR = _DATA_DIR / "databases"

_MAX_STEPS = 5
_SQL_TIMEOUT_SECONDS = 5
_SAMPLE_ROWS_LIMIT = 3

# Only allow SELECT queries
_FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


class BirdEnvironment(Environment[BirdSQLAction, BirdSQLObservation, BirdSQLState]):
    """OpenEnv environment for BIRD Text-to-SQL benchmark tasks."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(_TASKS_FILE) as f:
            self.task_registry: dict[str, dict] = json.load(f)
        with open(_SCHEMA_LINKING_FILE) as f:
            raw_linking: dict = json.load(f)
        # Extract schema_text string from each entry (supports both old
        # string-value format and new dict-value format with schema_text key)
        self._schema_linking: dict[str, str] = {}
        for tid, val in raw_linking.items():
            if isinstance(val, dict):
                self._schema_linking[tid] = val.get("schema_text", "")
            else:
                self._schema_linking[tid] = str(val)

        self._db: sqlite3.Connection | None = None
        self._current_task: dict | None = None
        self._step_count: int = 0
        self._episode_id: str = ""
        self._accumulated_reward: float = 0.0
        self._last_action: str = ""
        self._done: bool = False

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> BirdSQLObservation:
        task_id = kwargs.get("task_id")

        if seed is not None:
            random.seed(seed)
        if task_id is None:
            task_id = random.choice(list(self.task_registry.keys()))
        if task_id not in self.task_registry:
            raise ValueError(
                f"Unknown task_id: {task_id!r}. Available: {list(self.task_registry.keys())}"
            )

        task = self.task_registry[task_id]

        # Close previous DB if any
        if self._db is not None:
            self._db.close()

        # Load fresh in-memory copy of the database
        # check_same_thread=False is needed because the async server
        # runs reset/step in a thread pool but close() from the main thread.
        db_path = _DB_DIR / task["db_id"] / f"{task['db_id']}.sqlite"
        source = sqlite3.connect(str(db_path))
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        source.backup(self._db)
        source.close()

        # Extract sample rows
        sample_rows = self._get_sample_rows(_SAMPLE_ROWS_LIMIT)

        # Reset episode state
        self._current_task = task
        self._step_count = 0
        self._episode_id = episode_id or str(uuid4())
        self._accumulated_reward = 0.0
        self._last_action = ""
        self._done = False

        logger.info(
            "Reset episode %s -- task=%s db=%s",
            self._episode_id,
            task_id,
            task["db_id"],
        )

        return BirdSQLObservation(
            task_id=task_id,
            db_id=task["db_id"],
            difficulty=task["difficulty"],
            question=task["question"],
            evidence=task.get("evidence", ""),
            schema_linking=self._schema_linking.get(task_id, ""),
            sample_rows=sample_rows,
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: BirdSQLAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> BirdSQLObservation:
        if self._current_task is None:
            raise RuntimeError("Must call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_count += 1
        sql = action.sql_query.strip()
        self._last_action = sql

        task = self._current_task
        task_id = task["task_id"]

        # Safety: reject non-SELECT queries
        agent_results: list[tuple] | None = None
        agent_error: str | None = None
        agent_cols: list[str] = []

        if _FORBIDDEN_PATTERN.search(sql):
            reward = 0.00
            feedback = "Only SELECT queries are allowed"
            agent_error = "Forbidden SQL statement"
        elif not sql.upper().startswith(("SELECT", "WITH")):
            reward = 0.00
            feedback = "Query must start with SELECT or WITH (CTE)"
            agent_error = "Not a SELECT query"
        else:
            # Execute agent SQL
            agent_results, agent_error, agent_cols = self._safe_execute(sql)

            # Execute gold SQL
            gold_results, gold_error, gold_cols = self._safe_execute(task["SQL"])

            if gold_error:
                logger.error("Gold SQL failed for %s: %s", task_id, gold_error)
                reward = 0.0
                feedback = f"Internal error: gold SQL failed ({gold_error})"
            else:
                try:
                    from .grader import compute_reward
                except ImportError:
                    from server.grader import compute_reward

                reward, feedback = compute_reward(
                    gold_results,
                    agent_results,
                    agent_error,
                    gold_cols,
                    agent_cols,
                    task["SQL"],
                )

        self._accumulated_reward += reward
        done = (reward >= 0.9999) or (self._step_count >= _MAX_STEPS)
        self._done = done

        # Build execution result string for the observation
        if agent_error is not None:
            exec_result = agent_error
            exec_success = False
            row_count = 0
            col_names: list[str] = []
        else:
            display_rows = agent_results[:10] if agent_results else []
            exec_result = "\n".join(str(row) for row in display_rows)
            if agent_results and len(agent_results) > 10:
                exec_result += f"\n... ({len(agent_results)} total rows)"
            exec_success = True
            row_count = len(agent_results) if agent_results else 0
            col_names = agent_cols or []

        logger.info(
            "Step %d/%d -- task=%s reward=%.2f done=%s feedback=%s",
            self._step_count,
            _MAX_STEPS,
            task_id,
            reward,
            done,
            feedback,
        )

        return BirdSQLObservation(
            task_id=task_id,
            db_id=task["db_id"],
            difficulty=task["difficulty"],
            question=task["question"],
            evidence=task.get("evidence", ""),
            schema_linking=self._schema_linking.get(task_id, ""),
            sample_rows="",  # Don't resend sample rows on step
            execution_result=exec_result,
            execution_success=exec_success,
            row_count=row_count,
            column_names=col_names,
            reward=reward,
            done=done,
            feedback=feedback,
        )

    @property
    def state(self) -> BirdSQLState:
        return BirdSQLState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._current_task["task_id"] if self._current_task else "",
            db_id=self._current_task["db_id"] if self._current_task else "",
            difficulty=self._current_task["difficulty"] if self._current_task else "",
            max_steps=_MAX_STEPS,
            accumulated_reward=self._accumulated_reward,
            last_action=self._last_action,
            done=self._done,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_execute(
        self, sql: str
    ) -> tuple[list[tuple] | None, str | None, list[str]]:
        """Execute SQL safely with timeout. Returns (rows, error, column_names)."""
        if self._db is None:
            return None, "No database loaded", []
        try:
            # Enforce real execution timeout via progress handler
            deadline = time.monotonic() + _SQL_TIMEOUT_SECONDS

            def _check_timeout():
                if time.monotonic() > deadline:
                    return 1  # non-zero cancels the query
                return 0

            self._db.set_progress_handler(_check_timeout, 1000)
            cursor = self._db.execute(sql)
            rows = cursor.fetchall()
            self._db.set_progress_handler(None, 0)
            cols = (
                [desc[0] for desc in cursor.description]
                if cursor.description
                else []
            )
            return rows, None, cols
        except sqlite3.OperationalError as e:
            self._db.set_progress_handler(None, 0)
            if "interrupt" in str(e).lower():
                return None, f"Query timed out after {_SQL_TIMEOUT_SECONDS}s", []
            return None, str(e), []
        except sqlite3.Error as e:
            self._db.set_progress_handler(None, 0)
            return None, str(e), []

    def _get_sample_rows(self, limit: int = 3) -> str:
        """Get sample rows from each table."""
        if self._db is None:
            return ""
        cursor = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        parts: list[str] = []
        for table in tables:
            try:
                cur = self._db.execute(f'SELECT * FROM "{table}" LIMIT {limit}')
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                if rows:
                    header = " | ".join(cols)
                    sep = "-" * len(header)
                    row_strs = [
                        " | ".join(str(v) for v in row) for row in rows
                    ]
                    parts.append(
                        f"-- {table}\n{header}\n{sep}\n" + "\n".join(row_strs)
                    )
            except sqlite3.Error:
                continue

        return "\n\n".join(parts)

    def close(self) -> None:
        """Clean up database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None
