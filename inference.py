"""BIRD Text-to-SQL Inference Script.

Runs an LLM agent against the InsightXpert-OpenEnv environment, emitting
validator-exact [START]/[STEP]/[END] stdout lines per OpenEnv competition spec.
The agent generates SQL from natural language questions and self-corrects
using grader feedback across up to MAX_STEPS attempts per task.

Designed to never exit with a non-zero status: any error is logged as
[DEBUG] and a best-effort [END] line is emitted for every task.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, List, Optional, Tuple

from openai import OpenAI

# ── configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3-8B"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL")
HF_SPACE_REPO = os.getenv("HF_SPACE_REPO") or "NachiketKandari/InsightXpert-OpenEnv"

BENCHMARK = "bird-text2sql"
MAX_STEPS = 5
LLM_TIMEOUT_S = 30.0
TOTAL_BUDGET_S = float(os.getenv("INFERENCE_BUDGET_S", "1020"))  # 17 min, leaves 3-min safety
FALLBACK_SQL = "SELECT 1"

# ── system prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a SQL expert for SQLite databases. Given a database schema,
    sample data, an optional external knowledge hint, and a natural language
    question, generate the correct SQL query.

    Rules:
    - Output ONLY the SQL query, no explanation or markdown fences.
    - Use SQLite dialect (e.g., SUBSTR not SUBSTRING, IIF or CASE WHEN, no ILIKE).
    - Use table and column names exactly as they appear in the schema.
    - If the question involves a percentage, use CAST(... AS REAL) to avoid integer division.
    - If the evidence provides domain knowledge, use it to interpret the question correctly.
""").strip()


# ── stdout helpers (validator-exact format) ──────────────────────────────────


def log_debug(msg: str) -> None:
    print(f"[DEBUG] {msg}", flush=True)


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    error_val = error[:200] if error else "null"
    done_val = str(done).lower()
    action_one_line = " ".join(str(action).split())
    print(
        f"[STEP] step={step} action={action_one_line} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── task loading ─────────────────────────────────────────────────────────────


def load_task_ids() -> List[str]:
    candidates = [
        Path("data/tasks.json"),
        Path(__file__).resolve().parent / "data" / "tasks.json",
    ]
    for p in candidates:
        try:
            if p.exists():
                with p.open() as f:
                    return list(json.load(f).keys())
        except Exception as exc:
            log_debug(f"Failed reading {p}: {exc}")
    log_debug("tasks.json not found locally; relying on server-side default tasks")
    return []


# ── prompt / SQL helpers ─────────────────────────────────────────────────────


def build_messages(obs: Any, prev_sql: str, prev_feedback: str) -> List[dict]:
    parts: List[str] = []
    schema = getattr(obs, "schema_linking", "") or ""
    evidence = getattr(obs, "evidence", "") or ""
    question = getattr(obs, "question", "") or ""
    if schema:
        parts.append(f"DATABASE SCHEMA:\n{schema}")
    if evidence:
        parts.append(f"EXTERNAL KNOWLEDGE:\n{evidence}")
    parts.append(f"QUESTION: {question}")
    if prev_sql and prev_feedback:
        parts.append(
            f"\nYOUR PREVIOUS ATTEMPT:\n{prev_sql}\n\n"
            f"FEEDBACK: {prev_feedback}\n\n"
            "Please fix the SQL query based on this feedback."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def extract_sql(text: str) -> str:
    sql = (text or "").strip()
    if sql.startswith("```"):
        lines = [l for l in sql.split("\n") if not l.strip().startswith("```")]
        sql = "\n".join(lines).strip()
    return sql or FALLBACK_SQL


def get_sql(client: OpenAI, messages: List[dict]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
            timeout=LLM_TIMEOUT_S,
        )
        content = completion.choices[0].message.content or ""
        return extract_sql(content)
    except Exception as exc:
        log_debug(f"Model request failed: {exc}")
        return FALLBACK_SQL


# ── environment connection ───────────────────────────────────────────────────


async def connect_env():
    """Return an openenv client connected to the BIRD environment.

    Priority:
      1. IMAGE_NAME / LOCAL_IMAGE_NAME -> from_docker_image (matches validator sample)
      2. ENV_URL                       -> connect to an already-running server
      3. HF_SPACE_REPO                 -> pull & run the HF Space image
    """
    # Import lazily so the module can be imported even if client/models are missing.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from client import BirdText2SQLEnv  # noqa: E402

    if IMAGE_NAME:
        log_debug(f"Connecting via from_docker_image(image={IMAGE_NAME!r})")
        return await BirdText2SQLEnv.from_docker_image(IMAGE_NAME)

    if ENV_URL:
        log_debug(f"Connecting via ENV_URL={ENV_URL!r}")
        env = BirdText2SQLEnv(base_url=ENV_URL)
        await env.connect()
        return env

    log_debug(f"Connecting via from_env(repo_id={HF_SPACE_REPO!r})")
    return await BirdText2SQLEnv.from_env(HF_SPACE_REPO)


def emit_skipped(task_id: str, reason: str) -> None:
    log_start(task_id)
    log_debug(f"Task {task_id} skipped: {reason}")
    log_end(success=False, steps=0, score=0.0, rewards=[])


# ── task execution ───────────────────────────────────────────────────────────


async def run_task(env, client: OpenAI, task_id: str, deadline: float) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_id)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        prev_sql = ""
        prev_feedback = ""

        for step_num in range(1, MAX_STEPS + 1):
            if result.done or time.monotonic() >= deadline:
                break

            messages = build_messages(obs, prev_sql, prev_feedback)
            sql = get_sql(client, messages)

            # Import action type lazily
            from models import BirdSQLAction  # noqa: E402
            result = await env.step(BirdSQLAction(sql_query=sql))
            obs = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step_num

            err: Optional[str] = None
            if getattr(obs, "execution_success", True) is False:
                err = getattr(obs, "feedback", None)
            log_step(step=step_num, action=sql, reward=reward, done=done, error=err)

            if done:
                break

            prev_sql = sql
            prev_feedback = getattr(obs, "feedback", "") or ""

        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.95
    except Exception as exc:
        log_debug(f"Task {task_id} error: {exc}")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    start = time.monotonic()
    deadline = start + TOTAL_BUDGET_S

    log_debug(
        f"ENV_URL={ENV_URL!r} IMAGE_NAME={IMAGE_NAME!r} "
        f"API_BASE_URL={API_BASE_URL!r} MODEL_NAME={MODEL_NAME!r} "
        f"API_KEY={'set' if API_KEY else 'MISSING'} "
        f"BUDGET_S={TOTAL_BUDGET_S}"
    )

    task_ids = load_task_ids()
    if not task_ids:
        # Ensure we always emit at least 3 (task, score) pairs so the validator
        # can find graded tasks even if tasks.json is missing on disk.
        task_ids = ["simple_1", "simple_2", "simple_3"]
        log_debug(f"Using fallback task list: {task_ids}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    env = None
    try:
        env = await connect_env()
    except Exception as exc:
        log_debug(f"Environment connection failed: {exc}")
        traceback.print_exc()
        for task_id in task_ids:
            emit_skipped(task_id, f"env connect failed: {exc}")
        return

    try:
        for task_id in task_ids:
            if time.monotonic() >= deadline:
                emit_skipped(task_id, "time budget exhausted")
                continue
            try:
                await run_task(env, client, task_id, deadline)
            except Exception as exc:
                log_debug(f"Unexpected error on {task_id}: {exc}")
                traceback.print_exc()
                try:
                    log_end(success=False, steps=0, score=0.0, rewards=[])
                except Exception:
                    pass
    finally:
        try:
            await env.close()
        except Exception as exc:
            log_debug(f"env.close() error: {exc}")
        elapsed = time.monotonic() - start
        log_debug(f"Inference finished in {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        log_debug(f"FATAL unhandled exception: {exc}")
        traceback.print_exc()
    # Always exit 0 — the validator greps [START]/[STEP]/[END] from stdout,
    # and a non-zero exit forces the "unhandled exception" failure mode.
    sys.exit(0)
