"""BIRD Text-to-SQL Inference Script.

Runs an LLM agent against the InsightXpert-OpenEnv environment, emitting
validator-exact [START]/[STEP]/[END] stdout lines per OpenEnv competition spec.
The agent generates SQL from natural language questions and self-corrects
using grader feedback across up to MAX_STEPS attempts per task.
"""

from __future__ import annotations

import json
import os
import textwrap
import traceback
from typing import Any

from openai import OpenAI

from client import BirdText2SQLEnv
from models import BirdSQLAction

# ── configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "bird-text2sql"
MAX_STEPS = 5

with open("data/tasks.json") as _f:
    TASKS: list[str] = list(json.load(_f).keys())

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


def emit_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def emit_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None = None,
) -> None:
    done_s = str(done).lower()
    err_s = error[:200] if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_s} error={err_s}",
        flush=True,
    )


def emit_end(success: bool, steps: int, rewards: list[float]) -> None:
    success_s = str(success).lower()
    rewards_s = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_s} steps={steps} rewards={rewards_s}",
        flush=True,
    )


# ── prompt building ──────────────────────────────────────────────────────────


def build_prompt(
    obs: Any,
    prev_sql: str = "",
    prev_feedback: str = "",
) -> list[dict[str, str]]:
    """Build chat messages from the current observation."""
    user_parts: list[str] = []

    if obs.schema_linking:
        user_parts.append(f"DATABASE SCHEMA:\n{obs.schema_linking}")
    if obs.sample_rows:
        user_parts.append(f"SAMPLE DATA:\n{obs.sample_rows}")
    if obs.evidence:
        user_parts.append(f"EXTERNAL KNOWLEDGE:\n{obs.evidence}")

    user_parts.append(f"QUESTION: {obs.question}")

    if prev_sql and prev_feedback:
        user_parts.append(
            f"\nYOUR PREVIOUS ATTEMPT:\n{prev_sql}\n\n"
            f"FEEDBACK: {prev_feedback}\n\n"
            "Please fix the SQL query based on this feedback."
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# ── SQL extraction ───────────────────────────────────────────────────────────


def extract_sql(text: str) -> str:
    """Extract SQL from LLM response, stripping markdown fences if present."""
    sql = text.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        sql = "\n".join(lines).strip()
    return sql


# ── episode runner ───────────────────────────────────────────────────────────


def run_task(env: Any, client: OpenAI, task_id: str) -> None:
    """Run a single task episode with self-correction loop."""
    result = env.reset(task_id=task_id)
    obs = result.observation
    emit_start(task_id)

    rewards: list[float] = []
    prev_sql = ""
    prev_feedback = ""

    for step_num in range(1, MAX_STEPS + 1):
        messages = build_prompt(obs, prev_sql, prev_feedback)

        error: str | None = None
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            sql = extract_sql(response.choices[0].message.content or "")
        except Exception as exc:
            error = str(exc)[:200]
            emit_step(step_num, "ERROR", 0.00, True, error)
            break

        result = env.step(BirdSQLAction(sql_query=sql))
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done
        rewards.append(reward)

        error = obs.feedback if not obs.execution_success else None
        emit_step(step_num, sql, reward, done, error)

        if done:
            break

        # Save for self-correction on next step
        prev_sql = sql
        prev_feedback = obs.feedback

    success = any(r >= 1.0 for r in rewards)
    emit_end(success, len(rewards), rewards)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    with BirdText2SQLEnv(base_url=ENV_URL).sync() as env:
        for task_id in TASKS:
            try:
                run_task(env, client, task_id)
            except Exception:
                tb_lines = traceback.format_exc().splitlines()
                error_msg = (tb_lines[-1] if tb_lines else "Unknown error")[:200]
                emit_start(task_id)
                emit_step(1, "ERROR", 0.00, True, error_msg)
                emit_end(False, 0, [0.00])


if __name__ == "__main__":
    main()
