"""BIRD Text-to-SQL Inference Script.

Runs an LLM agent against the InsightXpert-OpenEnv environment, emitting
validator-exact [START]/[STEP]/[END] stdout lines per OpenEnv competition spec.
The agent generates SQL from natural language questions and self-corrects
using grader feedback across up to MAX_STEPS attempts per task.
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Any, List, Optional

from openai import OpenAI

from client import BirdText2SQLEnv
from models import BirdSQLAction

# ── configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3-8B"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL")

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
    print(
        f"[STEP] step={step} action={action} "
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


# ── main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    # ── diagnostic dump (visible in validator logs) ─────────────────────────
    print(
        f"[DEBUG] ENV_URL={ENV_URL!r} IMAGE_NAME={IMAGE_NAME!r} "
        f"API_BASE_URL={API_BASE_URL!r} MODEL_NAME={MODEL_NAME!r} "
        f"API_KEY={'set' if API_KEY else 'MISSING'}",
        flush=True,
    )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # ── connect to environment ──────────────────────────────────────────────
    try:
        if ENV_URL:
            env = BirdText2SQLEnv(base_url=ENV_URL)
            await env.connect()
        elif IMAGE_NAME:
            env = await BirdText2SQLEnv.from_docker_image(IMAGE_NAME)
        else:
            raise RuntimeError(
                "Neither ENV_URL nor IMAGE_NAME is set — cannot connect to environment"
            )
    except Exception as exc:
        print(f"[DEBUG] FATAL: environment connection failed: {exc}", flush=True)
        raise

    for task_id in TASKS:
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_id)

        try:
            result = await env.reset(task_id=task_id)
            obs = result.observation

            prev_sql = ""
            prev_feedback = ""

            for step_num in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                messages = build_prompt(obs, prev_sql, prev_feedback)

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=500,
                    )
                    sql = extract_sql(response.choices[0].message.content or "")
                except Exception as exc:
                    print(f"[DEBUG] Model request failed: {exc}", flush=True)
                    sql = "SELECT 1"

                result = await env.step(BirdSQLAction(sql_query=sql))
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done

                rewards.append(reward)
                steps_taken = step_num

                error = obs.feedback if not obs.execution_success else None
                log_step(step=step_num, action=sql, reward=reward, done=done, error=error)

                if done:
                    break

                prev_sql = sql
                prev_feedback = obs.feedback

            score = max(rewards) if rewards else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score >= 0.95

        except Exception as exc:
            print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    try:
        await env.close()
    except Exception as e:
        print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        import traceback
        print(f"[DEBUG] FATAL unhandled exception: {exc}", flush=True)
        traceback.print_exc()
        raise
