"""
BIRD Text-to-SQL Inference Script
Uses OpenAI-compatible client for LLM calls.
Emits [START]/[STEP]/[END] stdout logs per OpenEnv competition spec.
"""

from __future__ import annotations

import os

from openai import OpenAI

from client import BirdText2SQLEnv
from models import BirdSQLAction

# --- Mandatory env vars ---
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://router.huggingface.co/v1",
)
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "bird-text2sql"
MAX_STEPS = 5

# Load task IDs dynamically from tasks.json
import json as _json
with open("data/tasks.json") as _f:
    TASKS = list(_json.load(_f).keys())

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """\
You are a SQL expert for SQLite databases. Given a database schema, \
sample data, an optional external knowledge hint, and a natural language question, \
generate the correct SQL query.

Rules:
- Output ONLY the SQL query, no explanation or markdown fences.
- Use SQLite dialect (e.g., SUBSTR not SUBSTRING, IIF or CASE WHEN, no ILIKE).
- Use table and column names exactly as they appear in the schema.
- If the question involves a percentage, use CAST(... AS REAL) to avoid integer division.
- If the evidence provides domain knowledge, use it to interpret the question correctly."""


def build_prompt(obs, prev_sql: str = "", prev_feedback: str = "") -> list[dict]:
    """Build chat messages for the LLM."""
    user_parts = []
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


def extract_sql(text: str) -> str:
    """Extract SQL from LLM response, stripping markdown fences."""
    sql = text.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        sql = "\n".join(lines).strip()
    return sql


def run_task(env, task_id: str) -> None:
    """Run a single task episode."""
    result = env.reset(task_id=task_id)
    obs = result.observation
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    rewards: list[float] = []
    prev_sql = ""
    prev_feedback = ""

    for step_num in range(1, MAX_STEPS + 1):
        messages = build_prompt(obs, prev_sql, prev_feedback)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            sql = extract_sql(response.choices[0].message.content or "")
        except Exception as e:
            print(f"[STEP] step={step_num} action='ERROR' reward=0.00 done=false error={e}")
            continue

        result = env.step(BirdSQLAction(sql_query=sql))
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done
        rewards.append(reward)

        error_str = obs.feedback if not obs.execution_success else "null"
        print(
            f"[STEP] step={step_num} action={sql!r} "
            f"reward={reward:.2f} done={str(done).lower()} "
            f"error={error_str}"
        )

        if done:
            break

        # Save for self-correction on next step
        prev_sql = sql
        prev_feedback = obs.feedback

    success = any(r >= 0.99 for r in rewards)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={len(rewards)} rewards={rewards_str}"
    )


def main() -> None:
    env_url = os.getenv("ENV_URL", "http://localhost:7860")
    with BirdText2SQLEnv(base_url=env_url).sync() as env:
        for task_id in TASKS:
            try:
                run_task(env, task_id)
            except Exception as e:
                print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")
                print(f"[STEP] step=1 action='ERROR' reward=0.00 done=true error={e}")
                print(f"[END] success=false steps=0 rewards=0.00")


if __name__ == "__main__":
    main()
