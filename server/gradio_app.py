"""Custom Gradio UI for InsightXpert-OpenEnv.

Replaces the default OpenEnv web interface with a Text-to-SQL-specific
dashboard featuring reward visualization, difficulty badges, and step tracing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pathlib import Path

import gradio as gr
import pandas as pd


# ── Formatting helpers ──────────────────────────────────────────────────────

DIFF_BADGE = {
    "simple": "🟢 SIMPLE",
    "moderate": "🟡 MODERATE",
    "challenging": "🔴 CHALLENGING",
}


def _format_task_info(obs: Dict[str, Any]) -> str:
    """Format task observation as Markdown with difficulty badge."""
    diff = obs.get("difficulty", "")
    badge = DIFF_BADGE.get(diff, diff.upper())
    task_id = obs.get("task_id", "")
    db_id = obs.get("db_id", "")
    question = obs.get("question", "")
    evidence = obs.get("evidence", "")
    schema = obs.get("schema_linking", "")

    lines = [
        f"### {badge} &nbsp; `{task_id}`",
        f"**Database:** `{db_id}`",
        "",
        f"**Question:** {question}",
    ]
    if evidence:
        lines += ["", f"**Evidence:** {evidence}"]
    if schema:
        lines += [
            "",
            "<details><summary><b>Relevant Schema</b> (click to expand)</summary>",
            "",
            f"```sql\n{schema}\n```",
            "</details>",
        ]
    sample = obs.get("sample_rows", "")
    if sample:
        lines += [
            "",
            "<details><summary><b>Sample Rows</b> (click to expand)</summary>",
            "",
            f"```\n{sample}\n```",
            "</details>",
        ]
    return "\n".join(lines)


def _format_step_result(obs: Dict[str, Any], reward: float) -> str:
    """Format step result with reward bar and feedback."""
    filled = int(reward * 20)
    bar = "█" * filled + "░" * (20 - filled)
    if reward >= 0.95:
        color = "🟢"
    elif reward >= 0.5:
        color = "🟡"
    else:
        color = "🔴"

    done = obs.get("done", False)
    feedback = obs.get("feedback", "")
    exec_result = obs.get("execution_result", "")
    row_count = obs.get("row_count", 0)

    lines = [
        f"### {color} Reward: **{reward:.2f}** &nbsp; `{bar}`",
        "",
        f"**Feedback:** {feedback}",
    ]
    if exec_result:
        preview = exec_result[:500]
        if len(exec_result) > 500:
            preview += "\n..."
        lines += [
            "",
            f"**Result** ({row_count} rows):",
            f"```\n{preview}\n```",
        ]
    if done:
        lines.append("\n**Episode complete.**")
    return "\n".join(lines)


def _make_chart_df(records: List[Dict]) -> pd.DataFrame:
    """Build DataFrame for gr.LinePlot."""
    if not records:
        return pd.DataFrame({"Step": pd.Series(dtype="int"), "Reward": pd.Series(dtype="float")})
    return pd.DataFrame({
        "Step": [r["step"] for r in records],
        "Reward": [r["reward"] for r in records],
    })


def _make_trace_df(records: List[Dict]) -> pd.DataFrame:
    """Build DataFrame for episode trace table."""
    if not records:
        return pd.DataFrame({
            "Step": pd.Series(dtype="int"),
            "Reward": pd.Series(dtype="float"),
            "SQL": pd.Series(dtype="str"),
            "Feedback": pd.Series(dtype="str"),
        })
    return pd.DataFrame([{
        "Step": r["step"],
        "Reward": r["reward"],
        "SQL": r["sql"],
        "Feedback": r["feedback"],
    } for r in records])


# ── Reward system documentation ─────────────────────────────────────────────

REWARD_SYSTEM_MD = """
## Reward Function

Uses BIRD-standard evaluation metrics (Execution Accuracy + Soft-F1):

| Condition | Reward | Feedback |
|-----------|--------|----------|
| **Strict EX** — exact result-set match | **1.00** | Perfect match! |
| **Relaxed EX** — correct values, extra columns | **0.95** | Simplify SELECT clause |
| **Soft-F1 partial** — row-element partial credit | **0.20–0.90** | Precision/recall breakdown |
| Zero rows returned | 0.15 | Check WHERE conditions and JOINs |
| Runtime error (bad table/column) | 0.10 | Error details |
| SQL syntax error | 0.05 | Syntax error details |
| No valid SQL / forbidden statement | 0.00 | Rejected |

Order-sensitive comparison is used when the gold SQL contains `ORDER BY`.

---

## Task Distribution

| Difficulty | Count | SQL Patterns |
|------------|-------|-------------|
| 🟢 Simple | 50 | Basic SELECT, JOIN, WHERE, COUNT, ORDER BY |
| 🟡 Moderate | 50 | Multi-table JOIN, AVG, CASE WHEN, percentage calculations |
| 🔴 Challenging | 50 | Nested subqueries, CTE, IIF, STRFTIME, complex aggregation |

**Total: 150 tasks** across 5 SQLite databases from the BIRD Mini-Dev benchmark.

---

## Databases

| Database | Domain |
|----------|--------|
| `california_schools` | California school demographics, test scores, funding |
| `debit_card_specializing` | Czech financial transactions, card usage |
| `financial` | Czech banking — accounts, loans, transactions |
| `formula_1` | Formula 1 racing — drivers, constructors, results |
| `toxicology` | Molecular toxicology — atoms, bonds, molecules |

---

## Safety Constraints

- Only `SELECT` and `WITH` (CTE) queries are allowed
- Each episode gets a fresh in-memory copy of the database
- SQL execution timeout: 5 seconds
- Maximum 5 steps per episode, early termination on perfect score

---

## RL Training Compatibility

- `SUPPORTS_CONCURRENT_SESSIONS = True` (64 max)
- WebSocket-based client for TRL/GRPO integration
- Each session gets its own isolated in-memory DB copy
- Schema linking provided per task to ensure achievable rewards
"""


# ── Main builder ─────────────────────────────────────────────────────────────

def build_custom_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str = "OpenEnv Environment",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """Build a custom Gradio UI for InsightXpert-OpenEnv."""

    readme_content = ""
    if metadata and hasattr(metadata, "readme_content") and metadata.readme_content:
        readme_content = metadata.readme_content
    if not readme_content:
        readme_path = Path(__file__).resolve().parent.parent / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text()

    # ── Event handlers ──────────────────────────────────────────────────

    async def reset_env(_records):
        try:
            data = await web_manager.reset_environment()
        except Exception as e:
            return (
                f"**Error:** {e}",
                [],
                _make_chart_df([]),
                _make_trace_df([]),
                "",
                f'{{"error": "{e}"}}',
            )
        obs = data.get("observation", {})
        task_md = _format_task_info(obs)
        return (
            task_md,
            [],
            _make_chart_df([]),
            _make_trace_df([]),
            "*Enter a SQL query and click Execute SQL.*",
            json.dumps(data, indent=2),
        )

    async def step_form(sql_query, records):
        if not sql_query or not sql_query.strip():
            return (
                gr.update(),
                records,
                _make_chart_df(records),
                _make_trace_df(records),
                "**Please enter a SQL query.**",
                "",
            )
        try:
            data = await web_manager.step_environment({"sql_query": sql_query.strip()})
        except Exception as e:
            return (
                gr.update(),
                records,
                _make_chart_df(records),
                _make_trace_df(records),
                f"**Error:** {e}",
                f'{{"error": "{e}"}}',
            )
        obs = data.get("observation", {})
        reward = data.get("reward", 0.0)
        feedback = obs.get("feedback", "")
        step_num = len(records) + 1
        sql_preview = sql_query.strip()[:60]
        if len(sql_query.strip()) > 60:
            sql_preview += "..."
        new_records = records + [{
            "step": step_num,
            "reward": round(reward, 4),
            "sql": sql_preview,
            "feedback": feedback,
        }]
        result_md = _format_step_result(obs, reward)
        return (
            _format_task_info(obs),
            new_records,
            _make_chart_df(new_records),
            _make_trace_df(new_records),
            result_md,
            json.dumps(data, indent=2),
        )

    # ── Build UI ────────────────────────────────────────────────────────

    with gr.Blocks(title="InsightXpert-OpenEnv") as demo:
        step_records = gr.State([])

        gr.Markdown(
            "# InsightXpert-OpenEnv\n"
            "**BIRD Text-to-SQL** &mdash; "
            "150 tasks &middot; 5 databases &middot; "
            "3 difficulty levels &middot; RL environment"
        )

        with gr.Tabs():
            # ── Tab 1: Interactive Demo ──────────────────────────────
            with gr.Tab("Interactive Demo"):
                with gr.Row():
                    with gr.Column(scale=3):
                        task_display = gr.Markdown(
                            "*Click **Reset** to load a random task.*"
                        )
                    with gr.Column(scale=2):
                        reward_chart = gr.LinePlot(
                            value=_make_chart_df([]),
                            x="Step",
                            y="Reward",
                            title="Reward per Step",
                            y_lim=[0.0, 1.05],
                            height=220,
                        )

                sql_input = gr.Code(
                    language="sql",
                    label="SQL Query",
                    lines=4,
                    value="SELECT ",
                )

                with gr.Row():
                    step_btn = gr.Button(
                        "Execute SQL", variant="primary", scale=2
                    )
                    reset_btn = gr.Button(
                        "Reset / New Task", variant="secondary", scale=1
                    )

                result_display = gr.Markdown("*No result yet.*")

                trace_table = gr.DataFrame(
                    value=_make_trace_df([]),
                    label="Episode Trace",
                    wrap=True,
                    interactive=False,
                )

                with gr.Accordion("Raw JSON Response", open=False):
                    raw_json = gr.Code(
                        label="JSON",
                        language="json",
                        interactive=False,
                    )

            # ── Tab 2: Reward System ─────────────────────────────────
            with gr.Tab("Reward System"):
                gr.Markdown(REWARD_SYSTEM_MD)

            # ── Tab 3: About ─────────────────────────────────────────
            with gr.Tab("About"):
                if quick_start_md:
                    with gr.Accordion("Quick Start", open=True):
                        gr.Markdown(quick_start_md)
                gr.Markdown(readme_content or "*No README available.*")

        # ── Wire events ─────────────────────────────────────────────
        outputs = [
            task_display,
            step_records,
            reward_chart,
            trace_table,
            result_display,
            raw_json,
        ]
        reset_btn.click(fn=reset_env, inputs=[step_records], outputs=outputs)
        step_btn.click(fn=step_form, inputs=[sql_input, step_records], outputs=outputs)

    return demo
