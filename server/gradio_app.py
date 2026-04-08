"""Custom Gradio UI for InsightXpert-OpenEnv.

Replaces the default OpenEnv web interface with a Text-to-SQL-specific
dashboard featuring reward visualization, difficulty badges, and step tracing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd


# ── Constants ───────────────────────────────────────────────────────────────

DIFF_BADGE = {
    "simple": "🟢 SIMPLE",
    "moderate": "🟡 MODERATE",
    "challenging": "🔴 CHALLENGING",
}

DIFF_COLOR = {
    "simple": "#22c55e",
    "moderate": "#eab308",
    "challenging": "#ef4444",
}

CSS = """
.reward-bar-outer {
    background: #1e293b; border-radius: 8px; height: 28px;
    overflow: hidden; position: relative; border: 1px solid #334155;
}
.reward-bar-inner {
    height: 100%; border-radius: 7px; transition: width 0.4s ease;
    display: flex; align-items: center; justify-content: flex-end;
    padding-right: 8px; font-weight: 700; font-size: 13px; color: #fff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5); min-width: 40px;
}
.reward-perfect  { background: linear-gradient(90deg, #22c55e, #4ade80); }
.reward-high     { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.reward-mid      { background: linear-gradient(90deg, #eab308, #facc15); }
.reward-low      { background: linear-gradient(90deg, #ef4444, #f87171); }
.stat-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 14px 18px; text-align: center;
}
.stat-card .stat-value { font-size: 24px; font-weight: 700; color: #e2e8f0; }
.stat-card .stat-label { font-size: 12px; color: #94a3b8; margin-top: 2px; }
.tier-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 8px; margin-bottom: 6px;
}
.tier-bar {
    height: 18px; border-radius: 4px; flex-shrink: 0;
}
.tier-label { font-size: 14px; color: #e2e8f0; flex: 1; }
.tier-value { font-size: 14px; font-weight: 700; color: #e2e8f0; width: 50px; text-align: right; }
.step-result-card {
    border-radius: 10px; padding: 16px 20px; margin: 8px 0;
    border: 1px solid #334155;
}
.episode-done {
    background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(34,197,94,0.02));
    border-color: #22c55e;
}
.result-header { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
"""


# ── Formatting helpers ──────────────────────────────────────────────────────


def _schema_to_markdown(schema: str) -> str:
    """Convert schema linking text to formatted markdown."""
    lines: List[str] = []
    for raw_line in schema.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("Table:"):
            table_name = stripped.replace("Table:", "").strip().strip('"')
            if lines:
                lines.append("")
            lines.append(f"**`{table_name}`**")
        elif stripped.startswith("Columns:"):
            continue
        elif stripped.startswith("- "):
            col_text = stripped[2:]
            quote_end = col_text.rfind('"')
            if quote_end != -1:
                type_start = col_text.find("(", quote_end)
            else:
                type_start = col_text.find("(")
            if type_start != -1:
                type_end = col_text.find(")", type_start)
            else:
                type_end = -1
            if type_end != -1 and col_text[type_end + 1 : type_end + 3] == ": ":
                col_header = col_text[: type_end + 1]
                col_desc = col_text[type_end + 3 :].strip()
                lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;`{col_header}` — {col_desc}")
            else:
                lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;`{col_text}`")
        elif stripped:
            lines.append(stripped)
    return "\n\n".join(lines)


def _sample_rows_to_markdown(sample: str) -> str:
    """Convert pipe-separated sample rows to markdown tables."""
    tables: List[str] = []
    current_header = ""
    header_row = ""
    data_rows: List[str] = []

    def _flush():
        nonlocal header_row, data_rows
        if not header_row:
            return
        cols = [c.strip() for c in header_row.split("|")]
        md = "| " + " | ".join(cols) + " |\n"
        md += "| " + " | ".join("---" for _ in cols) + " |\n"
        for dr in data_rows:
            vals = [v.strip() for v in dr.split("|")]
            while len(vals) < len(cols):
                vals.append("")
            vals = vals[: len(cols)]
            md += "| " + " | ".join(vals) + " |\n"
        label = f"**`{current_header}`**\n\n" if current_header else ""
        tables.append(label + md)
        header_row = ""
        data_rows = []

    for raw_line in sample.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("-- "):
            _flush()
            current_header = stripped[3:].strip()
        elif set(stripped) <= {"-", " "}:
            continue
        elif "|" in stripped:
            if not header_row:
                header_row = stripped
            else:
                data_rows.append(stripped)
    _flush()
    return "\n\n".join(tables)


def _reward_bar_html(reward: float) -> str:
    """Render a colored reward progress bar as HTML."""
    pct = max(0, min(100, reward * 100))
    if reward >= 0.95:
        css_cls = "reward-perfect"
    elif reward >= 0.5:
        css_cls = "reward-high"
    elif reward >= 0.2:
        css_cls = "reward-mid"
    else:
        css_cls = "reward-low"
    return (
        f'<div class="reward-bar-outer">'
        f'<div class="reward-bar-inner {css_cls}" style="width:{max(pct, 8):.0f}%">'
        f'{reward:.2f}'
        f'</div></div>'
    )


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
        f"> {question}",
    ]
    if evidence:
        lines += ["", f"**Evidence:** _{evidence}_"]
    if schema:
        lines += [
            "",
            "<details><summary><b>Relevant Schema</b></summary>",
            "",
            f"```\n{schema}\n```",
            "",
            "</details>",
        ]
    sample = obs.get("sample_rows", "")
    if sample:
        sample_md = _sample_rows_to_markdown(sample)
        lines += [
            "",
            "<details><summary><b>Sample Rows</b></summary>",
            "",
            sample_md,
            "",
            "</details>",
        ]
    return "\n".join(lines)


def _format_step_result(obs: Dict[str, Any], reward: float) -> str:
    """Format step result with visual reward bar and feedback."""
    done = obs.get("done", False)
    feedback = obs.get("feedback", "")
    exec_result = obs.get("execution_result", "")
    row_count = obs.get("row_count", 0)
    exec_success = obs.get("execution_success", True)
    step_count = obs.get("step_count", "")

    bar = _reward_bar_html(reward)

    done_class = ' episode-done' if done and reward >= 0.95 else ''
    lines = [f'<div class="step-result-card{done_class}">']
    lines.append(f'<div class="result-header">')
    lines.append(f'<span style="font-size:20px;font-weight:700">Reward</span>')
    lines.append(f'</div>')
    lines.append(bar)
    lines.append(f'<div style="margin-top:10px">')

    if feedback:
        lines.append(f'<b>Feedback:</b> {feedback}')

    lines.append(f'</div>')

    if done:
        if reward >= 0.95:
            lines.append('<div style="margin-top:10px;color:#4ade80;font-weight:600">Episode complete &mdash; Perfect score!</div>')
        else:
            lines.append('<div style="margin-top:10px;color:#94a3b8;font-weight:600">Episode complete.</div>')

    lines.append('</div>')

    # Execution result as markdown below the HTML card
    md_lines = ["\n".join(lines)]
    if exec_result:
        preview = exec_result[:500]
        if len(exec_result) > 500:
            preview += "\n..."
        status = "executed" if exec_success else "error"
        md_lines += [
            "",
            f"**Query {status}** ({row_count} rows):",
            f"```\n{preview}\n```",
        ]
    return "\n".join(md_lines)


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


# ── Reward system visual ────────────────────────────────────────────────────

def _reward_tier_html() -> str:
    """Visual reward tier diagram."""
    tiers = [
        ("1.00", "Strict EX — exact result-set match", 100, "#22c55e"),
        ("0.95", "Relaxed EX — correct values, extra columns tolerated", 95, "#3b82f6"),
        ("0.20 – 0.90", "Soft-F1 — row-element partial credit", 70, "#eab308"),
        ("0.15", "Zero rows — query ran but returned nothing", 15, "#f97316"),
        ("0.10", "Runtime error — bad table/column reference", 10, "#ef4444"),
        ("0.05", "Syntax error — malformed SQL", 5, "#dc2626"),
        ("0.00", "Rejected — not a SELECT query or forbidden statement", 0, "#64748b"),
    ]
    rows = []
    for value, label, width, color in tiers:
        bar_w = max(width, 3)
        rows.append(
            f'<div class="tier-row">'
            f'<div class="tier-value">{value}</div>'
            f'<div class="tier-bar" style="width:{bar_w}%;background:{color}"></div>'
            f'<div class="tier-label">{label}</div>'
            f'</div>'
        )
    return '<div style="max-width:700px">' + "\n".join(rows) + '</div>'


def _stat_card(value: str, label: str) -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f'</div>'
    )


REWARD_SYSTEM_MD = f"""
## Reward Tiers

Rewards are computed using BIRD-standard evaluation metrics.
Order-sensitive comparison is used when the gold SQL contains `ORDER BY`.

{_reward_tier_html()}

---

## Task Distribution

<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:500px;margin:16px 0">
{_stat_card("50", "Simple")}
{_stat_card("50", "Moderate")}
{_stat_card("50", "Challenging")}
</div>

**150 total tasks** across **5 SQLite databases** from the BIRD Mini-Dev benchmark.

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

## Environment Constraints

| Parameter | Value |
|-----------|-------|
| Max steps per episode | 5 |
| SQL timeout | 5 seconds |
| Allowed queries | `SELECT`, `WITH` (CTE) |
| Database isolation | Fresh in-memory copy per episode |
| Concurrent sessions | Up to 64 (GRPO-ready) |
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
                f"**Error:** {e}",
                f'{{"error": "{e}"}}',
            )
        obs = data.get("observation", {})
        task_md = _format_task_info(obs)
        return (
            task_md,
            [],
            _make_chart_df([]),
            _make_trace_df([]),
            "*Write a SQL query above and click* ***Execute SQL*** *to begin.*",
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
            "**BIRD Text-to-SQL RL Environment** &mdash; "
            "translate natural language to SQL, "
            "get graded, self-correct"
        )

        # Stats row
        gr.HTML(
            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:8px 0 16px 0">'
            + _stat_card("150", "Tasks")
            + _stat_card("5", "Databases")
            + _stat_card("5", "Steps / Episode")
            + _stat_card("0 – 1", "Reward Range")
            + '</div>'
        )

        with gr.Tabs():
            # ── Tab 1: Interactive Demo ──────────────────────────────
            with gr.Tab("Interactive Demo"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        task_display = gr.Markdown(
                            "*Click* ***Reset*** *to load a random task.*"
                        )
                    with gr.Column(scale=2, min_width=280):
                        reward_chart = gr.LinePlot(
                            value=_make_chart_df([]),
                            x="Step",
                            y="Reward",
                            title="Reward per Step",
                            y_lim=[0.0, 1.05],
                            height=240,
                        )

                gr.Markdown("---")

                sql_input = gr.Code(
                    language="sql",
                    label="SQL Query",
                    lines=5,
                    value="SELECT ",
                )

                with gr.Row():
                    step_btn = gr.Button(
                        "Execute SQL", variant="primary", scale=2,
                    )
                    reset_btn = gr.Button(
                        "Reset / New Task", variant="secondary", scale=1,
                    )

                result_display = gr.Markdown("*No result yet.*")

                with gr.Accordion("Episode Trace", open=False):
                    trace_table = gr.DataFrame(
                        value=_make_trace_df([]),
                        label="Step History",
                        wrap=True,
                        interactive=False,
                    )

                with gr.Accordion("Raw JSON", open=False):
                    raw_json = gr.Code(
                        label="Response",
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
