---
title: InsightXpert-OpenEnv
emoji: 🐦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# BIRD Text-to-SQL OpenEnv Environment

> **From [InsightXpert](https://github.com/NachiketKandari/InsightXpert)** — Text-to-SQL has been a core focus of my work, from building [InsightXpert](https://github.com/NachiketKandari/InsightXpert) (3rd Place at InsightX Challenge, IIT Bombay) to ongoing [research](https://github.com/NachiketKandari/InsightXpert-Research) on chat-driven database interfaces. OpenEnv gives me the opportunity to take this further — by framing Text-to-SQL as an RL environment, we can train models that iteratively refine their SQL generation through graded feedback rather than static supervision.

An [OpenEnv](https://github.com/open-env/openenv)-compliant environment where an AI agent translates natural language questions into executable SQL queries against real-world databases from the [BIRD Mini-Dev](https://huggingface.co/datasets/birdsql/bird_mini_dev) benchmark.

## Overview

- **200 curated tasks** across 3 difficulty levels (85 simple, 65 moderate, 50 challenging)
- **5 SQLite databases** (california_schools, debit_card_specializing, financial, formula_1, toxicology) from the BIRD benchmark
- **Fine-grained partial rewards** (0.0-1.0) via execution accuracy + Soft-F1 scoring
- **Self-correction loop**: agents get up to 5 attempts per task with grader feedback
- Lightweight: SQLite in-memory, no GPU required

## Action & Observation Spaces

### Action: `BirdSQLAction`

| Field | Type | Description |
|-------|------|-------------|
| `sql_query` | `str` | The SQL SELECT query to execute |

### Observation: `BirdSQLObservation`

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Task identifier (e.g., `simple_1`, `moderate_3`) |
| `db_id` | `str` | Database name |
| `difficulty` | `str` | `simple` / `moderate` / `challenging` |
| `question` | `str` | Natural language question |
| `evidence` | `str` | External knowledge hint |
| `schema_linking` | `str` | Relevant tables and columns for this question |
| `sample_rows` | `str` | Sample data rows (3 per table, on reset only) |
| `execution_result` | `str` | Query output or error message |
| `execution_success` | `bool` | Did the SQL execute without error? |
| `row_count` | `int` | Number of result rows |
| `column_names` | `list[str]` | Column names in result |
| `reward` | `float` | Grader reward (0.0-1.0) |
| `done` | `bool` | Episode finished? |
| `feedback` | `str` | Grader feedback for the agent |

## Reward Function

Uses BIRD-standard evaluation metrics (EX + Soft-F1):

| Condition | Reward | Feedback |
|-----------|--------|----------|
| Empty / not SELECT | 0.00 | No valid SQL query provided |
| SQL syntax error | 0.05 | Syntax error details |
| Runtime error (bad table/column) | 0.10 | Runtime error details |
| Executes but returns 0 rows | 0.15 | Check WHERE conditions and JOINs |
| Partial value match (Soft-F1) | 0.20-0.90 | Precision/recall breakdown |
| Relaxed EX (extra columns) | 0.95 | Simplify SELECT clause |
| Strict EX (exact match) | 1.00 | Perfect match! |

Order-sensitive comparison is used only when the gold SQL contains ORDER BY.

## Tasks

200 tasks across 5 databases (california_schools, debit_card_specializing, financial, formula_1, toxicology):

| Difficulty | Count | SQL Patterns |
|------------|-------|-------------|
| Simple (85) | Basic SELECT, JOIN, WHERE, DISTINCT, COUNT, ORDER BY |
| Moderate (65) | Multi-table JOIN, AVG, CASE WHEN, percentage calculations |
| Challenging (50) | Nested subqueries, CTE, IIF, STRFTIME, complex aggregation |

Tasks are selected from [BIRD Mini-Dev](https://huggingface.co/datasets/birdsql/bird_mini_dev) with a mix of difficulty: some are reliably solvable by current models, some are partially solvable, and some remain unsolved — ensuring meaningful reward gradients for RL training.

## Setup

### Local Development

```bash
# Install dependencies
pip install -e ".[inference]"

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run the baseline agent (in another terminal)
export API_KEY=your_api_key
python inference.py
```

### Docker

```bash
docker build -t bird-text2sql-env .
docker run -p 7860:7860 bird-text2sql-env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use for inference |
| `HF_TOKEN` / `API_KEY` | - | HuggingFace API token |
| `ENV_URL` | `http://localhost:7860` | Environment server URL |

## File Structure

```
__init__.py              # Package exports
models.py                # Pydantic: BirdSQLAction, BirdSQLObservation, BirdSQLState
client.py                # BirdText2SQLEnv(EnvClient) -- WebSocket client
inference.py             # Baseline agent (OpenAI client, stdout logging)
openenv.yaml             # Environment manifest
pyproject.toml           # Dependencies
requirements.txt         # For Docker
Dockerfile               # Docker build
README.md                # This file
data/
  tasks.json             # Curated task registry (200 tasks)
  schema_linking.json    # Pre-computed perfect schema linking per task
  databases/             # BIRD Mini-Dev SQLite files
    california_schools/
    debit_card_specializing/
    financial/
    formula_1/
    toxicology/
schema_linking/          # Standalone schema linking pipeline (from InsightXpert)
  models.py              # Pydantic models (schema + profile)
  db.py                  # SQLite wrapper
  schema_extractor.py    # Extract schema via PRAGMA
  stats_collector.py     # Collect column statistics
  perfect_linker.py      # Parse gold SQL, render pruned schema
  build_linking.py       # CLI to regenerate schema_linking.json
server/
  __init__.py
  app.py                 # FastAPI app via create_app()
  gradio_app.py          # Custom Gradio UI for BIRD environment
  bird_environment.py    # BirdEnvironment(Environment) -- core logic
  grader.py              # Reward computation (Soft-F1 + execution accuracy)
```

## Perfect Schema Linking

### Why we provide schema linking

Each task observation includes a **pruned schema** showing only the tables and columns relevant to the question, along with column descriptions and sample values. This is a deliberate design decision to isolate the model's SQL composition ability:

- **Without schema linking, RL training is infeasible.** Complex databases have 10+ tables and hundreds of columns. Models would almost never produce correct SQL on the first attempt, making the reward signal too sparse for RL to learn anything (the probability of a good answer must be greater than zero).
- **With schema linking, the hard part remains.** The model still has to solve logical reasoning over relational data: correct JOINs, WHERE vs HAVING, subqueries, aggregation, NULL handling, CASE WHEN expressions, and order-sensitive sorting.
- **State-of-the-art models still only achieve 60-75% execution accuracy on BIRD even with perfect schema linking.** This confirms there is substantial room for RL-driven improvement in the SQL composition step itself.
- **This aligns with RL best practices.** By providing the right schema context upfront, we ensure the model can plausibly reach a correct answer, enabling meaningful reward gradients during training.

### How it works

The schema linking pipeline (from [InsightXpert](https://github.com/NachiketKandari/InsightXpert)) works in two stages:

1. **Column profiling**: For each column in the database, the profiler collects type information, cardinality, null counts, min/max values, and up to 20 sample values. An LLM then generates short natural-language descriptions of each column based on these statistics.

2. **Gold SQL parsing**: Using [sqlglot](https://github.com/tobymao/sqlglot), the gold SQL query is parsed to extract the exact tables, columns, and literals referenced. Unqualified columns are resolved against the schema. PK/FK join path columns are automatically included to preserve connectivity between linked tables.

The result is a pruned schema text per task containing only the relevant tables/columns, annotated with descriptions, PK/FK relationships, and low-cardinality value lists.

### The `schema_linking/` directory

The `schema_linking/` directory contains the standalone schema linking pipeline, extracted from InsightXpert for reproducibility:

```
schema_linking/
  __init__.py            # Package docs
  models.py              # Pydantic models (DatabaseSchema, DatabaseProfile, etc.)
  db.py                  # Minimal SQLite wrapper
  schema_extractor.py    # Extract schema via PRAGMA queries
  stats_collector.py     # Collect per-column statistics
  perfect_linker.py      # Parse gold SQL, render pruned schema
  build_linking.py       # CLI: regenerate data/schema_linking.json
```

To regenerate the schema linking data:

```bash
# From scratch (stats only, no LLM column descriptions):
python3 -m schema_linking.build_linking

# With pre-computed profiles from InsightXpert (includes LLM descriptions):
python3 -m schema_linking.build_linking --profiles-dir /path/to/insightxpert/profiles/mini_dev
```

Requires: `pip install sqlglot pydantic`

## Safety

- Only SELECT and WITH (CTE) queries are allowed (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE rejected)
- Each episode loads a fresh in-memory copy of the database (no state leakage)
- SQL execution has a 5-second timeout
- Maximum 5 steps per episode
