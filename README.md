# BIRD Text-to-SQL OpenEnv Environment

> **From [InsightXpert](https://github.com/NachiketKandari/InsightXpert)** — Text-to-SQL has been a core focus of my work, from building [InsightXpert](https://github.com/NachiketKandari/InsightXpert) (3rd Place at InsightX Challenge, IIT Bombay) to ongoing [research](https://github.com/NachiketKandari/InsightXpert-Research) on chat-driven database interfaces. OpenEnv gives me the opportunity to take this further — by framing Text-to-SQL as an RL environment, we can train models that iteratively refine their SQL generation through graded feedback rather than static supervision.

An [OpenEnv](https://github.com/open-env/openenv)-compliant environment where an AI agent translates natural language questions into executable SQL queries against real-world databases from the [BIRD Mini-Dev](https://huggingface.co/datasets/birdsql/bird_mini_dev) benchmark.

## Overview

- **3 curated tasks** across 3 difficulty levels (simple, moderate, challenging)
- **Formula 1 SQLite database** (14 tables, 96 columns) from the BIRD benchmark
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
| `schema_ddl` | `str` | CREATE TABLE statements |
| `sample_rows` | `str` | Sample data rows (3 per table, on reset only) |
| `execution_result` | `str` | Query output or error message |
| `execution_success` | `bool` | Did the SQL execute without error? |
| `row_count` | `int` | Number of result rows |
| `column_names` | `list[str]` | Column names in result |
| `reward` | `float` | Grader reward (0.0-1.0) |
| `done` | `bool` | Episode finished? |
| `feedback` | `str` | Grader feedback for the agent |

## Reward Function

### Tier 0: Syntax/Runtime Error (0.00-0.15)

| Condition | Reward |
|-----------|--------|
| Empty / not SELECT | 0.00 |
| SQL syntax error | 0.05 |
| Runtime error (bad table/column) | 0.10 |
| Executes but returns 0 rows | 0.15 |

### Tier 1: Structural Match (0.20-0.40)

| Condition | Reward |
|-----------|--------|
| Correct column count | 0.20 |
| Correct column names | 0.30 |
| Column names + row count match | 0.40 |

### Tier 2: Partial Value Match (0.40-0.90)

Soft-F1 on result sets: `reward = 0.40 + (F1 * 0.50)`

For single-value numeric results, a closeness gradient is used instead.

### Tier 3: Perfect Match (1.0)

Gold and agent result sets are identical (order-insensitive unless gold has ORDER BY).

## Tasks

3 tasks on the Formula 1 database:

| Difficulty | Task | SQL Patterns |
|------------|------|-------------|
| Simple | Races held on circuits in Germany | JOIN, WHERE, DISTINCT |
| Moderate | Drivers who didn't finish Bahrain GP 2007 | Multi-table JOIN, NULL check, COUNT |
| Challenging | Race completion % of Japanese drivers 2007-2009 | JOIN, aggregation, CAST, BETWEEN, percentage |

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
openenv/
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
    tasks.json             # Curated task registry (15 tasks)
    schema_linking.json    # Pre-computed perfect schema linking per task
    databases/             # BIRD Mini-Dev SQLite files
      formula_1/
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

- Only SELECT queries are allowed (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE rejected)
- Each episode loads a fresh in-memory copy of the database (no state leakage)
- SQL execution has a 5-second timeout
- Maximum 5 steps per episode
