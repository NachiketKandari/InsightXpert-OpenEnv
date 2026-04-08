# BIRD Text-to-SQL OpenEnv Environment

An [OpenEnv](https://github.com/open-env/openenv)-compliant environment where an AI agent translates natural language questions into executable SQL queries against real-world databases from the [BIRD Mini-Dev](https://huggingface.co/datasets/birdsql/bird_mini_dev) benchmark.

## Overview

- **15 curated tasks** across 3 difficulty levels (simple, moderate, challenging)
- **6 real-world SQLite databases** from the BIRD benchmark
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

15 tasks across 6 databases:

| Difficulty | Tasks | Databases |
|------------|-------|-----------|
| Simple (5) | Basic SELECT, WHERE, COUNT, JOIN | debit_card_specializing, toxicology, formula_1, california_schools, financial |
| Moderate (5) | JOIN + GROUP BY, HAVING, subqueries | european_football_2, toxicology, formula_1, california_schools, financial |
| Challenging (5) | Nested subqueries, percentages, CASE WHEN | european_football_2, toxicology, formula_1, california_schools, financial |

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
| `API_BASE_URL` | Gemini API | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | `gemini-2.0-flash` | Model to use for inference |
| `HF_TOKEN` / `API_KEY` | - | API key for the LLM |
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
    databases/             # BIRD Mini-Dev SQLite files
      california_schools/
      debit_card_specializing/
      european_football_2/
      financial/
      formula_1/
      toxicology/
  server/
    __init__.py
    app.py                 # FastAPI app via create_app()
    bird_environment.py    # BirdEnvironment(Environment) -- core logic
    grader.py              # Reward computation (Soft-F1 + execution accuracy)
```

## Safety

- Only SELECT queries are allowed (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE rejected)
- Each episode loads a fresh in-memory copy of the database (no state leakage)
- SQL execution has a 5-second timeout
- Maximum 5 steps per episode
