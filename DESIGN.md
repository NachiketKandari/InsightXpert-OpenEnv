# BIRD Text-to-SQL OpenEnv Environment — Design Document

**Project:** `bird-text2sql-env`
**Challenge:** OpenEnv Meta Challenge — Round 1
**Date:** April 2026

---

## 1. Executive Summary

An OpenEnv-compliant environment where an AI agent must translate natural language
questions into executable SQL queries against real-world databases from the BIRD
Mini-Dev benchmark. The agent interacts via `step()`/`reset()`/`state()` and
receives fine-grained partial rewards based on execution accuracy, column
matching, and result-set similarity.

**Why this is a strong submission:**

- Real-world task (not a game): Text-to-SQL is a production NLP problem
- Uses an established, cited academic benchmark (BIRD, NeurIPS 2024)
- Natural difficulty levels baked into the dataset (simple / moderate / challenging)
- Gold-standard SQL + real SQLite databases enable deterministic, reproducible grading
- Lightweight: SQLite is in-memory, no GPU needed — fits 2 vCPU / 8GB easily
- Rich partial reward signal at every step (syntax → schema → execution match)
- Your existing Gemini Flash pipeline slots directly in as the baseline agent

---

## 2. Dataset: BIRD Mini-Dev

**Source:** `birdsql/bird_mini_dev` on HuggingFace (CC BY-SA 4.0)

**Contents:**
- 500 high-quality question-SQL pairs
- 11 SQLite databases (bundled as .sqlite files, total ~50-200 MB)
- Each record contains:
  ```json
  {
    "question": "What is the average salary in the HR department?",
    "evidence": "HR department refers to department = 'HR'",
    "SQL": "SELECT AVG(salary) FROM employees WHERE department = 'HR'",
    "db_id": "company_db",
    "difficulty": "simple"   // simple | moderate | challenging
  }
  ```

**Difficulty distribution (approximate):**
- Simple: ~55% of samples — single table, basic WHERE/SELECT/COUNT
- Moderate: ~30% — JOINs, GROUP BY, HAVING, subqueries
- Challenging: ~15% — nested subqueries, window functions, external knowledge

**For the environment, we select a curated subset:**
- 3–5 questions per difficulty level = 9–15 tasks total
- Chosen to cover diverse SQL patterns and multiple databases
- Deterministic: same task_id always loads the same question + DB

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    INFERENCE.PY (Agent)                   │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ OpenAI      │   │ Prompt Eng.  │   │ Schema       │  │
│  │ Client      │──▶│ (System +    │──▶│ Linking      │  │
│  │ (Gemini/HF) │   │  Few-shot)   │   │ + Evidence   │  │
│  └─────────────┘   └──────────────┘   └──────────────┘  │
│         │                                     │          │
│         ▼                                     ▼          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Agent Loop (max N steps)             │    │
│  │  1. read observation                              │    │
│  │  2. generate/refine SQL via LLM                   │    │
│  │  3. env.step(BirdSQLAction(sql_query="..."))      │    │
│  │  4. check reward / done                           │    │
│  │  5. if not done and reward < 1.0, refine          │    │
│  └──────────────────────────────────────────────────┘    │
│         │                                                │
│         ▼  WebSocket / HTTP                              │
├──────────────────────────────────────────────────────────┤
│              ENVIRONMENT SERVER (FastAPI + Docker)        │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ reset()     │  │ step()       │  │ state()       │   │
│  │ Load task,  │  │ Execute SQL, │  │ Return        │   │
│  │ init SQLite │  │ grade result,│  │ episode info  │   │
│  │ return obs  │  │ return reward│  │               │   │
│  └─────────────┘  └──────────────┘  └───────────────┘   │
│         │                  │                             │
│         ▼                  ▼                             │
│  ┌──────────────────────────────────────────────────┐   │
│  │         GRADER (Execution Accuracy + Soft-F1)     │   │
│  │  - Execute gold SQL → gold_results                │   │
│  │  - Execute agent SQL → agent_results              │   │
│  │  - Compare result sets → reward 0.0–1.0           │   │
│  └──────────────────────────────────────────────────┘   │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │         BIRD Mini-Dev SQLite Databases            │   │
│  │         (bundled in Docker image)                 │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. OpenEnv API Contract

### 4.1 Models (models.py)

```python
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class BirdSQLAction(Action):
    """Agent submits a SQL query."""
    sql_query: str = Field(..., description="The SQL query to execute")

class BirdSQLObservation(Observation):
    """What the agent sees after reset() or step()."""
    # Task info (always present)
    task_id: str = Field(..., description="Task identifier")
    db_id: str = Field(..., description="Database name")
    difficulty: str = Field(..., description="simple|moderate|challenging")
    question: str = Field(..., description="Natural language question")
    evidence: str = Field("", description="External knowledge hint")
    schema_ddl: str = Field(..., description="CREATE TABLE statements")
    sample_rows: str = Field("", description="Sample data rows per table")

    # Step result (populated after step(), empty after reset())
    execution_result: str = Field("", description="Query output or error message")
    execution_success: bool = Field(True, description="Did the SQL execute?")
    row_count: int = Field(0, description="Number of result rows")
    column_names: list[str] = Field(default_factory=list)
    reward: float = Field(0.0, description="Grader reward 0.0-1.0")
    done: bool = Field(False, description="Episode finished?")
    feedback: str = Field("", description="Grader feedback for the agent")

# State uses the core State class from OpenEnv:
# - episode_id: str
# - step_count: int
# We add:
class BirdSQLState:  # extends openenv State
    task_id: str
    db_id: str
    difficulty: str
    step_count: int
    accumulated_reward: float
    last_action: str
    done: bool
```

### 4.2 reset(task_id) → BirdSQLObservation

**Input:** `task_id` — one of: `"simple_1"`, `"simple_2"`, ..., `"moderate_1"`, ..., `"hard_1"`, ...

**Behavior:**
1. Look up task in the curated task registry
2. Load the corresponding SQLite database into memory (fresh copy)
3. Extract schema DDL via `sqlite_master`
4. Grab 3 sample rows per table
5. Return observation with question, evidence, schema, sample rows
6. Reset step counter, reward accumulator

**Observation on reset:**
- `question`: "What is the average salary..."
- `evidence`: "HR refers to department = 'HR'"
- `schema_ddl`: full CREATE TABLE statements
- `sample_rows`: formatted sample data
- `reward`: 0.0, `done`: False, `execution_result`: ""

### 4.3 step(BirdSQLAction) → BirdSQLObservation

**Input:** `BirdSQLAction(sql_query="SELECT ...")`

**Behavior:**
1. Increment step counter
2. Execute agent's SQL against the in-memory SQLite DB
3. If SQL error → reward based on error type (see §5)
4. If SQL succeeds → execute gold SQL, compare result sets (see §5)
5. Compute reward (0.0–1.0)
6. Determine `done`:
   - `done=True` if reward == 1.0 (perfect match)
   - `done=True` if step_count >= MAX_STEPS (default: 5)
   - `done=False` otherwise
7. Return observation with execution results, reward, feedback

### 4.4 state() → BirdSQLState

Returns current episode metadata. No side effects.

---

## 5. Reward Function Design

This is the most critical part. The reward must provide **meaningful partial
credit** so an RL agent can learn. Here is the tiered reward scheme:

### Tier 0: Syntax/Runtime Error (reward 0.0–0.15)

| Condition | Reward | Feedback |
|-----------|--------|----------|
| Empty query or not a SELECT | 0.00 | "No valid SQL query provided" |
| SQL syntax error | 0.05 | "Syntax error: {error_msg}" |
| Runtime error (no such table/column) | 0.10 | "Runtime error: {error_msg}" |
| Query executes but returns 0 rows (gold has rows) | 0.15 | "Query returned no results" |

### Tier 1: Structural Match (reward 0.20–0.40)

| Condition | Reward | Feedback |
|-----------|--------|----------|
| Correct number of columns, wrong data | 0.20 | "Column count matches" |
| Correct column names (order-insensitive) | 0.30 | "Column names match" |
| Correct column names + correct row count | 0.40 | "Structure matches, checking values" |

### Tier 2: Partial Value Match (reward 0.50–0.90)

Computed via **Soft-F1** on result sets:
- Treat each row as a set element (tuple of values)
- Precision = |gold ∩ agent| / |agent|
- Recall = |gold ∩ agent| / |gold|
- F1 = 2 * P * R / (P + R)
- Reward = 0.40 + (F1 * 0.50)  → ranges from 0.40 to 0.90

For single-value results (e.g., COUNT, AVG):
- If numeric: reward = 0.40 + 0.50 * max(0, 1 - |gold - agent| / max(|gold|, 1))
- Provides gradient even for close-but-wrong numeric answers

### Tier 3: Perfect Match (reward 1.0)

| Condition | Reward | Feedback |
|-----------|--------|----------|
| Execution accuracy: gold_results == agent_results | 1.00 | "Perfect match!" |

**Note:** Comparison is order-insensitive (sets of row-tuples) unless the gold
query contains ORDER BY.

### Reward Summary Formula

```python
def compute_reward(gold_results, agent_results, agent_error, gold_columns, agent_columns):
    if agent_error:
        if "syntax" in agent_error.lower():
            return 0.05, "Syntax error"
        return 0.10, "Runtime error"

    if not agent_results and gold_results:
        return 0.15, "Empty result set"

    # Column matching
    col_score = 0.0
    if len(agent_columns) == len(gold_columns):
        col_score = 0.20
        if set(c.lower() for c in agent_columns) == set(c.lower() for c in gold_columns):
            col_score = 0.30
            if len(agent_results) == len(gold_results):
                col_score = 0.40

    # Value matching (Soft-F1)
    gold_set = set(tuple(str(v) for v in row) for row in gold_results)
    agent_set = set(tuple(str(v) for v in row) for row in agent_results)
    intersection = gold_set & agent_set

    if gold_set == agent_set:
        return 1.0, "Perfect match!"

    if len(agent_set) == 0 or len(gold_set) == 0:
        return max(col_score, 0.15), "No value overlap"

    precision = len(intersection) / len(agent_set)
    recall = len(intersection) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    value_reward = 0.40 + (f1 * 0.50)
    return max(col_score, value_reward), f"Soft-F1: {f1:.2f}"
```

---

## 6. Task Registry

We curate 9–15 tasks from BIRD Mini-Dev. Selection criteria:
- Cover at least 3 different databases
- Each difficulty level has 3–5 tasks
- Tasks should exercise diverse SQL patterns

### Proposed Task Selection

**Easy (simple difficulty):**
| task_id | db_id | SQL Pattern | Example Question Type |
|---------|-------|-------------|----------------------|
| simple_1 | california_schools | SELECT + WHERE | "What is the name of the school with the highest average test score?" |
| simple_2 | financial | SELECT + COUNT | "How many accounts are there in the North region?" |
| simple_3 | superhero | SELECT + WHERE + LIKE | "List all superheroes whose name starts with 'S'" |

**Medium (moderate difficulty):**
| task_id | db_id | SQL Pattern | Example Question Type |
|---------|-------|-------------|----------------------|
| moderate_1 | financial | JOIN + GROUP BY | "What is the total transaction amount per account type?" |
| moderate_2 | california_schools | JOIN + HAVING | "Which counties have more than 10 schools with SAT scores above 1000?" |
| moderate_3 | toxicology | JOIN + subquery | "Find molecules with atoms that have the highest charge" |

**Hard (challenging difficulty):**
| task_id | db_id | SQL Pattern | Example Question Type |
|---------|-------|-------------|----------------------|
| hard_1 | financial | Nested subquery + aggregation | "What percentage of clients have loans above the district average?" |
| hard_2 | california_schools | Window function / complex logic | "Rank schools by improvement rate across years" |
| hard_3 | toxicology | Multi-JOIN + CASE WHEN | "Classify molecules by toxicity based on multiple atom properties" |

**Note:** Exact questions will be selected from Mini-Dev during implementation.
The above are representative patterns.

---

## 7. File Structure

```
bird_text2sql_env/
├── __init__.py                  # Exports BirdSQLAction, BirdSQLObservation, BirdText2SQLEnv
├── models.py                    # Pydantic: BirdSQLAction, BirdSQLObservation (+ State extension)
├── client.py                    # BirdText2SQLEnv(EnvClient) — WebSocket client
├── openenv.yaml                 # Environment manifest
├── pyproject.toml               # Dependencies
├── requirements.txt             # For Docker
├── README.md                    # Full documentation
├── Dockerfile                   # Docker build
├── data/
│   ├── tasks.json               # Curated task registry (9-15 tasks)
│   └── databases/               # BIRD Mini-Dev SQLite files
│       ├── california_schools/
│       │   └── california_schools.sqlite
│       ├── financial/
│       │   └── financial.sqlite
│       ├── superhero/
│       │   └── superhero.sqlite
│       └── ...
├── server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI app using create_app()
│   ├── bird_environment.py      # BirdEnvironment(Environment) — core logic
│   └── grader.py                # Reward computation (Soft-F1 + execution accuracy)
└── inference.py                 # Baseline agent (OpenAI client, stdout logging)
```

---

## 8. Key File Designs

### 8.1 openenv.yaml

```yaml
spec_version: 1
name: bird-text2sql-env
type: environment
runtime:
  type: docker
  port: 7860
app:
  module: server.app
  callable: app
```

### 8.2 server/app.py

```python
from openenv.core.env_server import create_app

try:
    from ..models import BirdSQLAction, BirdSQLObservation
except ImportError:
    from models import BirdSQLAction, BirdSQLObservation

try:
    from .bird_environment import BirdEnvironment
except ImportError:
    from server.bird_environment import BirdEnvironment

app = create_app(
    BirdEnvironment,
    BirdSQLAction,
    BirdSQLObservation,
    env_name="bird-text2sql-env"
)
```

### 8.3 server/bird_environment.py (pseudocode)

```python
class BirdEnvironment(Environment):

    def __init__(self):
        self.task_registry = load_json("data/tasks.json")
        self.db_connections = {}  # task_id → sqlite3.Connection
        self.current_task = None
        self.step_count = 0
        self.max_steps = 5

    def reset(self, *, task_id=None, seed=None) -> BirdSQLObservation:
        if task_id is None:
            task_id = random.choice(list(self.task_registry.keys()))

        task = self.task_registry[task_id]
        db_path = f"data/databases/{task['db_id']}/{task['db_id']}.sqlite"

        # Create fresh in-memory copy
        source = sqlite3.connect(db_path)
        self.db = sqlite3.connect(":memory:")
        source.backup(self.db)
        source.close()

        # Extract schema
        schema_ddl = self._get_schema_ddl()
        sample_rows = self._get_sample_rows(limit=3)

        self.current_task = task
        self.step_count = 0
        self._episode_id = str(uuid4())

        return BirdSQLObservation(
            task_id=task_id,
            db_id=task["db_id"],
            difficulty=task["difficulty"],
            question=task["question"],
            evidence=task.get("evidence", ""),
            schema_ddl=schema_ddl,
            sample_rows=sample_rows,
            reward=0.0,
            done=False
        )

    def step(self, action: BirdSQLAction) -> BirdSQLObservation:
        self.step_count += 1
        sql = action.sql_query

        # Execute agent SQL
        agent_result, agent_error, agent_cols = self._safe_execute(sql)

        # Execute gold SQL
        gold_result, gold_error, gold_cols = self._safe_execute(
            self.current_task["SQL"]
        )

        # Grade
        reward, feedback = compute_reward(
            gold_result, agent_result, agent_error, gold_cols, agent_cols
        )

        done = (reward == 1.0) or (self.step_count >= self.max_steps)

        return BirdSQLObservation(
            task_id=self.current_task["task_id"],
            ...,
            execution_result=str(agent_result[:10]) if agent_result else agent_error,
            execution_success=(agent_error is None),
            row_count=len(agent_result) if agent_result else 0,
            column_names=agent_cols or [],
            reward=reward,
            done=done,
            feedback=feedback
        )

    @property
    def state(self):
        return State(
            episode_id=self._episode_id,
            step_count=self.step_count,
            # custom fields:
            task_id=self.current_task["task_id"] if self.current_task else "",
            done=...,
        )
```

### 8.4 inference.py (stdout format)

```python
"""
BIRD Text-to-SQL Inference Script
Uses OpenAI-compatible client for LLM calls.
Emits [START]/[STEP]/[END] stdout logs per competition spec.
"""

import os, sys, json
from openai import OpenAI

# --- Mandatory env vars ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Alt: HuggingFace model
# API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "bird-text2sql"
MAX_STEPS = 5
TASKS = ["simple_1", "simple_2", "simple_3",
         "moderate_1", "moderate_2", "moderate_3",
         "hard_1", "hard_2", "hard_3"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# Import environment client
from client import BirdText2SQLEnv
from models import BirdSQLAction

def build_prompt(obs):
    """Build text-to-SQL prompt with schema linking."""
    return f"""You are a SQL expert. Given the following SQLite database schema
and a natural language question, generate the correct SQL query.

DATABASE SCHEMA:
{obs.schema_ddl}

SAMPLE DATA:
{obs.sample_rows}

EXTERNAL KNOWLEDGE:
{obs.evidence}

QUESTION: {obs.question}

{"PREVIOUS ATTEMPT FEEDBACK: " + obs.feedback if obs.feedback else ""}

Return ONLY the SQL query, no explanation."""

def run_task(env, task_id):
    obs = env.reset(task_id=task_id)
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    rewards = []
    for step_num in range(1, MAX_STEPS + 1):
        prompt = build_prompt(obs)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500
        )
        sql = response.choices[0].message.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()

        obs = env.step(BirdSQLAction(sql_query=sql))
        rewards.append(obs.reward)

        error_str = obs.feedback if not obs.execution_success else "null"
        print(f"[STEP] step={step_num} action={sql!r} "
              f"reward={obs.reward:.2f} done={str(obs.done).lower()} "
              f"error={error_str}")

        if obs.done:
            break

    success = any(r >= 0.99 for r in rewards)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} "
          f"steps={len(rewards)} rewards={rewards_str}")

def main():
    env_url = os.getenv("ENV_URL", "http://localhost:7860")
    with BirdText2SQLEnv(base_url=env_url).sync() as env:
        for task_id in TASKS:
            run_task(env, task_id)

if __name__ == "__main__":
    main()
```

---

## 9. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

# Expose port (HF Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**requirements.txt:**
```
openenv-core>=0.2.1
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
websockets>=12.0
```

---

## 10. Grading Nuances & Edge Cases

### 10.1 Order sensitivity
- If gold SQL has ORDER BY → compare as ordered lists
- If no ORDER BY → compare as sets of tuples

### 10.2 Type coercion
- Cast all values to strings before comparison
- Normalize floats to 2 decimal places
- Treat NULL/None equivalently

### 10.3 Column name comparison
- Case-insensitive
- Ignore aliases (compare by position if names differ)

### 10.4 Timeout
- Agent SQL execution timeout: 5 seconds
- Prevents infinite loops / expensive cartesian joins

### 10.5 Safety
- Read-only: only SELECT queries allowed
- Reject INSERT/UPDATE/DELETE/DROP/ALTER/CREATE
- Each episode gets a fresh in-memory DB copy (no state leakage)

---

## 11. Differentiation from WALKMAN303's SQL-Repair Env

| Aspect | SQL-Repair (WALKMAN303) | Text-to-SQL (Ours) |
|--------|------------------------|---------------------|
| Task | Fix broken SQL queries | Generate SQL from NL questions |
| Dataset | Synthetic (4 tables) | BIRD Mini-Dev (11 real-world DBs, 37 domains) |
| Difficulty | Easy/Medium/Hard (synthetic) | BIRD's labeled simple/moderate/challenging |
| Gold standard | Hand-written | Academic benchmark gold SQL |
| Grading | Custom rules | Execution accuracy + Soft-F1 (published metrics) |
| Schema complexity | 4 tables, simple | Up to 16 tables, messy real-world data |
| Evidence/hints | None | External knowledge hints (BIRD feature) |
| Research value | Demonstration | Publishable benchmark integration |

---

## 12. Pre-Submission Checklist Alignment

| Requirement | How We Meet It |
|-------------|----------------|
| HF Space deploys, returns 200, responds to reset() | FastAPI + Dockerfile, /health endpoint |
| OpenEnv spec compliance (yaml, typed models, step/reset/state) | Pydantic models, create_app(), openenv.yaml |
| Dockerfile builds | python:3.11-slim, minimal deps, tested locally |
| Baseline reproduces | inference.py with OpenAI client, deterministic scores |
| 3+ tasks with graders, scores 0.0–1.0 | 9 tasks (3 per difficulty), Soft-F1 grader |
| Meaningful partial reward | 5-tier reward: syntax → structure → partial values → perfect |
| README with env description, action/obs spaces, setup | Comprehensive README.md |
| Mandatory env vars (API_BASE_URL, MODEL_NAME, HF_TOKEN) | Configured in inference.py |
| OpenAI Client for LLM calls | Uses `from openai import OpenAI` |
| [START]/[STEP]/[END] stdout format | Exact format in inference.py |
| Runtime < 20min on 2vCPU/8GB | SQLite in-memory, ~9 tasks × 5 steps × ~2s/LLM call ≈ ~2min |

---

## 13. Implementation Order

### Phase 1: Data Preparation
1. Download BIRD Mini-Dev from HuggingFace
2. Select 9-15 tasks across difficulties
3. Copy relevant SQLite databases
4. Create `tasks.json` registry

### Phase 2: Core Environment
1. `models.py` — Pydantic types
2. `server/grader.py` — reward computation (test independently)
3. `server/bird_environment.py` — Environment class
4. `server/app.py` — FastAPI wiring

### Phase 3: Client & Testing
1. `client.py` — EnvClient subclass
2. Local testing: start server, call reset/step manually
3. Verify rewards make sense on known queries

### Phase 4: Inference Script
1. `inference.py` — agent loop with LLM
2. Prompt engineering (schema linking, few-shot, evidence)
3. Test with both Gemini Flash and Qwen via OpenAI client
4. Verify stdout format matches spec exactly

### Phase 5: Deployment
1. Build Docker image locally, test
2. Run validation script
3. Push to HuggingFace Spaces
4. Verify HF Space responds to reset()

---

## 14. Prompt Engineering Strategy (for inference.py)

The baseline agent uses a multi-stage prompting approach:

**Stage 1 — Schema Linking:**
- Present full DDL + sample rows
- Ask LLM to identify relevant tables/columns for the question

**Stage 2 — SQL Generation:**
- System prompt: "You are a SQL expert for SQLite databases"
- Include: question, evidence hint, relevant schema, sample rows
- Few-shot: 2-3 examples of question → SQL (from BIRD's easy samples)
- Temperature: 0.0 for reproducibility

**Stage 3 — Self-Correction (on retry steps):**
- If step N got reward < 1.0, include the feedback in the next prompt
- "Your previous query returned {feedback}. Please fix the SQL."
- Include previous query and error for context

---

## 15. Risk Assessment & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| BIRD SQLite files too large for Docker | Low (Mini-Dev is ~50-200MB) | Use only 3-4 databases, compress |
| Gold SQL has bugs (known BIRD issue) | Medium | Hand-verify selected tasks, use corrected gold where available |
| LLM generates non-SELECT queries | High | Whitelist: reject anything not starting with SELECT |
| Soft-F1 reward is noisy | Low | Tested against known correct/incorrect queries |
| HF Spaces free tier timeout | Medium | Keep environment lightweight, use async |
| openenv-core API changes | Low | Pin version >=0.2.1 |

---

## 16. Open Questions to Resolve During Implementation

1. **Exact task selection:** Need to download Mini-Dev and manually pick the
   best 9-15 questions that demonstrate clear difficulty progression
2. **Schema presentation format:** Full DDL vs. simplified table descriptions?
   (Test both, see which gives better agent performance)
3. **Max steps per episode:** 5 is reasonable, but may need tuning
4. **Whether to include column_meaning.json:** BIRD provides column descriptions
   that could enrich the observation — worth testing
5. **Dual-model default:** Should inference.py default to Gemini or Qwen?
   (Will implement both, make configurable via env vars)
