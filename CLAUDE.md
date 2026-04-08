# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Authorship Policy

Do NOT create commits attributed to or co-authored by Claude. No `Co-Authored-By` lines referencing Claude or any AI. All commits must be authored solely by the human developer.

## What This Project Is

A Text-to-SQL task environment implementing the OpenEnv specification, deployed to HuggingFace Spaces (Docker SDK, port 7860). AI agents translate natural language questions into executable SQL queries against real-world SQLite databases from the BIRD benchmark. 200 tasks span 3 difficulty levels (85 simple, 65 moderate, 50 challenging) across 5 SQLite databases (california_schools, debit_card_specializing, financial, formula_1, toxicology).

## Commands

```bash
# Install dependencies (uses uv — uv.lock is committed)
pip install -e ".[inference]"

# Start environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline agent (requires API_KEY or HF_TOKEN env var)
python inference.py

# Docker build & run
docker build -t bird-text2sql-env .
docker run -p 7860:7860 bird-text2sql-env

# Rebuild tasks.json + schema_linking.json from InsightXpert perfect_linking data
python3 scripts/build_tasks.py

# Regenerate schema linking data (offline tool, not runtime)
python3 -m schema_linking.build_linking
# With InsightXpert profiles:
python3 -m schema_linking.build_linking --profiles-dir /path/to/insightxpert/profiles/mini_dev
```

## Environment Variables (for inference.py)

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | HF router LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen3-8B` | LLM model (must be on HuggingFace) |
| `HF_TOKEN` | Required (no default) | HuggingFace API token |
| `ENV_URL` | `http://localhost:7860` | Environment server URL |

## Architecture

The project follows the OpenEnv contract: typed action/observation/state models, reset/step/state endpoints via REST, and a YAML manifest.

**Core flow:** Client (`client.py`) connects via WebSocket to FastAPI server (`server/app.py`) which delegates to `BirdEnvironment` (`server/bird_environment.py`). The environment loads tasks from `data/tasks.json`, creates in-memory SQLite copies, executes agent SQL, and grades results via `server/grader.py`.

**Key modules:**
- `models.py` — Pydantic models: `BirdSQLAction` (agent input), `BirdSQLObservation` (env output), `BirdSQLState` (episode metadata). Note: `reward`, `done`, `episode_id`, `step_count` are inherited from OpenEnv base classes, not re-declared here.
- `server/bird_environment.py` — Core environment: task loading, DB management, SQL execution, schema/sample extraction. Enforces SELECT or WITH (CTE) queries only, 5-second timeout, 5-step episode limit, fresh in-memory DB per episode.
- `server/grader.py` — 4-tier reward computation (0.0-1.0): error tiers (0.00-0.15), Soft-F1 partial match (0.20-0.90), Relaxed EX (0.95), Strict EX (1.0). Order-sensitive only when gold SQL has ORDER BY.
- `server/app.py` — FastAPI app factory using `openenv.core.env_server.create_app()`. Exposes `/health`, `/reset`, `/step`, `/state`.
- `client.py` — `BirdText2SQLEnv` WebSocket client extending `EnvClient`. `env.reset()` and `env.step()` return `StepResult` objects (from `openenv.core.client_types`), not raw observations.
- `inference.py` — Baseline agent: iterates all 200 tasks (loaded dynamically from tasks.json), calls LLM, self-corrects using feedback from failed steps.
- `scripts/build_tasks.py` — Builds `data/tasks.json` and `data/schema_linking.json` by merging InsightXpert's `perfect_linking_bird_dev.json` and `perfect_linking_mini_dev_evidence.json` (enriched schema). Selects 200 tasks (85/65/50 by difficulty) across 5 DBs.
- `schema_linking/` — Offline data-generation tool (NOT runtime code). Generates `data/schema_linking.json` which the server loads at startup. Requires the `dev` extra (`pip install -e ".[dev]"`).

**Data:**
- `data/tasks.json` — 200 tasks with gold SQL (runtime-critical, loaded at startup)
- `data/schema_linking.json` — Schema linking data with evidence-enriched schema where available (runtime-critical, loaded at startup; server crashes if missing)
- `data/databases/{db_id}/{db_id}.sqlite` — 5 SQLite databases tracked via **Git LFS** (`.gitattributes` tracks `*.sqlite`)

## Key Constraints

- Only SELECT or WITH (CTE) queries are allowed (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE rejected via regex)
- Max 5 steps per episode; episode ends early on perfect reward (1.0)
- SQL execution timeout: 5 seconds
- Each episode gets a fresh in-memory copy of the database (no cross-episode state leakage)
- **Concurrent sessions supported** (`SUPPORTS_CONCURRENT_SESSIONS = True`, `max_concurrent_envs=64`) — each session gets its own in-memory DB copy
- No test suite exists; validation is done by running `inference.py` against all 200 tasks
- No linting, formatting, or CI tooling is configured

## Important Patterns

- **Dual import pattern:** All `server/` modules use try/except imports to support both `uvicorn server.app:app` (run from project root) and installed-package mode. This is intentional — do not "fix" it.
- **`[START]/[STEP]/[END]` stdout format:** `inference.py` emits structured logs in this format as required by the OpenEnv competition spec. Do not alter this format.
- **`DESIGN.md` / `PLAN.md` / `AUDIT.md`:** Internal planning docs, gitignored. These may or may not exist in the working tree and should not be treated as authoritative.
