#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Pre-Submission Validator
#
# Checks that your submission meets all requirements from the
# India Hackathon pre-submission checklist before you submit.
#
# Usage:
#   ./scripts/validate-submission.sh [ping_url]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#              If omitted, remote checks are skipped.
#
# Examples:
#   ./scripts/validate-submission.sh
#   ./scripts/validate-submission.sh https://nachiketkandari-insightxpert-openenv.hf.space
#

set -uo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

PASS=0
FAIL=0
WARN=0

pass() { ((PASS++)); printf "${GREEN}PASS${NC}  %s\n" "$1"; }
fail() { ((FAIL++)); printf "${RED}FAIL${NC}  %s\n" "$1"; }
warn() { ((WARN++)); printf "${YELLOW}WARN${NC}  %s\n" "$1"; }
info() { printf '%b%s%b\n' "${BOLD}" "---- $1" "${NC}"; }

PING_URL="${1:-}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

printf "\n${BOLD}OpenEnv Pre-Submission Validator${NC}\n"
printf "Repo: %s\n\n" "$REPO_DIR"

# ── 1. Required files ────────────────────────────────────────────────────────
info "Checking required files..."

for f in openenv.yaml Dockerfile inference.py models.py client.py requirements.txt README.md; do
  if [ -f "$REPO_DIR/$f" ]; then
    pass "$f exists"
  else
    fail "$f is MISSING"
  fi
done

if [ -d "$REPO_DIR/server" ]; then
  for f in server/app.py server/bird_environment.py server/grader.py; do
    if [ -f "$REPO_DIR/$f" ]; then
      pass "$f exists"
    else
      fail "$f is MISSING"
    fi
  done
fi

if [ -f "$REPO_DIR/data/tasks.json" ]; then
  pass "data/tasks.json exists"
else
  fail "data/tasks.json is MISSING"
fi

if [ -f "$REPO_DIR/data/schema_linking.json" ]; then
  pass "data/schema_linking.json exists"
else
  fail "data/schema_linking.json is MISSING"
fi

echo ""

# ── 2. openenv.yaml validation ───────────────────────────────────────────────
info "Validating openenv.yaml..."

if [ -f "$REPO_DIR/openenv.yaml" ]; then
  for field in spec_version name type runtime app port; do
    if grep -q "^${field}:" "$REPO_DIR/openenv.yaml"; then
      pass "openenv.yaml has '${field}'"
    else
      fail "openenv.yaml missing '${field}'"
    fi
  done
else
  fail "openenv.yaml not found — cannot validate"
fi

echo ""

# ── 3. Dockerfile validation ─────────────────────────────────────────────────
info "Validating Dockerfile..."

if [ -f "$REPO_DIR/Dockerfile" ]; then
  if grep -q "^FROM" "$REPO_DIR/Dockerfile"; then
    pass "Dockerfile has FROM instruction"
  else
    fail "Dockerfile missing FROM instruction"
  fi

  if grep -q "EXPOSE.*7860" "$REPO_DIR/Dockerfile"; then
    pass "Dockerfile exposes port 7860"
  else
    fail "Dockerfile does not expose port 7860"
  fi

  if grep -q "HEALTHCHECK" "$REPO_DIR/Dockerfile"; then
    pass "Dockerfile has HEALTHCHECK"
  else
    warn "Dockerfile missing HEALTHCHECK (recommended)"
  fi

  if grep -q "CMD" "$REPO_DIR/Dockerfile"; then
    pass "Dockerfile has CMD"
  else
    fail "Dockerfile missing CMD"
  fi
fi

echo ""

# ── 4. inference.py validation ────────────────────────────────────────────────
info "Validating inference.py..."

if [ -f "$REPO_DIR/inference.py" ]; then
  # OpenAI client
  if grep -q "from openai import OpenAI\|import openai" "$REPO_DIR/inference.py"; then
    pass "inference.py uses OpenAI client"
  else
    fail "inference.py does not use OpenAI client (required)"
  fi

  # Environment variables
  for var in API_BASE_URL MODEL_NAME HF_TOKEN; do
    if grep -q "$var" "$REPO_DIR/inference.py"; then
      pass "inference.py references $var"
    else
      fail "inference.py missing $var env var"
    fi
  done

  # [START]/[STEP]/[END] structured logs
  if grep -q '\[START\]' "$REPO_DIR/inference.py"; then
    pass "inference.py emits [START] logs"
  else
    fail "inference.py missing [START] log format"
  fi

  if grep -q '\[STEP\]' "$REPO_DIR/inference.py"; then
    pass "inference.py emits [STEP] logs"
  else
    fail "inference.py missing [STEP] log format"
  fi

  if grep -q '\[END\]' "$REPO_DIR/inference.py"; then
    pass "inference.py emits [END] logs"
  else
    fail "inference.py missing [END] log format"
  fi

  # Check log format fields
  if grep -q 'task=.*env=.*model=' "$REPO_DIR/inference.py"; then
    pass "[START] has required fields (task, env, model)"
  else
    warn "[START] may be missing required fields (task=, env=, model=)"
  fi

  if grep -q 'step=' "$REPO_DIR/inference.py" && grep -q 'action=' "$REPO_DIR/inference.py" && grep -q 'reward=' "$REPO_DIR/inference.py" && grep -q 'done=' "$REPO_DIR/inference.py"; then
    pass "[STEP] has required fields (step, action, reward, done)"
  else
    warn "[STEP] may be missing required fields"
  fi

  if grep -q 'success=.*steps=.*rewards=' "$REPO_DIR/inference.py"; then
    pass "[END] has required fields (success, steps, rewards)"
  else
    warn "[END] may be missing required fields"
  fi
fi

echo ""

# ── 5. Models validation ─────────────────────────────────────────────────────
info "Validating Pydantic models..."

if [ -f "$REPO_DIR/models.py" ]; then
  if grep -q "class.*Action" "$REPO_DIR/models.py"; then
    pass "models.py defines Action model"
  else
    fail "models.py missing Action model"
  fi

  if grep -q "class.*Observation" "$REPO_DIR/models.py"; then
    pass "models.py defines Observation model"
  else
    fail "models.py missing Observation model"
  fi

  if grep -q "class.*State" "$REPO_DIR/models.py"; then
    pass "models.py defines State model"
  else
    fail "models.py missing State model"
  fi
fi

echo ""

# ── 6. Task count ─────────────────────────────────────────────────────────────
info "Checking task count..."

if [ -f "$REPO_DIR/data/tasks.json" ]; then
  task_count=$(python3 -c "import json; print(len(json.load(open('$REPO_DIR/data/tasks.json'))))" 2>/dev/null)
  if [ -n "$task_count" ] && [ "$task_count" -ge 3 ]; then
    pass "$task_count tasks found (minimum 3 required)"
  else
    fail "Less than 3 tasks found (got: ${task_count:-0})"
  fi
else
  fail "Cannot count tasks — data/tasks.json missing"
fi

echo ""

# ── 7. Reward range validation ────────────────────────────────────────────────
info "Checking reward function bounds..."

if [ -f "$REPO_DIR/server/grader.py" ]; then
  # Check that all return values are in 0.0-1.0
  rewards=$(grep -oE 'return [0-9]+\.[0-9]+' "$REPO_DIR/server/grader.py" | awk '{print $2}')
  all_valid=true
  for r in $rewards; do
    if python3 -c "import sys; sys.exit(0 if 0.0 <= float('$r') <= 1.0 else 1)" 2>/dev/null; then
      :
    else
      fail "Reward value $r is outside [0.0, 1.0]"
      all_valid=false
    fi
  done
  if $all_valid; then
    pass "All reward return values are in [0.0, 1.0]"
  fi
fi

echo ""

# ── 8. Database files ─────────────────────────────────────────────────────────
info "Checking database files..."

db_count=0
for db in california_schools debit_card_specializing financial formula_1 toxicology; do
  db_file="$REPO_DIR/data/databases/$db/$db.sqlite"
  if [ -f "$db_file" ]; then
    size=$(wc -c < "$db_file" | tr -d ' ')
    if [ "$size" -gt 100 ]; then
      pass "$db.sqlite exists (${size} bytes)"
      ((db_count++))
    else
      warn "$db.sqlite exists but seems too small (${size} bytes) — Git LFS pointer?"
    fi
  else
    fail "$db.sqlite is MISSING"
  fi
done

echo ""

# ── 9. Python import check ───────────────────────────────────────────────────
info "Checking Python imports..."

if python3 -c "from openenv.core.env_server import create_app" 2>/dev/null; then
  pass "openenv-core is importable"
else
  warn "openenv-core not installed locally (ok if Docker builds)"
fi

if python3 -c "from openai import OpenAI" 2>/dev/null; then
  pass "openai package is importable"
else
  warn "openai package not installed locally"
fi

if python3 -c "import gradio" 2>/dev/null; then
  pass "gradio is importable"
else
  warn "gradio not installed locally"
fi

echo ""

# ── 10. Remote checks (if ping_url provided) ─────────────────────────────────
if [ -n "$PING_URL" ]; then
  info "Remote checks against $PING_URL ..."

  # Strip trailing slash
  PING_URL="${PING_URL%/}"

  # Health check
  http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$PING_URL/health" 2>/dev/null)
  if [ "$http_code" = "200" ]; then
    pass "/health returns 200"
  else
    fail "/health returned $http_code (expected 200)"
  fi

  # Reset endpoint
  reset_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 -X POST "$PING_URL/reset" 2>/dev/null)
  if [ "$reset_code" = "200" ]; then
    pass "/reset returns 200"
  else
    fail "/reset returned $reset_code (expected 200)"
  fi

  # State endpoint
  state_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$PING_URL/state" 2>/dev/null)
  if [ "$state_code" = "200" ]; then
    pass "/state returns 200"
  else
    fail "/state returned $state_code (expected 200)"
  fi
else
  info "No ping URL provided — skipping remote checks"
  warn "Run with your HF Space URL to test remote endpoints"
fi

echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
printf "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
printf "${GREEN}PASS: %d${NC}  ${RED}FAIL: %d${NC}  ${YELLOW}WARN: %d${NC}\n" "$PASS" "$FAIL" "$WARN"

if [ "$FAIL" -gt 0 ]; then
  printf "\n${RED}${BOLD}SUBMISSION NOT READY — fix the failures above.${NC}\n\n"
  exit 1
elif [ "$WARN" -gt 0 ]; then
  printf "\n${YELLOW}${BOLD}SUBMISSION LOOKS OK — review warnings above.${NC}\n\n"
  exit 0
else
  printf "\n${GREEN}${BOLD}ALL CHECKS PASSED — ready to submit!${NC}\n\n"
  exit 0
fi
