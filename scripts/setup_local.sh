!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$ROOT_DIR"

echo "== EPPN local setup script =="
echo "This script will: install python deps, copy .env.template -> .env (if needed), prompt for keys, start docker-compose (if available), start uvicorn services and agents, and run the integration manager."
echo

# 1) Install Python deps
if command -v pip >/dev/null 2>&1; then
  echo "[1/7] Installing Python dependencies..."
  pip install -r requirements.txt
else
  echo "pip not found on PATH; please install Python/pip and re-run."
fi

# 2) Ensure .env exists
if [ ! -f .env ] && [ -f .env.template ]; then
  echo "[2/7] Copying .env.template -> .env"
  cp .env.template .env
else
  echo "[2/7] .env already exists or .env.template missing"
fi

# helpers to set/overwrite env entries in .env
set_env_var() {
  key="$1"
  prompt="$2"
  default="$3"
  read -r -p "$prompt [$default]: " val
  val="${val:-$default}"
  if grep -qE "^$key=" .env; then
    sed -i -E "s|^($key)=.*|\1=\"$val\"|" .env
  else
    echo "$key=\"$val\"" >> .env
  fi
}

set_env_secret() {
  key="$1"
  prompt="$2"
  read -r -s -p "$prompt: " val
  echo
  if [ -z "$val" ]; then
    echo "No value entered for $key — leaving existing or blank."
    return
  fi
  if grep -qE "^$key=" .env; then
    sed -i -E "s|^($key)=.*|\1=\"$val\"|" .env
  else
    echo "$key=\"$val\"" >> .env
  fi
}

# 3) Prompt for keys (you must create accounts and supply these)
echo "[3/7] Prompting for external keys (AgentVerse, Cudos). Leave empty to keep existing .env values."
set_env_var "AGENTVERSE_API_URL" "AgentVerse API URL" "https://api.agentverse.ai"
set_env_secret "AGENTVERSE_API_KEY" "AgentVerse API key (secret)"
set_env_secret "CUDOS_PRIVATE_KEY" "Cudos private key (secret)"
set_env_var "CUDOS_WALLET_ADDRESS" "Cudos wallet address" ""

# Optional commonly required entries (defaults from DEPLOYMENT_GUIDE)
set_env_var "CUDOS_RPC_URL" "Cudos RPC URL" "https://rpc.cudos.org"
set_env_var "COGNITIVE_CORE_URL" "Cognitive core URL" "http://localhost:8001"
set_env_var "REDIS_URL" "Redis URL" "redis://localhost:6379"

# Create logs dir
mkdir -p logs

# 4) Start docker-compose if available
echo "[4/7] Starting docker-compose services (if docker available)..."
if command -v docker >/dev/null 2>&1; then
  if docker compose version >/dev/null 2>&1; then
    docker compose -f deployment/docker-compose.yml up -d || true
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f deployment/docker-compose.yml up -d || true
  else
    echo "No docker compose command found."
  fi
else
  echo "Docker not available; skipping docker-compose start."
fi

# 5) Start cognitive core and frontend (uvicorn) — use nohup to background and persist logs
echo "[5/7] Starting cognitive core and frontend (uvicorn)..."
if [ -f "cognitive_core/api/cognitive_api.py" ]; then
  nohup python -m uvicorn cognitive_core.api.cognitive_api:app --reload --port 8001 > logs/cognitive_core.log 2>&1 &
  echo "  cognitive_core -> logs/cognitive_core.log"
fi

if [ -d "frontend" ] && [ -f "frontend/app.py" ]; then
  (cd frontend && nohup python -m uvicorn app:app --reload --port 8000 > ../logs/frontend.log 2>&1 &)
  echo "  frontend -> logs/frontend.log"
fi

# 6) Start agents
echo "[6/7] Starting agent processes (background)..."
AGENTS=(
  "agents/librarian/main.py"
  "agents/interpreter/main.py"
  "agents/summarizer/main.py"
  "agents/ethical_analyst/main.py"
  "agents/communicator/main.py"
)
for a in "${AGENTS[@]}"; do
  if [ -f "$a" ]; then
    name=$(basename "$(dirname "$a")")
    nohup python "$a" > "logs/${name}.log" 2>&1 &
    echo "  started $a -> logs/${name}.log"
  fi
done

# 7) Run integration manager to register agents (if present)
echo "[7/7] Running integration manager (register agents)..."
if [ -f "integration_manager.py" ]; then
  nohup python integration_manager.py > logs/integration_manager.log 2>&1 &
  echo "  integration_manager -> logs/integration_manager.log"
else
  echo "integration_manager.py not found — skip."
fi

echo
echo "Setup script finished. Tail logs with: tail -f logs/*.log"
echo "You still need to create external accounts (AgentVerse, Cudos, etc.) and provide keys when prompted or paste them into .env."
# ...existing code..