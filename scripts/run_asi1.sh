#!/usr/bin/env bash
# Helper to run the ASI1.ai chat script.
# - Loads .env if present (simple key=val parser)
# - Otherwise prompts interactively for the ASI_ONE_API_KEY
# - Passes any args through to the python script

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/scripts/asi1_chat.py"
ENV_FILE="$ROOT_DIR/.env"

# Load .env if it exists (simple parsing: KEY=VALUE, ignore comments)
if [[ -f "$ENV_FILE" ]]; then
  echo "Loading environment from $ENV_FILE"
  # shellcheck disable=SC1090
  set -a
  # Use a subshell to avoid exporting other content accidentally
  (cat "$ENV_FILE" | sed -e 's/^\s*#.*//' -e '/^\s*$/d' > /tmp/asi1_env_sh)
  # shellcheck disable=SC1090
  source /tmp/asi1_env_sh
  set +a
  rm -f /tmp/asi1_env_sh
fi

# Prompt for key if not set
if [[ -z "${ASI_ONE_API_KEY:-}" ]]; then
  read -r -s -p "Enter ASI_ONE_API_KEY: " _KEY
  echo
  if [[ -z "${_KEY}" ]]; then
    echo "No key provided; aborting."
    exit 1
  fi
  export ASI_ONE_API_KEY="${_KEY}"
fi

# Run the python helper passing any CLI args
python3 "$SCRIPT" "$@"
