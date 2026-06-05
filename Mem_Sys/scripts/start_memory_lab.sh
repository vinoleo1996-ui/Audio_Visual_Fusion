#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/output/memory"
mkdir -p "$LOG_DIR"

LLAMA_BIN="${LLAMA_SERVER_BIN:-/home/nvidia/Documents/ProactiveOS_0.1/perception/face_lab/third_party/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MEMORY_MODEL_PATH:-/home/nvidia/Documents/Models/Qwen3.5-2B-Q4_K_M/Qwen3.5-2B-Q4_K_M.gguf}"
MODEL_NAME="${MEMORY_MODEL_NAME:-qwen3.5-2b-q4_k_m}"
LLM_HOST="${MEMORY_LLM_HOST:-127.0.0.1}"
UI_HOST="${MEMORY_UI_HOST:-0.0.0.0}"
LLM_START_PORT="${MEMORY_LLM_PORT:-8081}"
UI_START_PORT="${MEMORY_UI_PORT:-8095}"
CTX_SIZE="${MEMORY_CTX_SIZE:-4096}"
THREADS="${MEMORY_THREADS:-6}"
GPU_LAYERS="${MEMORY_GPU_LAYERS:-999}"
TIMEOUT_S="${MEMORY_TIMEOUT_S:-60}"
OPEN_BROWSER=0

for arg in "$@"; do
  case "$arg" in
    --open)
      OPEN_BROWSER=1
      ;;
    --help|-h)
      cat <<EOF
Usage: bash scripts/start_memory_lab.sh [--open]

Environment overrides:
  LLAMA_SERVER_BIN     llama-server binary path
  MEMORY_MODEL_PATH    GGUF model path
  MEMORY_MODEL_NAME    model alias/name
  MEMORY_LLM_PORT      preferred model port, default 8081
  MEMORY_UI_PORT       preferred UI port, default 8095
  MEMORY_UI_HOST       UI bind host, default 0.0.0.0
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

port_free() {
  local host="$1"
  local port="$2"
  python3 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

find_free_port() {
  local host="$1"
  local port="$2"
  while ! port_free "$host" "$port"; do
    port=$((port + 1))
  done
  echo "$port"
}

model_server_ok() {
  local port="$1"
  python3 - "$port" "$MODEL_NAME" <<'PY'
import json
import sys
import urllib.request

port = int(sys.argv[1])
model = sys.argv[2]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5) as response:
        body = json.loads(response.read().decode("utf-8"))
except Exception:
    sys.exit(1)

text = json.dumps(body)
if model in text:
    sys.exit(0)
sys.exit(1)
PY
}

wait_http() {
  local url="$1"
  local seconds="$2"
  local deadline=$((SECONDS + seconds))
  while (( SECONDS < deadline )); do
    if python3 - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    if response.status >= 500:
        raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

require_file "$LLAMA_BIN" "llama-server"
require_file "$MODEL_PATH" "Qwen model"

LLM_PORT="$LLM_START_PORT"
LLM_REUSED=0
if ! port_free "$LLM_HOST" "$LLM_PORT"; then
  if model_server_ok "$LLM_PORT"; then
    LLM_REUSED=1
  else
    LLM_PORT="$(find_free_port "$LLM_HOST" "$((LLM_PORT + 1))")"
  fi
fi

if [[ "$LLM_REUSED" == "0" ]]; then
  LLAMA_LOG="$LOG_DIR/llama_server_${LLM_PORT}.log"
  LLAMA_PID="$LOG_DIR/llama_server_${LLM_PORT}.pid"
  setsid "$LLAMA_BIN" \
    -m "$MODEL_PATH" \
    --alias "$MODEL_NAME" \
    --host "$LLM_HOST" \
    --port "$LLM_PORT" \
    --ctx-size "$CTX_SIZE" \
    --threads "$THREADS" \
    --gpu-layers "$GPU_LAYERS" \
    --reasoning off \
    > "$LLAMA_LOG" 2>&1 < /dev/null &
  echo "$!" > "$LLAMA_PID"
  if ! wait_http "http://127.0.0.1:${LLM_PORT}/v1/models" 120; then
    echo "llama-server did not become ready. Log: $LLAMA_LOG" >&2
    tail -n 80 "$LLAMA_LOG" >&2 || true
    exit 1
  fi
fi

UI_PORT="$(find_free_port "$UI_HOST" "$UI_START_PORT")"
MEMORY_LOG="$LOG_DIR/memory_service_${UI_PORT}.log"
MEMORY_PID="$LOG_DIR/memory_service_${UI_PORT}.pid"
setsid python3 "$ROOT_DIR/scripts/memory_service.py" \
  --db "$ROOT_DIR/output/memory/memory.sqlite3" \
  serve \
  --host "$UI_HOST" \
  --port "$UI_PORT" \
  --endpoint "http://127.0.0.1:${LLM_PORT}/v1/chat/completions" \
  --model "$MODEL_NAME" \
  --timeout-s "$TIMEOUT_S" \
  --fallback-on-llm-error \
  > "$MEMORY_LOG" 2>&1 < /dev/null &
echo "$!" > "$MEMORY_PID"

if ! wait_http "http://127.0.0.1:${UI_PORT}/memory/config" 20; then
  echo "memory_service did not become ready. Log: $MEMORY_LOG" >&2
  tail -n 80 "$MEMORY_LOG" >&2 || true
  exit 1
fi

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
LOCAL_URL="http://127.0.0.1:${UI_PORT}/memory/ui/"
if [[ -n "${LAN_IP:-}" && "$UI_HOST" == "0.0.0.0" ]]; then
  LAN_URL="http://${LAN_IP}:${UI_PORT}/memory/ui/"
else
  LAN_URL="$LOCAL_URL"
fi

if [[ "$OPEN_BROWSER" == "1" ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$LOCAL_URL" >/dev/null 2>&1 || true
  fi
fi

cat <<EOF
Memory Lab is ready.

UI:
  local: $LOCAL_URL
  lan:   $LAN_URL

Backend:
  memory_service: http://127.0.0.1:${UI_PORT}
  llama_server:   http://127.0.0.1:${LLM_PORT}
  model:          $MODEL_NAME
  model_reused:   $LLM_REUSED

Logs:
  memory: $MEMORY_LOG
  llama:  ${LLAMA_LOG:-reused existing server}
EOF
