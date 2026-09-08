#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HOST="${LOCAL_PROXY_HOST:-127.0.0.1}"
PORT="${LOCAL_PROXY_PORT:-8787}"
exec python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT"
