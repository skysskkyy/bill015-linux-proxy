# BILL-015 Linux Codex Desktop Proxy

Local OpenAI-compatible proxy for **Codex Desktop on Linux**. It keeps the BILL-015 bridge: force `emit_value`, abort after wrapper arguments complete, replay native Responses SSE to the client.

Version: see `VERSION` (0.5.0).

## What it does

```
Codex Desktop  →  127.0.0.1:8787/v1/responses
               →  ingest Lite additional_tools + tool catalog
               →  pack context by token priority
               →  upstream packy: one emit_value tool, many calls allowed
               →  abort before response.completed
               →  replay function_call / custom_tool_call / tool_search_call
```

## Linux start

```bash
cd /path/to/bill015_local_proxy
python3 -m pip install -r requirements.txt
./scripts/start_local_proxy.sh
```

Health:

```bash
curl http://127.0.0.1:8787/healthz
curl http://127.0.0.1:8787/v1/models
```

Config is `config.local.json` in the repo, or `$BILL015_CONFIG_PATH`, or `$XDG_CONFIG_HOME/bill015/config.json`.

Point Codex Desktop at the proxy (`~/.codex/config.toml`):

```toml
[model_providers.bill015]
name = "bill015"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
```

Do not commit `config.local.json`. It holds API keys.

## Modes

Codex summary-compaction requests sent to `/v1/responses` are forwarded to upstream `/v1/responses` with their original JSON body, using the configured upstream credentials. Their native SSE/JSON responses and HTTP errors are returned directly. This compaction-only exception bypasses the `emit_value` bridge and its passthrough guard; `dry-run` still makes no upstream call. Explicit `/v1/responses/compact` requests keep using the dedicated compact endpoint.

- `exploit` — BILL-015 emit_value bridge (default)
- `verify` — same, plus optional `/api/user/self` delta if cookie is set
- `dry-run` — no upstream call
- `normal` — blocked by default (`block_normal_mode`)

## Tests

```bash
python3 -m pip install -r requirements-dev.txt
python3 scripts/selftest.py
python3 -m pytest -q
python3 -m ruff check app scripts tests
```

## Layout

```
app/config     nested settings
app/ingest     Codex Desktop / Lite request → Turn
app/context    token-budget packer
app/bridge     emit_value schema / prompt / parse
app/upstream   multi-emit_value collector + abort
app/replay     Desktop-faithful SSE
app/ops        capacity, audit, health
```
