# Changelog

All notable changes to this local proxy are tracked here.

## [0.3.2] - 2026-07-12

### Added
- Upgraded the BILL-015 `emit_value.tool_calls` contract from a generic text-only tool directory to a per-turn, dynamically generated typed `oneOf` schema.
- Each advertised Codex/local/MCP tool now gets exact `type` / `namespace` / `name` enums and native parameter schemas under `arguments`, while preserving legacy JSON-string parsing compatibility.

### Tests
- Added regression coverage proving typed tool schemas are embedded in the upstream payload and structured object arguments still resolve into native Codex tool calls.

## [0.3.1] - 2026-07-12

### Changed
- Added guarded upstream retries for pre-stream `HTTP 5xx`, `do_request_failed`, and transient network failures.
- Retries are skipped after any upstream SSE event is observed to preserve the early-abort/BILL-015 semantics.

### Tests
- Added coverage for pre-stream retry behavior and sanitized stream failure handling.

## [0.3.0] - 2026-07-10

### Changed
- Split Chat Completions compatibility into `app/chat_events.py` so `app/response_events.py` focuses on Responses event synthesis.
- Split upstream timeout, auth headers, passthrough payload preparation, and normal forwarding into `app/upstream_client.py`.
- Added `app/config_schema.py` to validate local config shape and expose non-fatal config warnings through `/healthz`.
- Preserved compatibility facade exports for older imports while moving implementation into focused modules.

### Tests
- Added config schema regression coverage for unknown-field warnings.
- Verified Codex CLI can use this local proxy as its Responses provider at `http://127.0.0.1:8787/v1` and successfully run local-tool tasks through it.

## [0.2.2] - 2026-07-10

### Fixed
- Preserve upstream non-2xx status codes for non-stream normal Responses forwarding.
- Mark repaired malformed `emit_value` arguments as both malformed and repaired in audit state.
- Expand audit redaction for common token, secret, cookie, and password key names.

### Changed
- Safer default timeout values for upstream total, idle, and `arguments.done` waits.
- Default audit behavior no longer stores assistant answers unless explicitly enabled.
- Aligned JSON/YAML example configs with the current runtime behavior and added a tool-bridge strict-mode switch.
- Removed unused upstream media-shrinking helpers from the forwarding path.

### Tests
- Added focused regression tests for argument repair flags, upstream error status preservation, secret-key redaction, and strict tool-bridge behavior.

## [0.2.1] - 2026-07-08

### Added
- Auto-expansion for browser/MCP-related `tool_search` calls to expose Playwright, Chrome, node_repl, jshook, and computer-use tool families more reliably.
- Version metadata is now read from `VERSION` and exposed through FastAPI root/health responses.

### Changed
- Strengthened upstream bridge prompt guidance for broad local tool discovery queries.
- `config.local.example.json` now defaults to `exploit` mode to match the intended BILL-015 bridge path.

### Tests
- Extended offline selftest coverage for MCP namespace parsing and deferred browser/tool discovery expansion.

## [0.2.0] - 2026-07-08

### Added
- Native-style Responses event lifecycle synthesis for message, function call, custom tool call, tool search, failed, incomplete, and cancelled streams.
- Synthetic Responses/Chat usage estimation with text, tool schema, tool history, image, metadata, cached-token, and compaction handling.
- Codex tool-loop feedback parsing for successful, failed, repeated, pending, and deferred local tool calls.
- Compaction request detection and `/v1/responses/compact` compatibility.
- Native request replay and usage calibration helper scripts.

### Changed
- Model mapping now preserves requested model IDs by default while still supporting aliases.
- `web_search` bridge requests are rewritten to local `tool_search` instead of server-side web search.

## [0.1.0] - 2026-07-08

### Added
- BILL-015 local Codex proxy baseline.
- Responses API compatibility for Codex Desktop.
- `emit_value` bridge for final answers and local tool calls.
- Native-style `function_call`, `custom_tool_call`, `tool_search_call` synthesis.
- Parallel local tool-call event synthesis.
- Local config loading via `config.local.json` with `config.local.example.json` template.
- Synthetic `usage` metadata for Codex context display.
- Compaction request detection and local compaction prompt path.
- Deferred tool catalog extraction from `tool_search_output` history.
- `web_search` rewrite to local `tool_search` for browser/node_repl/shell based fetching.
- `/v1/responses/compact` compatibility route.
- Tool-loop optimization implementation plan in `TOOL_LOOP_OPTIMIZATION_PLAN.md`.

### Security
- Added `.gitignore` rules to keep local secrets, runtime logs, PID files, pycache, and captured native request evidence out of Git.
