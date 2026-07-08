# Changelog

All notable changes to this local proxy are tracked here.

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
