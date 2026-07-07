# Changelog

All notable changes to this local proxy are tracked here.

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
