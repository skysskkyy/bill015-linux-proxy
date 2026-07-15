# Changelog

## [0.4.7] - 2026-07-15

### Changed
- Removed the prompt pressure that made the upstream model write a progress/commentary sentence for every tool call; routine tool calls now use an empty `answer` in the examples while still allowing genuinely useful pre-tool text to be displayed.
- Raised default local request concurrency from 2 to 4 and lowered the exact-tokenization ceiling from 500k to 120k chars so multiple long-context tasks spend less time contending on local tokenization.

### Fixed
- Optimized local context compaction by avoiding full payload token re-estimation after every removed history item; long histories now use item-level token deltas inside the removal loop and only re-check full payloads at key boundaries.

### Verified
- Smoke-tested `https://anpin.ai` with the supplied key using temporary config only: `gpt-5.5` and GPT-5.6 Responses-Lite-shaped requests both returned through the strict-zero bridge, with GPT-5.6 still de-Lited upstream.

## [0.4.6] - 2026-07-15

### Fixed
- Fail-closed GPT-5.6 strict-zero bridge transport by ingesting Codex Responses Lite `additional_tools` locally but de-Liting the upstream BILL-015 request back to top-level `tools`/`instructions` with only `emit_value`.
- Strip `x-openai-internal-codex-responses-lite`, `responses_lite`, and `use_responses_lite` from strict-zero upstream headers/body metadata so client Lite markers cannot re-enable the charged upstream route.
- Keep non-strict/passthrough Responses Lite behavior unchanged for native Codex compatibility.
- Added audit fields for client-vs-upstream Responses Lite state.

### Tests
- Added regressions for GPT-5.6 strict-zero de-Lite payload/header behavior while preserving CLI tool discovery and multi-agent tool schemas.
## [0.4.5] - 2026-07-15

### Fixed
- Corrected the CLI/Desktop tool-discovery boundary: native `tool_search_call` remains available in Codex CLI, while the unsupported app-server `DynamicToolCall` path stays separate.
- Rebuilt Responses Lite bridge requests in Codex-native shape with `additional_tools` and developer instructions in `input`, no top-level `tools`/`instructions`, `parallel_tool_calls=false`, and `reasoning.context=all_turns`.
- Preserved Codex session, thread, window, originator, turn-state, and subagent headers without forwarding client authorization credentials.
- Added lossless discovery and invocation coverage for multi-agent v1 namespace tools and v2 plain function tools.

### Tests
- Added regressions for GPT-5.6 Responses Lite CLI browser/tool discovery, multi-agent v1/v2 calls, Codex header forwarding, and credential replacement.

## [0.4.4] - 2026-07-15

### Fixed
- Aligned Codex Responses Lite handling with upstream Codex request shape by detecting `additional_tools`, preserving local tool discovery, and setting `reasoning.context=all_turns`.
- Propagated Codex request controls through bridge and passthrough paths: `prompt_cache_options`, `service_tier`, and explicit `truncation`.
- Added the Codex Responses Lite internal header for passthrough and bridge upstream calls.

### Tests
- Added regressions for Responses Lite tool catalog extraction, passthrough header propagation, and request-control preservation.

## [0.4.3] - 2026-07-13

### Fixed
- Added bounded recovery for mid-stream `ReadTimeout` and local `response.function_call_arguments.done` wait timeouts.
- Stream recovery reinforces the final-action instruction, compacts near-limit payloads, and retries without surfacing Codex `stream disconnected before completion` errors.
- Raised default upstream/idle/args-done wait budgets to 900s for long reasoning/tool-selection turns.
- Aligned local compaction summaries with Codex history shape (`user`/`input_text`) and added last-resort truncation for oversized latest user turns.

### Tests
- Added regressions for args-done timeout recovery, mid-stream ReadTimeout recovery, Codex-shaped compaction summaries, and oversized latest-user clipping.

## [0.4.2] - 2026-07-13

### Fixed
- Added Codex-style pre-turn context compaction at 90% of the configured context window.
- Added mid-turn recovery for `context_length_exceeded` by removing oldest history first while preserving the latest user request and latest tool batch.
- Replaced the synthetic empty-stream success message with bounded reasoning-only retries and a structured failure after exhaustion.

### Tests
- Added regressions for context-error detection, latest-turn preservation, and reasoning-only retry prompting.
## [0.4.1] - 2026-07-13

### Changed
- Rotate to the next API key when an upstream HTTP or SSE error contains `If this seems wrong, try rephrasing your request`.
- Preserve per-request key de-duplication, exhaustion protection, strict-zero behavior, and reason-specific audit entries.

### Tests
- Added matcher, HTTP integration, and SSE integration regressions for the rephrase-request error.
## [0.4.0] - 2026-07-13

### Added
- Added an ordered, de-duplicated, concurrency-safe API key pool with legacy single-key compatibility.
- Added targeted key rotation for upstream HTTP or SSE error objects containing `sequence_number: 113`, including strict-zero mode.
- Added non-secret key-pool health metadata and per-request key-switch audit fields.

### Tests
- Added regressions for HTTP and SSE rotation, non-target errors, key exhaustion, and normal JSON forwarding.
All notable changes to this local proxy are tracked here.

## [0.3.9] - 2026-07-13

### Fixed
- Replaced image/screenshot inputs with an upstream-visible local-proxy notice instead of returning strict-zero `422` or forwarding billable image payloads.
- Sanitized image data URLs, image MIME parts, and `image_url`/screenshot fields before normalization and upstream payload construction.
- Completed client streams safely when upstream closes without `response.function_call_arguments.done`, extracting completed message text when present and otherwise returning a local retry note instead of disconnecting.

### Tests
- Added regressions for image sanitization and no-arguments upstream stream completion.

## [0.3.8] - 2026-07-13

### Fixed
- Restored `emit_value` as the default bridge strategy for strict-zero operation after native-tool-first was observed to create upstream usage records.
- Added a fail-closed guard: when `bill015.strict_zero=true`, any configured `native_tool_first` strategy is downgraded to `emit_value`.
- Kept `native_tool_first` available only as an explicit experimental mode when `strict_zero=false`.

## [0.3.7] - 2026-07-13

### Changed
- Defaulted `bill015.bridge_strategy` to `native_tool_first`, exposing the current Codex native tools directly upstream instead of forcing every action through the synthetic `emit_value.tool_calls` wrapper.
- Added `submit_final_answer` as the final-answer tool so direct answers still end at a function-call arguments boundary.
- Kept `emit_value` as an explicit fallback via `bill015.bridge_strategy="emit_value"` and retained it for compaction requests.

### Fixed
- Parsed direct native function calls, custom tool input, `tool_search_call`/`web_search_call`/`computer_call` output items, and legacy nameless `emit_value` arguments from the abort boundary.
- Updated dry-run/selftest/config docs for the new native-tool-first default.

## [0.3.6] - 2026-07-12

### Changed
- Raised default normal `bill015.max_output_tokens` from 2048 to 8192 and `bill015.max_answer_chars` from 16384 to 65536 for long answers, patches, and JSON function arguments.
- Added independent `bill015.compaction_max_output_tokens`, defaulting to 8192, so compaction no longer shares the normal output cap implicitly.
- Raised default wait budgets: upstream total timeout 300s, `args_done_timeout_ms` 300000, and upstream idle timeout 180000.
- Defaulted `upstream_retries` to 0 to avoid multiple real upstream attempts under strict-zero usage.

### Tests
- Added regression coverage for default budget values and compaction-specific output capping.

## [0.3.5] - 2026-07-12

### Changed
- Reworked native context assembly around an estimated token budget instead of fixed character truncation: reserve output room first, then prioritize current request, latest tool batch/state, and tail-prioritized related history.
- Hoisted the latest user request without duplicating it inside the replay transcript.
- Preserved the complete latest parallel tool-output batch instead of slicing `latest_outputs` to the last three records.
- Tool feedback now reports batch/call IDs and head/tail output snippets with exit/error metadata under a token budget.

### Tests
- Added coverage for latest-user de-duplication, token-budgeted transcript priority, and parallel batches larger than three outputs.

## [0.3.4] - 2026-07-12

### Changed
- Default `tool_bridge.allow_unknown_tools` is now `false`, so model-requested tools not present in the current Codex tool registry are dropped fail-closed.
- When no local tools are registered and unknown tools are disabled, the upstream `emit_value` schema now restricts `mode` to `answer` and `tool_calls` to an empty array instead of exposing a generic fallback tool shape.
- Updated local/example configs to prefer `tool_search` discovery over hallucinated direct tool calls.

### Tests
- Added regressions for no-config default behavior and no-registry strict schema generation.

## [0.3.3] - 2026-07-12

### Fixed
- Added `bill015.strict_zero` fail-closed mode, enabled by default, to block `auto-passthrough` and `normal` forwarding paths that can create billable/non-aborted upstream calls on unexpected models.
- In strict-zero mode, upstream retries are disabled so a transient failure cannot accidentally create a second upstream generation attempt.
- Fixed tool-followup prompt construction so the latest user request remains pinned before older transcript content even when recent tool output is injected.

### Tests
- Added regressions for strict-zero passthrough blocking and latest-user-request priority after tool feedback.

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
