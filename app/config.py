from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal

from .config_schema import validate_local_config

Mode = Literal["exploit", "verify", "normal", "dry-run"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.local.json"
CONFIG_WARNINGS: list[str] = []


def _load_local_config() -> dict[str, Any]:
    global CONFIG_WARNINGS
    CONFIG_WARNINGS = []
    # Default behavior: read local project config. An env override is kept only for tests/advanced use.
    raw_path = os.getenv("BILL015_CONFIG_PATH", "")
    path = Path(raw_path) if raw_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            CONFIG_WARNINGS = ["<root>: config JSON must be an object"]
            return {}
        CONFIG_WARNINGS = validate_local_config(obj)
        return obj
    except Exception as e:
        CONFIG_WARNINGS = [f"<root>: failed to read config: {type(e).__name__}: {e}"]
        return {}


LOCAL_CONFIG = _load_local_config()


def _cfg(path: str, default: Any = None) -> Any:
    cur: Any = LOCAL_CONFIG
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _env_str(name: str, cfg_path: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is not None:
        return raw
    val = _cfg(cfg_path, default)
    return default if val is None else str(val)


def _env_bool(name: str, cfg_path: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        raw = _cfg(cfg_path, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, cfg_path: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        raw = _cfg(cfg_path, default)
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, cfg_path: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        raw = _cfg(cfg_path, default)
    try:
        return float(raw)
    except Exception:
        return default


def _cfg_dict(path: str, default: Dict[str, str]) -> Dict[str, str]:
    val = _cfg(path, default)
    if isinstance(val, dict):
        return {str(k): str(v) for k, v in val.items()}
    return dict(default)


def _cfg_list(path: str, default: list[str]) -> list[str]:
    val = _cfg(path, default)
    if isinstance(val, list):
        return [str(v) for v in val if str(v).strip()]
    if isinstance(val, str) and val.strip():
        return [part.strip() for part in val.split(",") if part.strip()]
    return list(default)


def _legacy_strict_zero_default() -> bool:
    """Compatibility fallback for configs created before 0.4.11."""
    return _env_bool("BILL015_STRICT_ZERO", "bill015.strict_zero", True)


@dataclass
class Settings:
    config_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)
    host: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_HOST", "server.host", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_PORT", "server.port", 8787))
    cors_allow_origins: list[str] = field(default_factory=lambda: _cfg_list("server.cors_allow_origins", []))
    cors_allow_origin_regex: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_CORS_ALLOW_ORIGIN_REGEX", "server.cors_allow_origin_regex", r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"))
    mode: Mode = field(default_factory=lambda: _env_str("LOCAL_PROXY_MODE", "mode", "exploit").strip().lower())  # type: ignore[assignment]
    upstream_base_url: str = field(default_factory=lambda: _env_str("PACKY_BASE_URL", "upstream.base_url", "https://packyapi.com").rstrip("/"))
    upstream_api_key_env: str = field(default_factory=lambda: _env_str("PACKY_API_KEY_ENV", "upstream.api_key_env", "PACKY_API_KEY"))
    upstream_api_key_file_value: str = field(default_factory=lambda: _env_str("", "upstream.api_key", ""))
    upstream_api_keys_file_value: list[str] = field(default_factory=lambda: _cfg_list("upstream.api_keys", []))
    default_model: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_MODEL", "model.default", "gpt-5.5"))
    model_aliases: Dict[str, str] = field(default_factory=lambda: _cfg_dict("model.aliases", {"codex-gpt55": "gpt-5.5", "codex-gpt54": "gpt-5.4"}))
    force_default_model: bool = field(default_factory=lambda: _env_bool("LOCAL_PROXY_FORCE_DEFAULT_MODEL", "model.force_default", False))
    bridge_strategy: str = field(default_factory=lambda: _env_str("BILL015_BRIDGE_STRATEGY", "bill015.bridge_strategy", "emit_value"))
    function_name: str = field(default_factory=lambda: _env_str("BILL015_FUNCTION_NAME", "bill015.function_name", "emit_value"))
    final_answer_tool_name: str = field(default_factory=lambda: _env_str("BILL015_FINAL_ANSWER_TOOL_NAME", "bill015.final_answer_tool_name", "submit_final_answer"))
    native_tool_choice: str = field(default_factory=lambda: _env_str("BILL015_NATIVE_TOOL_CHOICE", "bill015.native_tool_choice", "required"))
    native_parallel_tool_calls: bool = field(default_factory=lambda: _env_bool("BILL015_NATIVE_PARALLEL_TOOL_CALLS", "bill015.native_parallel_tool_calls", False))
    answer_field: str = field(default_factory=lambda: _env_str("BILL015_ANSWER_FIELD", "bill015.answer_field", "answer"))
    max_output_tokens: int = field(default_factory=lambda: _env_int("BILL015_MAX_OUTPUT_TOKENS", "bill015.max_output_tokens", 8192))
    compaction_max_output_tokens: int = field(default_factory=lambda: _env_int("BILL015_COMPACTION_MAX_OUTPUT_TOKENS", "bill015.compaction_max_output_tokens", 8192))
    strict_zero: bool = field(default_factory=lambda: _env_bool("BILL015_STRICT_ZERO", "bill015.strict_zero", True))
    force_emit_value: bool = field(
        default_factory=lambda: _env_bool("BILL015_FORCE_EMIT_VALUE", "bill015.force_emit_value", _legacy_strict_zero_default())
    )
    block_passthrough: bool = field(
        default_factory=lambda: _env_bool("BILL015_BLOCK_PASSTHROUGH", "bill015.block_passthrough", _legacy_strict_zero_default())
    )
    block_normal_mode: bool = field(
        default_factory=lambda: _env_bool("BILL015_BLOCK_NORMAL_MODE", "bill015.block_normal_mode", _legacy_strict_zero_default())
    )
    reasoning_effort: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_REASONING_EFFORT", "reasoning.effort", ""))
    reasoning_summary: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_REASONING_SUMMARY", "reasoning.summary", ""))
    max_answer_chars: int = field(default_factory=lambda: _env_int("BILL015_MAX_ANSWER_CHARS", "bill015.max_answer_chars", 65536))
    max_request_bytes: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_MAX_REQUEST_BYTES", "limits.max_request_bytes", 1048576))
    max_concurrency: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_MAX_CONCURRENCY", "limits.max_concurrency", 4))
    # 0 means disabled/infinite. Positive values are enforced by upstream.py.
    upstream_timeout_seconds: float = field(default_factory=lambda: _env_float("PACKY_TIMEOUT_SECONDS", "upstream.timeout_seconds", 900.0))
    upstream_retries: int = field(default_factory=lambda: _env_int("BILL015_UPSTREAM_RETRIES", "limits.upstream_retries", 2))
    upstream_retry_backoff_ms: int = field(default_factory=lambda: _env_int("BILL015_UPSTREAM_RETRY_BACKOFF_MS", "limits.upstream_retry_backoff_ms", 700))
    context_window_tokens: int = field(default_factory=lambda: _env_int("BILL015_CONTEXT_WINDOW_TOKENS", "limits.context_window_tokens", 128000))
    auto_compact_percent: int = field(default_factory=lambda: _env_int("BILL015_AUTO_COMPACT_PERCENT", "limits.auto_compact_percent", 90))
    compact_target_percent: int = field(default_factory=lambda: _env_int("BILL015_COMPACT_TARGET_PERCENT", "limits.compact_target_percent", 75))
    context_recovery_retries: int = field(default_factory=lambda: _env_int("BILL015_CONTEXT_RECOVERY_RETRIES", "limits.context_recovery_retries", 3))
    empty_stream_retries: int = field(default_factory=lambda: _env_int("BILL015_EMPTY_STREAM_RETRIES", "limits.empty_stream_retries", 2))
    stream_recovery_retries: int = field(default_factory=lambda: _env_int("BILL015_STREAM_RECOVERY_RETRIES", "limits.stream_recovery_retries", 2))
    latest_tool_output_max_chars: int = field(default_factory=lambda: _env_int("BILL015_LATEST_TOOL_OUTPUT_MAX_CHARS", "limits.latest_tool_output_max_chars", 12000))
    args_done_timeout_ms: int = field(default_factory=lambda: _env_int("BILL015_ARGS_DONE_TIMEOUT_MS", "limits.args_done_timeout_ms", 900000))
    upstream_idle_timeout_ms: int = field(default_factory=lambda: _env_int("BILL015_UPSTREAM_IDLE_TIMEOUT_MS", "limits.upstream_idle_timeout_ms", 900000))
    client_heartbeat_interval_ms: int = field(default_factory=lambda: _env_int("BILL015_CLIENT_HEARTBEAT_INTERVAL_MS", "limits.client_heartbeat_interval_ms", 5000))
    cyber_policy_rotate: bool = field(default_factory=lambda: _env_bool("BILL015_CYBER_POLICY_ROTATE", "key_pool.cyber_policy_rotate", True))
    failed_key_cooldown_seconds: float = field(default_factory=lambda: _env_float("BILL015_FAILED_KEY_COOLDOWN_SECONDS", "key_pool.failed_key_cooldown_seconds", 600.0))
    evidence_dir: Path = field(default_factory=lambda: Path(_env_str("LOCAL_PROXY_LOG_DIR", "logging.dir", str(PROJECT_ROOT / "proxy_evidence"))))
    store_prompts: bool = field(default_factory=lambda: _env_bool("LOCAL_PROXY_STORE_PROMPTS", "logging.store_prompts", False))
    store_answers: bool = field(default_factory=lambda: _env_bool("LOCAL_PROXY_STORE_ANSWERS", "logging.store_answers", False))
    rotate_mb: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_LOG_ROTATE_MB", "logging.rotate_mb", 10))
    admin_token: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_ADMIN_TOKEN", "admin.token", ""))
    circuit_failures: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_CIRCUIT_FAILURES", "limits.circuit_failures", 0))
    packy_cookie: str = field(default_factory=lambda: _env_str("PACKY_COOKIE", "upstream.cookie", ""))
    packy_user_id: str = field(default_factory=lambda: _env_str("PACKY_USER_ID", "upstream.user_id", "192833"))
    usage_estimator: str = field(default_factory=lambda: _env_str("BILL015_USAGE_ESTIMATOR", "usage.estimator", "auto"))
    usage_prefer_tiktoken: bool = field(default_factory=lambda: _env_bool("BILL015_USAGE_PREFER_TIKTOKEN", "usage.prefer_tiktoken", True))
    usage_include_tools_schema: bool = field(default_factory=lambda: _env_bool("BILL015_USAGE_INCLUDE_TOOLS_SCHEMA", "usage.include_tools_schema", True))
    usage_include_images: bool = field(default_factory=lambda: _env_bool("BILL015_USAGE_INCLUDE_IMAGES", "usage.include_images", True))
    usage_cache_ratio_default: float = field(default_factory=lambda: _env_float("BILL015_USAGE_CACHE_RATIO_DEFAULT", "usage.cache_ratio_default", 0.85))
    usage_max_text_for_exact_tokenize: int = field(default_factory=lambda: _env_int("BILL015_USAGE_MAX_TEXT_FOR_EXACT_TOKENIZE", "usage.max_text_for_exact_tokenize", 120000))
    usage_audit_breakdown: bool = field(default_factory=lambda: _env_bool("BILL015_USAGE_AUDIT_BREAKDOWN", "usage.audit_breakdown", True))
    responses_fidelity_level: str = field(default_factory=lambda: _env_str("BILL015_RESPONSES_FIDELITY_LEVEL", "responses_events.fidelity_level", "native"))
    responses_emit_reasoning_summary: bool = field(default_factory=lambda: _env_bool("BILL015_RESPONSES_EMIT_REASONING_SUMMARY", "responses_events.emit_reasoning_summary", False))
    responses_emit_annotations: bool = field(default_factory=lambda: _env_bool("BILL015_RESPONSES_EMIT_ANNOTATIONS", "responses_events.emit_annotations", True))
    responses_allow_web_search_call_event: bool = field(default_factory=lambda: _env_bool("BILL015_RESPONSES_ALLOW_WEB_SEARCH_CALL_EVENT", "responses_events.allow_web_search_call_event", False))
    responses_emit_incomplete_on_truncation: bool = field(default_factory=lambda: _env_bool("BILL015_RESPONSES_EMIT_INCOMPLETE_ON_TRUNCATION", "responses_events.emit_incomplete_on_truncation", True))
    responses_strict_sequence_numbers: bool = field(default_factory=lambda: _env_bool("BILL015_RESPONSES_STRICT_SEQUENCE_NUMBERS", "responses_events.strict_sequence_numbers", True))
    responses_typed_keepalive: bool = field(default_factory=lambda: _env_bool("BILL015_RESPONSES_TYPED_KEEPALIVE", "responses_events.typed_keepalive", False))
    responses_chunk_size: int = field(default_factory=lambda: _env_int("BILL015_RESPONSES_CHUNK_SIZE", "responses_events.chunk_size", 256))
    tool_bridge_allow_unknown_tools: bool = field(default_factory=lambda: _env_bool("BILL015_TOOL_BRIDGE_ALLOW_UNKNOWN_TOOLS", "tool_bridge.allow_unknown_tools", False))
    tool_bridge_auto_expand_search: bool = field(default_factory=lambda: _env_bool("BILL015_TOOL_BRIDGE_AUTO_EXPAND_SEARCH", "tool_bridge.auto_expand_search", False))
    tool_bridge_local_web_research_preflight: bool = field(default_factory=lambda: _env_bool("BILL015_TOOL_BRIDGE_LOCAL_WEB_RESEARCH_PREFLIGHT", "tool_bridge.local_web_research_preflight", True))
    tool_bridge_schema_max_tools: int = field(default_factory=lambda: _env_int("BILL015_TOOL_BRIDGE_SCHEMA_MAX_TOOLS", "tool_bridge.schema_max_tools", 256))

    @property
    def upstream_api_keys(self) -> list[str]:
        """Return the ordered, de-duplicated API key pool.

        The environment variable keeps its legacy precedence as the primary
        key. ``upstream.api_keys`` supplies additional rotation keys.
        """
        env_value = os.getenv(self.upstream_api_key_env, "").strip()
        primary = env_value or self.upstream_api_key_file_value.strip()
        candidates = [primary, *self.upstream_api_keys_file_value]
        return list(dict.fromkeys(key.strip() for key in candidates if key and key.strip()))

    @property
    def upstream_api_key(self) -> str:
        keys = self.upstream_api_keys
        return keys[0] if keys else ""

    @property
    def upstream_configured(self) -> bool:
        return bool(self.upstream_api_keys)
    def map_model(self, model: str | None) -> str:
        """Map the client-requested model for the upstream call.

        Earlier versions forced every unknown model to default_model, which made
        switching Codex/CC Switch from gpt-5.5 to gpt-5.4 impossible. Default is
        now native passthrough: aliases are still honored, an empty model uses
        default_model, and force_default_model can be enabled only when the user
        explicitly wants one fixed upstream model.
        """
        if not model:
            return self.default_model
        mapped = self.model_aliases.get(str(model), str(model))
        if self.force_default_model:
            return self.model_aliases.get(str(model), self.default_model)
        return mapped

    def public_model_ids(self) -> list[str]:
        ids = [self.default_model]
        ids.extend(self.model_aliases.keys())
        return list(dict.fromkeys(ids))

    def validate_mode(self) -> None:
        if self.mode not in {"exploit", "verify", "normal", "dry-run"}:
            self.mode = "exploit"


settings = Settings()
settings.validate_mode()
if settings.bridge_strategy not in {"native_tool_first", "emit_value"}:
    settings.bridge_strategy = "emit_value"
if settings.force_emit_value and settings.bridge_strategy == "native_tool_first":
    # Native-tool-first exposes real tools upstream and can let the provider
    # account for prompt/output tokens before our local abort boundary.  In
    # safe bridge mode fails closed back to the proven forced emit_value bridge.
    settings.bridge_strategy = "emit_value"
settings.config_warnings = CONFIG_WARNINGS
