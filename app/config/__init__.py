from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .schema import validate_local_config

Mode = Literal["exploit", "verify", "normal", "dry-run"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_WARNINGS: list[str] = []


def _xdg_config_path() -> Path:
    base = os.getenv("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "bill015" / "config.json"


def _resolve_config_path() -> Path:
    raw = os.getenv("BILL015_CONFIG_PATH", "").strip()
    if raw:
        return Path(raw)
    local = PROJECT_ROOT / "config.local.json"
    if local.exists():
        return local
    xdg = _xdg_config_path()
    if xdg.exists():
        return xdg
    return local


def _load_local_config() -> dict[str, Any]:
    global CONFIG_WARNINGS
    CONFIG_WARNINGS = []
    path = _resolve_config_path()
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
DEFAULT_CONFIG_PATH = _resolve_config_path()


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


def _cfg_dict(path: str, default: dict[str, str]) -> dict[str, str]:
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
    return _env_bool("BILL015_STRICT_ZERO", "bill015.strict_zero", True)


@dataclass
class Settings:
    config_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)
    host: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_HOST", "server.host", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_PORT", "server.port", 8787))
    cors_allow_origins: list[str] = field(default_factory=lambda: _cfg_list("server.cors_allow_origins", []))
    cors_allow_origin_regex: str = field(
        default_factory=lambda: _env_str(
            "LOCAL_PROXY_CORS_ALLOW_ORIGIN_REGEX",
            "server.cors_allow_origin_regex",
            r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        )
    )
    mode: Mode = field(default_factory=lambda: _env_str("LOCAL_PROXY_MODE", "mode", "exploit").strip().lower())  # type: ignore[assignment]
    upstream_base_url: str = field(default_factory=lambda: _env_str("PACKY_BASE_URL", "upstream.base_url", "https://packyapi.com").rstrip("/"))
    upstream_api_key_env: str = field(default_factory=lambda: _env_str("PACKY_API_KEY_ENV", "upstream.api_key_env", "PACKY_API_KEY"))
    upstream_api_key_file_value: str = field(default_factory=lambda: _env_str("", "upstream.api_key", ""))
    upstream_api_keys_file_value: list[str] = field(default_factory=lambda: _cfg_list("upstream.api_keys", []))
    default_model: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_MODEL", "model.default", "gpt-5.5"))
    model_aliases: dict[str, str] = field(
        default_factory=lambda: _cfg_dict("model.aliases", {"codex-gpt55": "gpt-5.5", "codex-gpt54": "gpt-5.4"})
    )
    force_default_model: bool = field(default_factory=lambda: _env_bool("LOCAL_PROXY_FORCE_DEFAULT_MODEL", "model.force_default", False))
    function_name: str = field(default_factory=lambda: _env_str("BILL015_FUNCTION_NAME", "bill015.function_name", "emit_value"))
    answer_field: str = field(default_factory=lambda: _env_str("BILL015_ANSWER_FIELD", "bill015.answer_field", "answer"))
    max_output_tokens: int = field(default_factory=lambda: _env_int("BILL015_MAX_OUTPUT_TOKENS", "bill015.max_output_tokens", 8192))
    compaction_max_output_tokens: int = field(
        default_factory=lambda: _env_int("BILL015_COMPACTION_MAX_OUTPUT_TOKENS", "bill015.compaction_max_output_tokens", 8192)
    )
    force_emit_value: bool = field(
        default_factory=lambda: _env_bool("BILL015_FORCE_EMIT_VALUE", "bill015.force_emit_value", _legacy_strict_zero_default())
    )
    block_passthrough: bool = field(
        default_factory=lambda: _env_bool("BILL015_BLOCK_PASSTHROUGH", "bill015.block_passthrough", _legacy_strict_zero_default())
    )
    block_normal_mode: bool = field(
        default_factory=lambda: _env_bool("BILL015_BLOCK_NORMAL_MODE", "bill015.block_normal_mode", _legacy_strict_zero_default())
    )
    reasoning_effort: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_REASONING_EFFORT", "reasoning.effort", "high"))
    reasoning_summary: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_REASONING_SUMMARY", "reasoning.summary", "auto"))
    max_answer_chars: int = field(default_factory=lambda: _env_int("BILL015_MAX_ANSWER_CHARS", "bill015.max_answer_chars", 65536))
    max_tool_argument_chars: int = field(default_factory=lambda: _env_int("BILL015_MAX_TOOL_ARGUMENT_CHARS", "bill015.max_tool_argument_chars", 262144))
    max_total_emit_value_chars: int = field(default_factory=lambda: _env_int("BILL015_MAX_TOTAL_EMIT_VALUE_CHARS", "bill015.max_total_emit_value_chars", 393216))
    max_emit_value_calls: int = field(default_factory=lambda: _env_int("BILL015_MAX_EMIT_VALUE_CALLS", "bill015.max_emit_value_calls", 8))
    progress_continuation_rounds: int = field(
        default_factory=lambda: _env_int("BILL015_PROGRESS_CONTINUATION_ROUNDS", "bill015.progress_continuation_rounds", 3)
    )
    emit_value_quiet_ms: int = field(default_factory=lambda: _env_int("BILL015_EMIT_VALUE_QUIET_MS", "bill015.emit_value_quiet_ms", 250))
    max_request_bytes: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_MAX_REQUEST_BYTES", "limits.max_request_bytes", 1048576))
    max_concurrency: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_MAX_CONCURRENCY", "limits.max_concurrency", 12))
    max_queue_size: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_MAX_QUEUE_SIZE", "limits.max_queue_size", 8))
    queue_wait_timeout_ms: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_QUEUE_WAIT_TIMEOUT_MS", "limits.queue_wait_timeout_ms", 60000))
    request_total_timeout_ms: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_REQUEST_TOTAL_TIMEOUT_MS", "limits.request_total_timeout_ms", 1200000))
    upstream_timeout_seconds: float = field(default_factory=lambda: _env_float("PACKY_TIMEOUT_SECONDS", "upstream.timeout_seconds", 600.0))
    upstream_retries: int = field(default_factory=lambda: _env_int("BILL015_UPSTREAM_RETRIES", "limits.upstream_retries", 2))
    upstream_retry_backoff_ms: int = field(default_factory=lambda: _env_int("BILL015_UPSTREAM_RETRY_BACKOFF_MS", "limits.upstream_retry_backoff_ms", 700))
    context_window_tokens: int = field(default_factory=lambda: _env_int("BILL015_CONTEXT_WINDOW_TOKENS", "limits.context_window_tokens", 128000))
    auto_compact_percent: int = field(default_factory=lambda: _env_int("BILL015_AUTO_COMPACT_PERCENT", "limits.auto_compact_percent", 90))
    compact_target_percent: int = field(default_factory=lambda: _env_int("BILL015_COMPACT_TARGET_PERCENT", "limits.compact_target_percent", 75))
    context_recovery_retries: int = field(default_factory=lambda: _env_int("BILL015_CONTEXT_RECOVERY_RETRIES", "limits.context_recovery_retries", 2))
    empty_stream_retries: int = field(default_factory=lambda: _env_int("BILL015_EMPTY_STREAM_RETRIES", "limits.empty_stream_retries", 1))
    stream_recovery_retries: int = field(default_factory=lambda: _env_int("BILL015_STREAM_RECOVERY_RETRIES", "limits.stream_recovery_retries", 1))
    latest_tool_output_max_chars: int = field(default_factory=lambda: _env_int("BILL015_LATEST_TOOL_OUTPUT_MAX_CHARS", "limits.latest_tool_output_max_chars", 12000))
    args_done_timeout_ms: int = field(default_factory=lambda: _env_int("BILL015_ARGS_DONE_TIMEOUT_MS", "limits.args_done_timeout_ms", 600000))
    upstream_idle_timeout_ms: int = field(default_factory=lambda: _env_int("BILL015_UPSTREAM_IDLE_TIMEOUT_MS", "limits.upstream_idle_timeout_ms", 300000))
    client_heartbeat_interval_ms: int = field(default_factory=lambda: _env_int("BILL015_CLIENT_HEARTBEAT_INTERVAL_MS", "limits.client_heartbeat_interval_ms", 5000))
    circuit_failures: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_CIRCUIT_FAILURES", "limits.circuit_failures", 0))
    cyber_policy_rotate: bool = field(default_factory=lambda: _env_bool("BILL015_CYBER_POLICY_ROTATE", "key_pool.cyber_policy_rotate", True))
    failed_key_cooldown_seconds: float = field(default_factory=lambda: _env_float("BILL015_FAILED_KEY_COOLDOWN_SECONDS", "key_pool.failed_key_cooldown_seconds", 600.0))
    max_policy_rotations_per_request: int = field(
        default_factory=lambda: _env_int("BILL015_MAX_POLICY_ROTATIONS_PER_REQUEST", "key_pool.max_policy_rotations_per_request", 8)
    )
    cyber_policy_retry_delay_seconds: float = field(
        default_factory=lambda: _env_float("BILL015_CYBER_POLICY_RETRY_DELAY_SECONDS", "key_pool.cyber_policy_retry_delay_seconds", 0.0)
    )
    evidence_dir: Path = field(default_factory=lambda: Path(_env_str("LOCAL_PROXY_LOG_DIR", "logging.dir", str(PROJECT_ROOT / "proxy_evidence"))))
    store_prompts: bool = field(default_factory=lambda: _env_bool("LOCAL_PROXY_STORE_PROMPTS", "logging.store_prompts", False))
    store_answers: bool = field(default_factory=lambda: _env_bool("LOCAL_PROXY_STORE_ANSWERS", "logging.store_answers", False))
    rotate_mb: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_LOG_ROTATE_MB", "logging.rotate_mb", 10))
    admin_token: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_ADMIN_TOKEN", "admin.token", ""))
    packy_cookie: str = field(default_factory=lambda: _env_str("PACKY_COOKIE", "upstream.cookie", ""))
    packy_user_id: str = field(default_factory=lambda: _env_str("PACKY_USER_ID", "upstream.user_id", ""))
    usage_prefer_tiktoken: bool = field(default_factory=lambda: _env_bool("BILL015_USAGE_PREFER_TIKTOKEN", "usage.prefer_tiktoken", True))
    usage_max_text_for_exact_tokenize: int = field(
        default_factory=lambda: _env_int("BILL015_USAGE_MAX_TEXT_FOR_EXACT_TOKENIZE", "usage.max_text_for_exact_tokenize", 120000)
    )
    tool_bridge_allow_unknown_tools: bool = field(
        default_factory=lambda: _env_bool("BILL015_TOOL_BRIDGE_ALLOW_UNKNOWN_TOOLS", "tool_bridge.allow_unknown_tools", False)
    )
    tool_bridge_local_web_research_preflight: bool = field(
        default_factory=lambda: _env_bool(
            "BILL015_TOOL_BRIDGE_LOCAL_WEB_RESEARCH_PREFLIGHT", "tool_bridge.local_web_research_preflight", True
        )
    )
    tool_bridge_schema_max_tools: int = field(default_factory=lambda: _env_int("BILL015_TOOL_BRIDGE_SCHEMA_MAX_TOOLS", "tool_bridge.schema_max_tools", 256))
    tool_bridge_selection_max_tools: int = field(
        default_factory=lambda: _env_int("BILL015_TOOL_BRIDGE_SELECTION_MAX_TOOLS", "tool_bridge.selection_max_tools", 160)
    )
    tool_bridge_catalog_max_chars: int = field(default_factory=lambda: _env_int("BILL015_TOOL_BRIDGE_CATALOG_MAX_CHARS", "tool_bridge.catalog_max_chars", 120000))
    multimodal_strategy: str = field(default_factory=lambda: _env_str("BILL015_MULTIMODAL_STRATEGY", "multimodal.strategy", "reject").strip().lower())
    multimodal_max_images: int = field(default_factory=lambda: _env_int("BILL015_MULTIMODAL_MAX_IMAGES", "multimodal.max_images", 8))
    multimodal_max_image_bytes: int = field(default_factory=lambda: _env_int("BILL015_MULTIMODAL_MAX_IMAGE_BYTES", "multimodal.max_image_bytes", 10485760))
    responses_chunk_size: int = field(default_factory=lambda: _env_int("BILL015_RESPONSES_CHUNK_SIZE", "responses_events.chunk_size", 256))
    web_enabled: bool = field(default_factory=lambda: _env_bool("BILL015_WEB_ENABLED", "web.enabled", True))
    web_backend: str = field(default_factory=lambda: _env_str("BILL015_WEB_BACKEND", "web.backend", "firecrawl"))
    web_api_url: str = field(default_factory=lambda: _env_str("FIRECRAWL_API_URL", "web.api_url", "http://127.0.0.1:3002").rstrip("/"))
    web_api_key: str = field(default_factory=lambda: _env_str("FIRECRAWL_API_KEY", "web.api_key", ""))
    web_search_limit_default: int = field(default_factory=lambda: _env_int("BILL015_WEB_SEARCH_LIMIT", "web.search_limit_default", 5))
    web_extract_char_limit: int = field(default_factory=lambda: _env_int("BILL015_WEB_EXTRACT_CHAR_LIMIT", "web.extract_char_limit", 15000))
    web_timeout_seconds: float = field(default_factory=lambda: _env_float("BILL015_WEB_TIMEOUT_SECONDS", "web.timeout_seconds", 60.0))
    web_max_rounds_per_request: int = field(default_factory=lambda: _env_int("BILL015_WEB_MAX_ROUNDS", "web.max_rounds_per_request", 8))
    config_warnings: list[str] = field(default_factory=list)

    @property
    def upstream_api_keys(self) -> list[str]:
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
if settings.multimodal_strategy not in {"reject", "local_extract", "native_passthrough"}:
    settings.multimodal_strategy = "reject"
settings.config_warnings = CONFIG_WARNINGS
