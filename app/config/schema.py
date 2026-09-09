from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerConfig(StrictConfigModel):
    host: str | None = None
    port: int | None = None
    cors_allow_origins: list[str] | None = None
    cors_allow_origin_regex: str | None = None


class UpstreamConfig(StrictConfigModel):
    base_url: str | None = None
    api_key: str | None = None
    api_keys: list[str] | None = None
    api_key_env: str | None = None
    cookie: str | None = None
    user_id: str | None = None
    timeout_seconds: float | None = None


class ModelConfig(StrictConfigModel):
    default: str | None = None
    aliases: dict[str, str] | None = None
    force_default: bool | None = None


class Bill015Config(StrictConfigModel):
    function_name: str | None = None
    answer_field: str | None = None
    max_output_tokens: int | None = None
    compaction_max_output_tokens: int | None = None
    max_answer_chars: int | None = None
    force_emit_value: bool | None = None
    block_passthrough: bool | None = None
    block_normal_mode: bool | None = None
    max_tool_argument_chars: int | None = None
    max_total_emit_value_chars: int | None = None
    max_emit_value_calls: int | None = None
    progress_continuation_rounds: int | None = None
    emit_value_quiet_ms: int | None = None
    strict_zero: bool | None = None
    malformed_retries: int | None = None


class LoggingConfig(StrictConfigModel):
    dir: str | None = None
    store_prompts: bool | None = None
    store_answers: bool | None = None
    rotate_mb: int | None = None
    payload_probe: bool | None = None


class UsageConfig(StrictConfigModel):
    estimator: str | None = None
    prefer_tiktoken: bool | None = None
    include_tools_schema: bool | None = None
    max_text_for_exact_tokenize: int | None = None


class ReasoningConfig(StrictConfigModel):
    effort: str | None = None
    summary: str | None = None


class LimitsConfig(StrictConfigModel):
    max_concurrency: int | None = None
    max_queue_size: int | None = None
    queue_wait_timeout_ms: int | None = None
    request_total_timeout_ms: int | None = None
    max_request_bytes: int | None = None
    upstream_retries: int | None = None
    upstream_retry_backoff_ms: int | None = None
    context_window_tokens: int | None = None
    model_context_windows: dict[str, int] | None = None
    auto_compact_percent: int | None = None
    compact_target_percent: int | None = None
    context_recovery_retries: int | None = None
    empty_stream_retries: int | None = None
    stream_recovery_retries: int | None = None
    latest_tool_output_max_chars: int | None = None
    args_done_timeout_ms: int | None = None
    upstream_idle_timeout_ms: int | None = None
    circuit_failures: int | None = None
    client_heartbeat_interval_ms: int | None = None


class KeyPoolConfig(StrictConfigModel):
    cyber_policy_rotate: bool | None = None
    failed_key_cooldown_seconds: float | None = None
    max_policy_rotations_per_request: int | None = None
    cyber_policy_retry_delay_seconds: float | None = None


class AdminConfig(StrictConfigModel):
    token: str | None = None


class ToolBridgeConfig(StrictConfigModel):
    allow_unknown_tools: bool | None = None
    local_web_research_preflight: bool | None = None
    schema_max_tools: int | None = None
    selection_max_tools: int | None = None
    catalog_max_chars: int | None = None


class WebConfig(StrictConfigModel):
    enabled: bool | None = None
    backend: str | None = None
    api_url: str | None = None
    api_key: str | None = None
    search_limit_default: int | None = None
    extract_char_limit: int | None = None
    timeout_seconds: float | None = None
    max_rounds_per_request: int | None = None


class MultimodalConfig(StrictConfigModel):
    strategy: Literal["reject", "local_extract", "native_passthrough"] | None = None
    max_images: int | None = None
    max_image_bytes: int | None = None


class LocalConfigSchema(StrictConfigModel):
    server: ServerConfig | None = None
    upstream: UpstreamConfig | None = None
    mode: Literal["exploit", "verify", "normal", "dry-run"] | None = None
    model: ModelConfig | None = None
    bill015: Bill015Config | None = None
    logging: LoggingConfig | None = None
    usage: UsageConfig | None = None
    reasoning: ReasoningConfig | None = None
    limits: LimitsConfig | None = None
    key_pool: KeyPoolConfig | None = None
    admin: AdminConfig | None = None
    tool_bridge: ToolBridgeConfig | None = None
    multimodal: MultimodalConfig | None = None
    web: WebConfig | None = None
    responses_events: dict[str, Any] | None = None


def validate_local_config(obj: dict[str, Any]) -> list[str]:
    try:
        LocalConfigSchema.model_validate(obj)
        return []
    except ValidationError as e:
        warnings: list[str] = []
        for err in e.errors():
            loc = ".".join(str(part) for part in err.get("loc", ())) or "<root>"
            msg = str(err.get("msg") or "invalid value")
            warnings.append(f"{loc}: {msg}")
        return warnings[:50]
