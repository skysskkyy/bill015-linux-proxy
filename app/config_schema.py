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
    max_answer_chars: int | None = None
    strict_zero: bool | None = None


class LoggingConfig(StrictConfigModel):
    dir: str | None = None
    store_prompts: bool | None = None
    store_answers: bool | None = None
    rotate_mb: int | None = None


class UsageConfig(StrictConfigModel):
    estimator: str | None = None
    prefer_tiktoken: bool | None = None
    include_tools_schema: bool | None = None
    include_images: bool | None = None
    cache_ratio_default: float | None = None
    max_text_for_exact_tokenize: int | None = None
    audit_breakdown: bool | None = None


class ReasoningConfig(StrictConfigModel):
    effort: str | None = None
    summary: str | None = None


class ResponsesEventsConfig(StrictConfigModel):
    fidelity_level: str | None = None
    emit_reasoning_summary: bool | None = None
    emit_annotations: bool | None = None
    allow_web_search_call_event: bool | None = None
    emit_incomplete_on_truncation: bool | None = None
    strict_sequence_numbers: bool | None = None
    typed_keepalive: bool | None = None
    chunk_size: int | None = None


class LimitsConfig(StrictConfigModel):
    max_concurrency: int | None = None
    max_request_bytes: int | None = None
    upstream_retries: int | None = None
    upstream_retry_backoff_ms: int | None = None
    args_done_timeout_ms: int | None = None
    upstream_idle_timeout_ms: int | None = None
    circuit_failures: int | None = None
    client_heartbeat_interval_ms: int | None = None


class AdminConfig(StrictConfigModel):
    token: str | None = None


class ToolBridgeConfig(StrictConfigModel):
    allow_unknown_tools: bool | None = None


class LocalConfigSchema(StrictConfigModel):
    server: ServerConfig | None = None
    upstream: UpstreamConfig | None = None
    mode: Literal["exploit", "verify", "normal", "dry-run"] | None = None
    model: ModelConfig | None = None
    bill015: Bill015Config | None = None
    logging: LoggingConfig | None = None
    usage: UsageConfig | None = None
    reasoning: ReasoningConfig | None = None
    responses_events: ResponsesEventsConfig | None = None
    limits: LimitsConfig | None = None
    admin: AdminConfig | None = None
    tool_bridge: ToolBridgeConfig | None = None


def validate_local_config(obj: dict[str, Any]) -> list[str]:
    """Return human-readable schema warnings without blocking local startup."""
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
