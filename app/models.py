from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

ClientAPI = Literal["responses", "chat.completions"]


@dataclass
class NormalizedRequest:
    model: str
    instructions: str
    user_input: str
    want_stream: bool
    client_api: ClientAPI
    is_primary_path: bool
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    client_metadata: dict[str, Any] | None = None
    original_model: str | None = None
    raw_input: Any = None
    raw_tools: Any = None
    request_kind: str = "turn"
    is_compaction: bool = False
    estimated_input_tokens: int = 0
    usage_estimate: Any = None
    parallel_tool_calls: bool = True
    tool_choice: Any = "auto"
    prompt_cache_key: str | None = None
    text_config: dict[str, Any] | None = None
    tools_summary: str = ""
    tools_catalog: str = ""
    tool_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_history: Any = None
    latest_tool_summary: str = ""
    pending_tool_call_count: int = 0
    latest_tool_failed: bool = False


@dataclass
class BridgeToolCall:
    id: str
    name: str
    arguments: str
    call_type: str = "function"
    requested_name: str | None = None
    namespace: str | None = None


@dataclass
class Bill015Result:
    local_request_id: str
    upstream_response_id: str | None = None
    answer: str = ""
    bridge_mode: str = "answer"
    tool_calls: list[BridgeToolCall] = field(default_factory=list)
    raw_arguments: str = ""
    function_call_seen: bool = False
    args_done_seen: bool = False
    upstream_completed_seen: bool = False
    aborted: bool = False
    malformed_function_args: bool = False
    repaired_args: bool = False
    duration_ms: int = 0
    event_sequence: list[str] = field(default_factory=list)
    error: str | None = None
    verify_pre: dict[str, Any] | None = None
    verify_post: dict[str, Any] | None = None
    verify_delta: dict[str, Any] | None = None


def local_response_id() -> str:
    return "resp_local_" + uuid.uuid4().hex
