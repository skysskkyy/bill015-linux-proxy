from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

ClientAPI = Literal["responses", "chat.completions"]
CallType = Literal["function", "custom", "tool_search"]

TOOL_CALL_TYPES = {
    "function_call",
    "custom_tool_call",
    "tool_search_call",
    "web_search_call",
    "computer_call",
}
TOOL_OUTPUT_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "tool_search_output",
    "computer_call_output",
    "web_search_output",
    "tool_result",
}


def local_response_id() -> str:
    return "resp_local_" + uuid.uuid4().hex


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


@dataclass
class ToolSpec:
    alias: str
    name: str
    namespace: str | None
    call_type: CallType
    raw_type: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    core: bool = False


@dataclass
class Catalog:
    specs: dict[str, ToolSpec] = field(default_factory=dict)
    selected: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    dropped: int = 0

    def get(self, name: str) -> ToolSpec | None:
        key = (name or "").strip()
        if key in self.specs:
            return self.specs[key]
        lower = key.lower()
        for alias, spec in self.specs.items():
            if alias.lower() == lower or spec.name.lower() == lower:
                return spec
        return None

    @property
    def selected_specs(self) -> list[ToolSpec]:
        out: list[ToolSpec] = []
        seen: set[str] = set()
        for alias in self.selected:
            spec = self.specs.get(alias)
            if spec and spec.alias not in seen:
                out.append(spec)
                seen.add(spec.alias)
        return out


@dataclass
class BridgeToolCall:
    id: str
    name: str
    arguments: str
    call_type: CallType = "function"
    namespace: str | None = None
    input: str = ""
    execution: str | None = None
    search_arguments: dict[str, Any] | None = None


@dataclass
class WrapperCall:
    mode: str
    answer: str
    tool_calls: list[BridgeToolCall]
    raw_arguments: str
    malformed: bool = False


@dataclass
class Turn:
    model: str
    want_stream: bool
    client_api: ClientAPI
    items: list[dict[str, Any]]
    current_user: str
    catalog: Catalog
    instructions: str = ""
    system_text: str = ""
    developer_text: str = ""
    reasoning: dict[str, Any] | None = None
    request_headers: dict[str, str] = field(default_factory=dict)
    max_output_tokens: int | None = None
    temperature: float | None = None
    metadata: dict[str, Any] | None = None
    client_metadata: dict[str, Any] | None = None
    request_kind: str = "turn"
    is_compaction: bool = False
    responses_lite: bool = False
    estimated_input_tokens: int = 0
    original_model: str | None = None
    raw_body: dict[str, Any] = field(default_factory=dict)
    image_replacements: int = 0
    loss_notices: list[str] = field(default_factory=list)
    web_intent: bool = False


@dataclass
class TurnResult:
    local_request_id: str
    answer: str = ""
    commentary: str = ""
    tool_calls: list[BridgeToolCall] = field(default_factory=list)
    wrapper_count: int = 0
    raw_arguments: str = ""
    aborted: bool = False
    args_done_seen: bool = False
    upstream_completed_seen: bool = False
    upstream_response_id: str | None = None
    duration_ms: int = 0
    retry_count: int = 0
    retry_reasons: list[str] = field(default_factory=list)
    event_sequence: list[str] = field(default_factory=list)
    error: str | None = None
    malformed: bool = False
    upstream_key_index: int | None = None
    upstream_key_count: int = 0
    key_switch_count: int = 0
    compaction_count: int = 0
    clipped_tool_output_count: int = 0
    packed_item_count: int = 0
    loss_notices: list[str] = field(default_factory=list)
    verify_pre: dict[str, Any] | None = None
    verify_post: dict[str, Any] | None = None
    verify_delta: dict[str, Any] | None = None
