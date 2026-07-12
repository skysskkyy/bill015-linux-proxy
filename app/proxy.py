from __future__ import annotations

from .chat_events import chat_json, chat_sse_generator

# Compatibility facade: keep existing imports stable while implementation lives
# in smaller modules with clearer responsibilities.
from .models import Bill015Result, BridgeToolCall, NormalizedRequest, local_response_id
from .normalization import (
    collect_deferred_tools_from_input,
    combine_tool_catalogs,
    detect_request_kind,
    extract_context_messages,
    extract_ordered_instruction_context,
    extract_role_contexts,
    flatten_chat_messages,
    flatten_content,
    flatten_responses_input,
    last_user_instruction,
    model_identity_instruction,
    native_input_transcript,
    normalize_chat_request,
    normalize_responses_request,
    parse_codex_turn_metadata,
    request_needs_passthrough,
)
from .payloads import (
    build_bill015_payload,
    build_compaction_bill015_payload,
    build_emit_value_schema,
)
from .response_events import response_json, responses_sse_generator
from .tool_bridge import build_client_tool_catalog, parse_function_arguments
from .upstream import (
    audit_from_result,
    collect_bill015_result_from_events,
    compute_delta,
    dry_run_response,
    execute_bill015,
    fetch_user_self,
)
from .upstream_client import normal_forward_json, normal_forward_stream, prepare_passthrough_payload

__all__ = [
    'Bill015Result',
    'BridgeToolCall',
    'NormalizedRequest',
    'local_response_id',
    'model_identity_instruction',
    'flatten_content',
    'flatten_responses_input',
    'flatten_chat_messages',
    'parse_codex_turn_metadata',
    'detect_request_kind',
    'extract_context_messages',
    'extract_ordered_instruction_context',
    'extract_role_contexts',
    'native_input_transcript',
    'last_user_instruction',
    'request_needs_passthrough',
    'collect_deferred_tools_from_input',
    'combine_tool_catalogs',
    'normalize_responses_request',
    'normalize_chat_request',
    'build_emit_value_schema',
    'build_compaction_bill015_payload',
    'build_bill015_payload',
    'build_client_tool_catalog',
    'parse_function_arguments',
    'collect_bill015_result_from_events',
    'audit_from_result',
    'compute_delta',
    'dry_run_response',
    'execute_bill015',
    'fetch_user_self',
    'normal_forward_json',
    'normal_forward_stream',
    'prepare_passthrough_payload',
    'chat_json',
    'chat_sse_generator',
    'response_json',
    'responses_sse_generator',
]
