from __future__ import annotations

import json
from typing import Any

from ..config import Settings, settings
from ..protocol.models import BridgeToolCall, Catalog, WrapperCall, new_call_id
from ..protocol.names import is_namespace_only, recover_tool_name, resolve_catalog_tool, split_tool_identity
from .exec_source import is_code_mode_exec, normalize_exec_source, unwrap_exec_source


def _load_object(raw: str) -> tuple[dict[str, Any], bool]:
    text = (raw or "").strip()
    if not text:
        return {}, True
    try:
        obj = json.loads(text)
        return (obj, False) if isinstance(obj, dict) else ({}, True)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj, True
            except Exception:
                pass
        return {}, True


def _as_args_string(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(value)


def _resolve(name: str, namespace: str, catalog: Catalog, cfg: Settings) -> tuple[str, str | None, str] | None:
    spec = resolve_catalog_tool(catalog, name, namespace or None)
    if spec is not None:
        if is_namespace_only(spec.name, spec.namespace) or not spec.name:
            return None
        return spec.name, spec.namespace, spec.call_type
    raw, ns = split_tool_identity(name, namespace)
    if not raw:
        return None
    if is_namespace_only(raw, ns):
        if not cfg.tool_bridge_allow_unknown_tools:
            return None
        if raw.lower() in {"functions", "mcp", "web"}:
            return None
        if ns and raw.lower() == ns.lower():
            ns = None
        return raw, ns, "function"
    if cfg.tool_bridge_allow_unknown_tools:
        return raw, ns, "function"
    return None


def _one_call(obj: dict[str, Any], catalog: Catalog, cfg: Settings) -> BridgeToolCall | None:
    name, namespace = recover_tool_name(obj)
    if not name and namespace:
        name, namespace = namespace, None
    resolved = _resolve(name, namespace or "", catalog, cfg)
    if resolved is None:
        return None
    tool_name, ns, call_type = resolved
    arguments = obj.get("arguments")
    raw_input = obj.get("input")
    if is_code_mode_exec(tool_name, ns):
        call_type = "custom"
        if isinstance(raw_input, str) and raw_input.strip():
            src = raw_input
        elif isinstance(arguments, dict):
            src = next((str(arguments[k]) for k in ("input", "code", "source", "script") if isinstance(arguments.get(k), str) and arguments.get(k).strip()), _as_args_string(arguments))
        elif isinstance(arguments, str):
            src = arguments
        else:
            src = _as_args_string(arguments)
        input_text = normalize_exec_source(src)
        return BridgeToolCall(
            id=new_call_id(),
            name=tool_name,
            arguments=input_text,
            call_type="custom",
            namespace=ns,
            input=input_text,
        )
    if call_type == "custom":
        input_text = raw_input if isinstance(raw_input, str) and raw_input else _as_args_string(arguments)
        if not input_text and isinstance(arguments, str) and arguments not in {"", "{}"}:
            input_text = arguments
        input_text = unwrap_exec_source(input_text) if input_text.startswith("{") and '"input"' in input_text else input_text
        return BridgeToolCall(
            id=new_call_id(),
            name=tool_name,
            arguments=input_text,
            call_type="custom",
            namespace=ns,
            input=input_text,
        )
    if call_type == "tool_search":
        if isinstance(arguments, dict):
            search = arguments
        else:
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else {}
            except Exception:
                parsed = {}
            search = parsed if isinstance(parsed, dict) else {"query": str(arguments or name)}
        if "query" not in search:
            search = {"query": str(search.get("q") or name), "limit": int(search.get("limit") or 12)}
        return BridgeToolCall(
            id=new_call_id(),
            name="tool_search",
            arguments="",
            call_type="tool_search",
            execution="client",
            search_arguments=search,
        )
    arg_text = _as_args_string(arguments)
    if (not arg_text or arg_text in {"", "{}"}) and isinstance(raw_input, str) and raw_input:
        arg_text = raw_input
    if len(arg_text) > cfg.max_tool_argument_chars:
        return None
    return BridgeToolCall(id=new_call_id(), name=tool_name, arguments=arg_text, call_type="function", namespace=ns)


def parse_emit_value(raw: str, catalog: Catalog, cfg: Settings = settings) -> WrapperCall:
    obj, malformed = _load_object(raw)
    mode = str(obj.get("mode") or "").strip().lower()
    answer = obj.get(cfg.answer_field)
    answer_text = answer if isinstance(answer, str) else ("" if answer is None else str(answer))
    raw_calls = obj.get("tool_calls")
    calls: list[BridgeToolCall] = []
    if isinstance(raw_calls, list):
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            parsed = _one_call(item, catalog, cfg)
            if parsed is not None:
                calls.append(parsed)
    if calls:
        mode = "tool_call"
    else:
        mode = "answer"
    total = len(raw) + sum(len(call.arguments) + len(call.input) for call in calls)
    if total > cfg.max_total_emit_value_chars:
        return WrapperCall(mode="answer", answer=answer_text or "tool payload exceeded local integrity limit", tool_calls=[], raw_arguments=raw, malformed=True)
    return WrapperCall(mode=mode, answer=answer_text, tool_calls=calls, raw_arguments=raw, malformed=malformed)
