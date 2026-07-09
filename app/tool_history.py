from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

CALL_TYPES = {
    "function_call",
    "custom_tool_call",
    "tool_search_call",
    "web_search_call",
    "computer_call",
}

OUTPUT_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "tool_result",
    "tool_search_output",
    "computer_call_output",
    "web_search_output",
}

EXIT_CODE_RE = re.compile(r"(?im)^\s*Exit code:\s*(-?\d+)\s*$")
WALL_TIME_RE = re.compile(r"(?im)^\s*Wall time:\s*([^\r\n]+)")
ERROR_LINE_RE = re.compile(r"(?i)(error|failed|failure|traceback|exception|stderr|timeout|denied)")


@dataclass
class ToolCallRecord:
    call_id: str
    name: str
    call_type: str
    arguments: str
    index: int


@dataclass
class ToolOutputRecord:
    call_id: str
    name: str
    call_type: str
    output: str
    exit_code: int | None
    success: bool | None
    stderr_excerpt: str
    stdout_excerpt: str
    raw_chars: int
    index: int


@dataclass
class ToolHistory:
    calls: list[ToolCallRecord] = field(default_factory=list)
    outputs: list[ToolOutputRecord] = field(default_factory=list)
    latest_outputs: list[ToolOutputRecord] = field(default_factory=list)
    pending_calls: list[ToolCallRecord] = field(default_factory=list)
    failed_outputs: list[ToolOutputRecord] = field(default_factory=list)
    successful_outputs: list[ToolOutputRecord] = field(default_factory=list)
    exposed_deferred_tools: list[dict[str, Any]] = field(default_factory=list)


def parse_tool_history(value: Any) -> ToolHistory:
    history = ToolHistory()
    if not isinstance(value, list):
        return history

    calls_by_id: dict[str, ToolCallRecord] = {}
    output_call_ids: set[str] = set()

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        typ = str(item.get("type") or "")
        if typ in CALL_TYPES:
            record = _parse_call(item, index)
            if record:
                history.calls.append(record)
                calls_by_id[record.call_id] = record
            continue
        if typ in OUTPUT_TYPES:
            output = _parse_output(item, index, calls_by_id)
            if output:
                history.outputs.append(output)
                if output.call_id:
                    output_call_ids.add(output.call_id)
            if typ == "tool_search_output":
                history.exposed_deferred_tools.extend(_extract_deferred_tools(item.get("tools")))

    history.latest_outputs = history.outputs[-3:]
    history.pending_calls = [call for call in history.calls if call.call_id and call.call_id not in output_call_ids]
    history.failed_outputs = [out for out in history.outputs if out.success is False]
    history.successful_outputs = [out for out in history.outputs if out.success is True]
    return history


def render_tool_feedback_for_model(history: ToolHistory) -> str:
    if not (
        history.calls
        or history.outputs
        or history.pending_calls
        or history.exposed_deferred_tools
    ):
        return ""

    lines: list[str] = ["Recent local Codex tool results:"]

    if len(history.calls) > 1 or len(history.latest_outputs) > 1:
        completed = len(history.outputs)
        succeeded = len(history.successful_outputs)
        failed = len(history.failed_outputs)
        lines.extend(
            [
                "",
                "Parallel/history batch result:",
                f"- {len(history.calls)} calls requested",
                f"- {completed} outputs completed",
                f"- {succeeded} success",
                f"- {failed} failed",
            ]
        )

    for offset, output in enumerate(history.latest_outputs, start=1):
        call = _find_call(history.calls, output.call_id)
        status = _status_text(output.success)
        lines.extend(
            [
                "",
                f"[{offset}] {output.name or '<unknown_tool>'} call_id={output.call_id or '<missing>'} status={status}",
                "arguments:",
                _clip(call.arguments if call else "", 3000) or "<not available>",
                "",
                "result:",
                _result_header(output),
                "Key output:",
                _clip(output.stdout_excerpt or output.output, 12000),
            ]
        )
        if output.stderr_excerpt:
            lines.extend(["", "Key error lines:", _clip(output.stderr_excerpt, 3000)])

    if history.failed_outputs:
        lines.append("")
        lines.append("Failed calls:")
        for output in history.failed_outputs[-5:]:
            lines.append(f"- {output.call_id or '<missing>'} {output.name or '<unknown_tool>'}: {_one_line(output.stderr_excerpt or output.stdout_excerpt or output.output, 240)}")

    if history.pending_calls:
        lines.extend(["", "Pending local tool calls without outputs:"])
        for call in history.pending_calls[-8:]:
            lines.append(f"- {call.call_id} {call.name} {_one_line(call.arguments, 260)}")
        lines.append("Do not assume their result.")

    repeated = _repeated_successful_calls(history)
    if repeated:
        lines.extend(["", "Recently repeated calls:"])
        for call in repeated[:5]:
            lines.append(f"- {call.name} {_one_line(call.arguments, 260)} already succeeded.")
        lines.append("Do not call them again unless arguments change.")

    exposed = summarize_exposed_deferred_tools(history.exposed_deferred_tools)
    if exposed:
        lines.extend(["", "Tool search exposed these local tools:"])
        for name in exposed[:30]:
            lines.append(f"- {name}")
        lines.append("These tools are now callable by exact name. Prefer them over another tool_search if suitable.")

    lines.extend(
        [
            "",
            "Decision guidance:",
            "- If the tool output fully answers the user, return mode=answer.",
            "- If more local action is needed, return mode=tool_call.",
            "- If a tool failed, inspect the error and either retry with corrected arguments or explain the blocker.",
        ]
    )
    return "\n".join(lines)


def is_repeated_call(name: str, arguments: str, recent_calls: list[ToolCallRecord]) -> bool:
    current = (str(name or "").lower(), _normalize_arguments(arguments))
    matches = 0
    for call in recent_calls:
        if (call.name.lower(), _normalize_arguments(call.arguments)) == current:
            matches += 1
    return matches > 1


def is_repeated_successful_call(name: str, arguments: str, history: ToolHistory | None) -> bool:
    if not history:
        return False
    current = (str(name or "").lower(), _normalize_arguments(arguments))
    successful_ids = {out.call_id for out in history.successful_outputs}
    for call in history.calls:
        if call.call_id not in successful_ids:
            continue
        if (call.name.lower(), _normalize_arguments(call.arguments)) == current:
            return True
    return False


def summarize_exposed_deferred_tools(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []

    def add(name: str) -> None:
        if name and name not in names:
            names.append(name)

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        typ = str(tool.get("type") or "")
        name = str(tool.get("name") or "")
        if typ == "namespace":
            for sub in tool.get("tools") or []:
                if isinstance(sub, dict):
                    sub_name = str(sub.get("name") or "")
                    add(f"{name}.{sub_name}" if name and sub_name else sub_name)
            continue
        add(name or typ)
    return names


def _parse_call(item: dict[str, Any], index: int) -> ToolCallRecord | None:
    call_id = str(item.get("call_id") or item.get("id") or "")
    typ = str(item.get("type") or "")
    name = _call_name(item, typ)
    arguments = _call_arguments(item, typ)
    if not call_id and not name:
        return None
    return ToolCallRecord(call_id=call_id, name=name, call_type=typ, arguments=arguments, index=index)


def _parse_output(item: dict[str, Any], index: int, calls_by_id: dict[str, ToolCallRecord]) -> ToolOutputRecord | None:
    call_id = str(item.get("call_id") or item.get("id") or "")
    call = calls_by_id.get(call_id)
    typ = str(item.get("type") or "")
    name = str(item.get("name") or (call.name if call else "") or _output_default_name(typ))
    output_text = _output_to_text(item)
    exit_code = _extract_exit_code(output_text)
    success = _infer_success(item, output_text, exit_code)
    stderr_excerpt = _stderr_excerpt(output_text)
    stdout_excerpt = _stdout_excerpt(output_text)
    return ToolOutputRecord(
        call_id=call_id,
        name=name,
        call_type=typ,
        output=output_text,
        exit_code=exit_code,
        success=success,
        stderr_excerpt=stderr_excerpt,
        stdout_excerpt=stdout_excerpt,
        raw_chars=len(output_text),
        index=index,
    )


def _call_name(item: dict[str, Any], typ: str) -> str:
    if isinstance(item.get("name"), str):
        return item["name"]
    if typ == "tool_search_call":
        return "tool_search"
    if typ == "web_search_call":
        return "web_search"
    if typ == "computer_call":
        return "computer"
    return ""


def _call_arguments(item: dict[str, Any], typ: str) -> str:
    if typ == "custom_tool_call":
        value = item.get("input", item.get("arguments", ""))
    elif typ == "web_search_call":
        value = item.get("action", item.get("arguments", {}))
    elif typ == "computer_call":
        value = item.get("action", item.get("arguments", {}))
    else:
        value = item.get("arguments", item.get("input", ""))
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _output_default_name(typ: str) -> str:
    if typ == "tool_search_output":
        return "tool_search"
    if typ == "computer_call_output":
        return "computer"
    if typ == "web_search_output":
        return "web_search"
    return ""


def _output_to_text(item: dict[str, Any]) -> str:
    if "output" in item:
        value = item.get("output")
    elif "tools" in item:
        value = {"tools": item.get("tools")}
    elif "result" in item:
        value = item.get("result")
    else:
        value = {k: v for k, v in item.items() if k not in {"type", "call_id", "id"}}
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _extract_exit_code(text: str) -> int | None:
    match = EXIT_CODE_RE.search(text or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _infer_success(item: dict[str, Any], text: str, exit_code: int | None) -> bool | None:
    status = str(item.get("status") or "").lower()
    if status in {"completed", "success", "succeeded"}:
        if exit_code is None or exit_code == 0:
            return True
    if status in {"failed", "error", "incomplete", "cancelled", "canceled"}:
        return False
    if exit_code is not None:
        return exit_code == 0
    stripped = (text or "").strip()
    if stripped.startswith("Success."):
        return True
    if stripped and ERROR_LINE_RE.search(stripped):
        return False
    return None


def _stderr_excerpt(text: str, max_lines: int = 40) -> str:
    lines = [line for line in (text or "").splitlines() if ERROR_LINE_RE.search(line)]
    return "\n".join(lines[-max_lines:])


def _stdout_excerpt(text: str, max_lines: int = 100, max_chars: int = 12000) -> str:
    lines = (text or "").splitlines()
    tail = "\n".join(lines[-max_lines:]) if lines else (text or "")
    return _clip(tail, max_chars)


def _result_header(output: ToolOutputRecord) -> str:
    parts: list[str] = []
    if output.exit_code is not None:
        parts.append(f"Exit code: {output.exit_code}")
    wall = WALL_TIME_RE.search(output.output or "")
    if wall:
        parts.append(f"Wall time: {wall.group(1).strip()}")
    parts.append(f"Raw output chars: {output.raw_chars}")
    return "\n".join(parts)


def _find_call(calls: list[ToolCallRecord], call_id: str) -> ToolCallRecord | None:
    for call in reversed(calls):
        if call.call_id == call_id:
            return call
    return None


def _status_text(success: bool | None) -> str:
    if success is True:
        return "success"
    if success is False:
        return "failed"
    return "unknown"


def _extract_deferred_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [tool for tool in value if isinstance(tool, dict)][-50:]


def _repeated_successful_calls(history: ToolHistory) -> list[ToolCallRecord]:
    successful_ids = {out.call_id for out in history.successful_outputs}
    seen: dict[tuple[str, str], ToolCallRecord] = {}
    repeated: list[ToolCallRecord] = []
    for call in history.calls:
        key = (call.name.lower(), _normalize_arguments(call.arguments))
        if key in seen and (call.call_id in successful_ids or seen[key].call_id in successful_ids):
            repeated.append(call)
        else:
            seen[key] = call
    return repeated


def _normalize_arguments(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return re.sub(r"\s+", " ", text)


def _clip(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head
    return text[:head] + f"\n...[truncated {len(text) - max_chars} chars]...\n" + text[-tail:]


def _one_line(text: str, max_chars: int) -> str:
    return _clip(re.sub(r"\s+", " ", str(text or "")).strip(), max_chars)
