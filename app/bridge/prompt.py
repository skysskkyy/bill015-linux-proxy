from __future__ import annotations

from ..config import Settings, settings
from ..ingest.catalog import catalog_json
from ..protocol.models import Turn


def build_instructions(turn: Turn, cfg: Settings = settings) -> str:
    name = cfg.function_name
    parts = [
        "You are talking to Codex Desktop through a local transport wrapper.",
        f"You MUST call the function `{name}` to do anything. Never write a normal assistant message.",
        "You MAY call that function more than once in this turn when you have independent local actions.",
        f"Direct final answer: mode='answer', {cfg.answer_field}=<text>, tool_calls=[].",
        "Need local tools: mode='tool_call', answer='' unless a short commentary is genuinely useful, tool_calls=[...].",
        "Function tools: arguments is a JSON string matching the tool schema.",
        "Custom/freeform tools (apply_patch, grammar exec): put the raw payload in input, arguments='{}'.",
        "tool_search: arguments is a JSON object {query, limit}, execution is client-side.",
        "Prefer apply_patch for text/file edits. Use shell/exec for inspect/build/test.",
        "Do not invent tool names. If a needed tool is deferred, call tool_search first.",
        "This wrapper is transport-only. Do not reduce planning, inspection, or validation quality.",
    ]
    if turn.is_compaction:
        parts.append(
            "COMPACTION MODE: produce a handoff summary covering the goal, files touched, last tool outcomes, and open questions. "
            "mode='answer'. Do not request tools."
        )
    catalog_blob = catalog_json(turn.catalog, cfg)
    if catalog_blob and catalog_blob != "[]":
        parts.append("Callable tools this turn (use exact names/namespaces):")
        parts.append(catalog_blob)
        if turn.catalog.deferred:
            preview = ", ".join(turn.catalog.deferred[:20])
            parts.append(
                f"[LOCAL TOOL SELECTION] {len(turn.catalog.selected)} tools are callable; "
                f"{len(turn.catalog.deferred)} are deferred. Use tool_search for deferred capabilities. Examples: {preview}."
            )
    else:
        parts.append("No concrete local tool catalog was sent this turn. Do not invent tools; answer or wait.")
    if turn.web_intent:
        parts.append(
            "This turn needs live web/URL information. Use a local browser/Chrome/Playwright/node_repl/HTTP tool. "
            "Do not assume a hosted web_search exists. If only tool_search is visible, call it first."
        )
    if turn.loss_notices:
        parts.extend(turn.loss_notices)
    if turn.developer_text:
        parts.append("Developer context:\n" + turn.developer_text[:8000])
    if turn.system_text:
        parts.append("System context:\n" + turn.system_text[:4000])
    if turn.instructions:
        parts.append("Client instructions:\n" + turn.instructions[:4000])
    return "\n\n".join(parts)
