from __future__ import annotations

import re

from ..protocol.models import TurnResult

PROGRESS_START = re.compile(
    r"(?is)^\s*(?:我先|我继续|我来先|让我先|接下来我(?:将)?|"
    r"(?:i(?:['’]ll| will)|let me|i am going to|i['’]m going to)\b)"
)
PROGRESS_FUTURE = re.compile(
    r"(?is)(再(?:给你|整理|检查|说明)|然后(?:再)?(?:给你|整理|检查|说明)|"
    r"then (?:i(?:['’]ll| will)|give you|share|summarize|provide)|"
    r"and then (?:i(?:['’]ll| will)|give you))"
)
PROGRESS_WORK = re.compile(r"(?is)(检查|查看|inspect|look (?:at|through)|scan|read files)")
FINDINGS = re.compile(
    r"(?is)(```|/home/|/mnt/|\bREADME\b|\.py\b|\.ts\b|\.rs\b|结论|概览如下|项目用途是|this project)"
)


def looks_like_progress(text: str) -> bool:
    blob = (text or "").strip()
    if not blob or len(blob) > 400:
        return False
    if FINDINGS.search(blob):
        return False
    if not PROGRESS_START.search(blob):
        return False
    return bool(PROGRESS_FUTURE.search(blob) or PROGRESS_WORK.search(blob))


def should_continue_progress(result: TurnResult) -> bool:
    if result.tool_calls:
        return False
    if (result.answer or "").strip():
        return False
    return bool((result.commentary or "").strip())
