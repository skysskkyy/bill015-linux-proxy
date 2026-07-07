from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LOCAL_PROXY_MODE", "dry-run")
os.environ.setdefault("LOCAL_PROXY_LOG_DIR", str(ROOT / "proxy_evidence" / "replay"))

from app.proxy import normalize_responses_request  # noqa: E402


def load_request(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(obj, dict) and isinstance(obj.get("request"), dict):
        obj = obj["request"]
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def iter_paths(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    return sorted(base.glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay native Codex request logs through normalize_responses_request().")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(ROOT / "proxy_evidence" / "native_long_chat" / "extracted"),
        help="JSON file or directory containing native request JSON files.",
    )
    args = parser.parse_args()

    paths = iter_paths(Path(args.path))
    if not paths:
        raise SystemExit(f"no JSON files found under {args.path}")

    ok = 0
    with_tools = 0
    with_feedback = 0
    compactions = 0
    deferred = 0
    nonzero_usage = 0
    compaction_cached_ok = 0
    max_input_tokens = 0
    failures: list[str] = []

    for path in paths:
        try:
            body = load_request(path)
            n = normalize_responses_request(body)
            ok += 1
            if n.usage_estimate and n.usage_estimate.input_tokens > 0:
                nonzero_usage += 1
                max_input_tokens = max(max_input_tokens, n.usage_estimate.input_tokens)
            elif body.get("input"):
                raise AssertionError("usage_estimate.input_tokens is zero")
            if n.tool_history and (n.tool_history.calls or n.tool_history.outputs):
                with_tools += 1
                if n.latest_tool_summary:
                    with_feedback += 1
            if n.is_compaction:
                compactions += 1
                if n.usage_estimate.cached_tokens >= int(n.usage_estimate.input_tokens * 0.80):
                    compaction_cached_ok += 1
                else:
                    raise AssertionError("compaction cached_tokens ratio is too low")
            if n.tool_history and n.tool_history.exposed_deferred_tools:
                deferred += 1
                if not n.tools_catalog:
                    raise AssertionError("deferred tools found but tools_catalog is empty")
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")

    print(f"[replay] files={len(paths)} ok={ok} failed={len(failures)}")
    print(f"[replay] tool_histories={with_tools} feedback={with_feedback} compactions={compactions} deferred_tool_outputs={deferred}")
    print(f"[replay] usage_nonzero={nonzero_usage} max_input_tokens={max_input_tokens} compaction_cached_ok={compaction_cached_ok}")
    if failures:
        print("[replay] failures:")
        for item in failures[:20]:
            print(f"- {item}")
        raise SystemExit(1)
    print("[replay] native log replay passed")


if __name__ == "__main__":
    main()
