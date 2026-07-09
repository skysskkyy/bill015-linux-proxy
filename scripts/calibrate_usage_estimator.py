from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LOCAL_PROXY_MODE", "dry-run")
os.environ.setdefault("LOCAL_PROXY_LOG_DIR", str(ROOT / "proxy_evidence" / "calibrate_usage"))

from app.usage_estimator import estimate_responses_usage_from_body  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def request_paths(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    patterns = ["request_*.json", "compaction_request_*.json", "*.json"]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(base.glob(pattern)))
        if found:
            break
    return [p for p in found if not p.name.startswith("completed_")]


def completed_paths(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    return sorted(base.glob("completed_*.json"))


def native_usage_from_completed(path: Path) -> dict[str, Any] | None:
    try:
        obj = read_json(path)
    except Exception:
        return None
    if isinstance(obj, dict):
        if isinstance(obj.get("usage"), dict):
            return obj["usage"]
        response = obj.get("response")
        if isinstance(response, dict) and isinstance(response.get("usage"), dict):
            return response["usage"]
    return None


def numeric_id(path: Path) -> str | None:
    match = re.search(r"(\d+)", path.stem)
    return match.group(1) if match else None


def pct_error(estimated: int, native: int) -> float:
    if native <= 0:
        return 0.0
    return (estimated - native) * 100.0 / native


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local usage estimates against captured native response.completed usage.")
    parser.add_argument(
        "--requests",
        default=str(ROOT / "proxy_evidence" / "native_usage_compaction"),
        help="Directory or JSON file with native request bodies.",
    )
    parser.add_argument(
        "--completed",
        default=str(ROOT / "proxy_evidence" / "native_usage_compaction"),
        help="Directory or JSON file with response.completed JSON files.",
    )
    args = parser.parse_args()

    reqs = request_paths(Path(args.requests))
    native_by_id: dict[str, dict[str, Any]] = {}
    for completed in completed_paths(Path(args.completed)):
        ident = numeric_id(completed)
        usage = native_usage_from_completed(completed)
        if ident and usage:
            native_by_id[ident] = usage
    errors: list[float] = []

    print(f"[calibrate] requests={len(reqs)} native_completed_usages={len(native_by_id)}")
    for path in reqs:
        try:
            body = read_json(path)
            if isinstance(body, dict) and isinstance(body.get("request"), dict):
                body = body["request"]
            if not isinstance(body, dict):
                raise ValueError("request JSON is not an object")
            estimate = estimate_responses_usage_from_body(body)
            native = native_by_id.get(numeric_id(path) or "")
            if native and int(native.get("input_tokens") or 0) > 0:
                native_input = int(native.get("input_tokens") or 0)
                err = pct_error(estimate.input_tokens, native_input)
                errors.append(err)
                print(f"{path.name}: native input_tokens={native_input} estimated={estimate.input_tokens} error={err:+.2f}%")
            else:
                print(f"{path.name}: estimated input_tokens={estimate.input_tokens} cached={estimate.cached_tokens} method={estimate.estimate_method}")
        except Exception as exc:
            print(f"{path.name}: ERROR {type(exc).__name__}: {exc}")

    if errors:
        avg_abs = sum(abs(x) for x in errors) / len(errors)
        print(f"[calibrate] paired={len(errors)} avg_abs_error={avg_abs:.2f}%")
        if avg_abs > 15:
            print("[calibrate] recommendation: tune heuristic divisors or prefer installing tiktoken for this environment.")
    else:
        print("[calibrate] no paired nonzero native usage found; estimates printed for baseline inspection.")


if __name__ == "__main__":
    main()
