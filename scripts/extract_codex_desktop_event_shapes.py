from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "proxy_evidence" / "native_event_samples"
SSE_PREFIX = "SSE event:"
FORBIDDEN_RESPONSE_FIELDS = {
    "safety_identifier",
    "rate_limits",
    "code_review_rate_limits",
    "additional_rate_limits",
    "credits",
    "promo",
    "openai_verification_recommendation",
}
OBSERVED_OPTIONAL_EXTENSIONS = [
    "codex.rate_limits",
    "keepalive",
    "response.metadata",
]


def default_sqlite_path() -> Path | None:
    candidates: list[Path] = []
    home = Path.home()
    candidates.append(home / ".codex" / "logs_2.sqlite")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Codex" / "logs_2.sqlite")
        candidates.append(Path(local) / "OpenAI" / "Codex" / "logs_2.sqlite")
    for path in candidates:
        if path.exists():
            return path
    return None


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def string_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def redact_value(value: Any, *, include_preview: bool = False) -> Any:
    if isinstance(value, str):
        out: dict[str, Any] = {"type": "str", "len": len(value), "sha256_16": string_digest(value)}
        if include_preview:
            preview = value.replace("\r", "\\r").replace("\n", "\\n")
            if len(preview) > 48:
                preview = preview[:24] + "…" + preview[-16:]
            out["preview"] = preview
        return out
    if isinstance(value, list):
        return {"type": "array", "len": len(value), "items": [redact_value(v, include_preview=include_preview) for v in value[:3]]}
    if isinstance(value, dict):
        return {str(k): redact_value(v, include_preview=include_preview) for k, v in sorted(value.items())}
    return value


def load_events(sqlite_path: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    con = sqlite3.connect(str(sqlite_path))
    try:
        rows = con.execute(
            """
            select id, ts, ts_nanos, feedback_log_body from logs
            where target='codex_api::sse::responses'
              and feedback_log_body like 'SSE event:%'
            order by id desc
            limit ?
            """,
            (limit,),
        ).fetchall()
    finally:
        con.close()
    events: list[dict[str, Any]] = []
    parse_failures = 0
    for row_id, ts, ts_nanos, body in rows:
        raw = str(body)[len(SSE_PREFIX) :].strip()
        try:
            obj = json.loads(raw)
        except Exception:
            parse_failures += 1
            continue
        if isinstance(obj, dict):
            obj["__log_id"] = row_id
            obj["__log_ts"] = ts
            obj["__log_ts_nanos"] = ts_nanos
            events.append(obj)
        else:
            parse_failures += 1
    # Restore chronological order after querying latest-first.
    events.reverse()
    return events, parse_failures


def build_matrix(events: list[dict[str, Any]], *, include_preview: bool = False) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    total = 0
    for obj in events:
        obj = {k: v for k, v in obj.items() if not k.startswith("__log_")}
        typ = str(obj.get("type") or "<no_type>")
        total += 1
        rec = by_type.setdefault(
            typ,
            {
                "count": 0,
                "fields": {},
                "keysets": collections.Counter(),
                "examples_redacted": [],
                "required_fields": [],
                "optional_fields": [],
            },
        )
        rec["count"] += 1
        rec["keysets"][tuple(sorted(obj.keys()))] += 1
        for key, value in obj.items():
            f = rec["fields"].setdefault(key, {"count": 0, "types": collections.Counter()})
            f["count"] += 1
            f["types"][type_name(value)] += 1
        if len(rec["examples_redacted"]) < 3:
            rec["examples_redacted"].append(redact_value(obj, include_preview=include_preview))

    for _typ, rec in by_type.items():
        count = max(1, int(rec["count"]))
        fields_out = {}
        required: list[str] = []
        optional: list[str] = []
        for field, f in sorted(rec["fields"].items()):
            field_count = int(f["count"])
            ratio = field_count / count
            types = dict(sorted(f["types"].items()))
            fields_out[field] = {"count": field_count, "ratio": ratio, "types": types}
            if ratio >= 0.95 or field in {"type", "sequence_number", "response", "item", "output_index", "item_id", "content_index", "delta"}:
                required.append(field)
            else:
                optional.append(field)
        rec["fields"] = fields_out
        rec["keysets"] = [{"count": n, "keys": list(keys)} for keys, n in rec["keysets"].most_common(12)]
        rec["required_fields"] = sorted(set(required))
        rec["optional_fields"] = sorted(set(optional))

    return {
        "generated_at": int(time.time()),
        "source": "Codex Desktop logs_2.sqlite target=codex_api::sse::responses",
        "total_events": total,
        "event_types": by_type,
        "observed_optional_extensions": OBSERVED_OPTIONAL_EXTENSIONS,
        "forbidden_response_fields": sorted(FORBIDDEN_RESPONSE_FIELDS),
        "redaction": "strings are stored as length and sha256_16 only unless --include-preview is used",
    }


def write_outputs(matrix: dict[str, Any], events: list[dict[str, Any]], out_dir: Path, *, include_preview: bool, sqlite_path: Path, parse_failures: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "field_matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "events_redacted.jsonl").open("w", encoding="utf-8") as fh:
        for obj in events:
            public = {k: v for k, v in obj.items() if not k.startswith("__log_")}
            fh.write(json.dumps(redact_value(public, include_preview=include_preview), ensure_ascii=False, separators=(",", ":")) + "\n")

    lines = [
        "# Codex Desktop Responses SSE field matrix",
        "",
        f"- Source: `{sqlite_path}`",
        f"- Events parsed: {matrix['total_events']}",
        f"- Parse failures: {parse_failures}",
        f"- Redaction: {matrix['redaction']}",
        "",
        "## Event types",
        "",
    ]
    for typ, rec in sorted(matrix["event_types"].items(), key=lambda kv: (-kv[1]["count"], kv[0])):
        lines.append(f"- `{typ}` count={rec['count']} required={','.join(rec['required_fields']) or '-'} optional={','.join(rec['optional_fields']) or '-'}")
    lines.extend(
        [
            "",
            "## Optional Codex Desktop extensions observed",
            "",
            *[f"- `{name}`" for name in matrix["observed_optional_extensions"]],
            "",
            "## Forbidden local synthesis fields",
            "",
            *[f"- `{name}`" for name in matrix["forbidden_response_fields"]],
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract redacted Codex Desktop Responses SSE event field shapes from logs_2.sqlite.")
    parser.add_argument("--sqlite", dest="sqlite_path", default="", help="Path to logs_2.sqlite. Defaults to ~/.codex/logs_2.sqlite.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for field_matrix.json, events_redacted.jsonl, summary.md.")
    parser.add_argument("--limit", type=int, default=20000, help="Maximum latest SSE events to inspect.")
    parser.add_argument("--include-preview", action="store_true", help="Include short redacted string previews in addition to len/hash.")
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else default_sqlite_path()
    if not sqlite_path or not sqlite_path.exists():
        raise SystemExit("logs_2.sqlite not found; pass --sqlite PATH")
    events, parse_failures = load_events(sqlite_path, max(1, args.limit))
    if not events:
        raise SystemExit(f"no Codex Desktop Responses SSE events found in {sqlite_path}")
    matrix = build_matrix(events, include_preview=args.include_preview)
    write_outputs(matrix, events, Path(args.out), include_preview=args.include_preview, sqlite_path=sqlite_path, parse_failures=parse_failures)
    print(f"[extract] source={sqlite_path}")
    print(f"[extract] events={matrix['total_events']} parse_failures={parse_failures} out={Path(args.out)}")
    print("[extract] wrote field_matrix.json events_redacted.jsonl summary.md")


if __name__ == "__main__":
    main()
