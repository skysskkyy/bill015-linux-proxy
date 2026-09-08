from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["LOCAL_PROXY_MODE"] = "dry-run"
os.environ["LOCAL_PROXY_LOG_DIR"] = str(ROOT / "proxy_evidence" / "selftest")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    client = TestClient(app)
    r = client.get("/healthz")
    assert_true(r.status_code == 200, "/healthz failed")
    print("[ok] /healthz", r.json()["mode"])

    r = client.get("/v1/models")
    assert_true(r.status_code == 200 and r.json()["data"], "/v1/models failed")
    print("[ok] /v1/models", [x["id"] for x in r.json()["data"]])

    body = {"model": "codex-gpt55", "input": "Return the word OK.", "stream": False}
    r = client.post("/v1/responses", json=body)
    assert_true(r.status_code == 200, "/v1/responses dry-run failed")
    data = r.json()
    assert_true(data.get("payload", {}).get("tool_choice", {}).get("name") == "emit_value", "dry-run payload wrong")
    assert_true(data.get("payload", {}).get("parallel_tool_calls") is True, "parallel emit_value not enabled")
    print("[ok] /v1/responses dry-run")

    r = client.post("/v1/responses/compact", json={"model": "gpt-5.5", "input": "summarize the session"})
    assert_true(r.status_code == 200 and "output" in r.json(), "compact failed")
    print("[ok] /v1/responses/compact")
    print("selftest passed")


if __name__ == "__main__":
    main()
