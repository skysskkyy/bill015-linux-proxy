from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_healthz_and_models():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "safe_bridge" in body
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [row["id"] for row in r.json()["data"]]
    assert settings.default_model in ids


def test_dry_run_responses(monkeypatch):
    monkeypatch.setattr(settings, "mode", "dry-run")
    client = TestClient(app)
    r = client.post("/v1/responses", json={"model": "codex-gpt55", "input": "Return the word OK.", "stream": False})
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "dry-run"
    assert data["payload"]["tool_choice"]["name"] == "emit_value"
    assert data["payload"]["parallel_tool_calls"] is True
    assert data["payload"]["stream"] is True
    names = [tool.get("name") for tool in data["payload"].get("tools", [])]
    blob = json.dumps(data["payload"])
    assert "web_search" in blob
    assert "web_extract" in blob
    assert names == ["emit_value"]


def test_dry_run_rewrites_agent_message_encrypted_content(monkeypatch):
    monkeypatch.setattr(settings, "mode", "dry-run")
    client = TestClient(app)
    r = client.post(
        "/v1/responses",
        json={
            "model": "gpt-6-astra",
            "stream": False,
            "input": [
                {
                    "type": "agent_message",
                    "id": "amsg_1",
                    "author": "/root",
                    "recipient": "/root/child",
                    "content": [
                        {"type": "input_text", "text": "Payload:\n"},
                        {"type": "encrypted_content", "encrypted_content": "Do the child task."},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 200
    payload = r.json()["payload"]
    agent = next(item for item in payload["input"] if item.get("type") == "agent_message")
    assert agent["content"][1] == {"type": "input_text", "text": "Do the child task."}
    assert '"type": "encrypted_content"' not in json.dumps(payload["input"])


def test_upstream_headers_look_like_codex():
    from app.upstream.client import NATIVE_ORIGINATOR, NATIVE_USER_AGENT, upstream_auth_headers

    headers = upstream_auth_headers(
        api_key="sk-test",
        client_headers={
            "Authorization": "Bearer client-secret",
            "Cookie": "session=abc",
            "session-id": "sess_1",
            "thread-id": "thr_1",
            "originator": "codex_cli_rs",
            "User-Agent": "codex_cli_rs/0.153.4 (Linux 6.18.33; x86_64) rust",
            "x-codex-turn-metadata": "{}",
            "x-openai-internal-codex-responses-lite": "true",
        },
    )
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["user-agent"].startswith("codex_cli_rs/")
    assert headers["originator"] == "codex_cli_rs"
    assert headers["session-id"] == "sess_1"
    assert headers["thread-id"] == "thr_1"
    assert "x-codex-turn-metadata" in headers
    assert "x-openai-internal-codex-responses-lite" not in headers
    assert "cookie" not in {k.lower() for k in headers if k.lower() == "cookie"} or headers.get("cookie") != "session=abc"
    fallback = upstream_auth_headers(api_key="sk-test", client_headers={})
    assert fallback["user-agent"] == NATIVE_USER_AGENT
    assert fallback["originator"] == NATIVE_ORIGINATOR


def test_dry_run_instructions_are_not_relay_fingerprints(monkeypatch):
    monkeypatch.setattr(settings, "mode", "dry-run")
    client = TestClient(app)
    r = client.post("/v1/responses", json={"model": "gpt-6-astra", "input": "Return OK.", "stream": False})
    assert r.status_code == 200
    blob = str(r.json()["payload"]).lower()
    assert "transport wrapper" not in blob
    assert "bill-015" not in blob
    assert "secondary distribution" not in blob


def test_block_normal_mode(monkeypatch):
    monkeypatch.setattr(settings, "mode", "normal")
    monkeypatch.setattr(settings, "block_normal_mode", True)
    client = TestClient(app)
    r = client.post("/v1/responses", json={"model": "gpt-5.5", "input": "hi", "stream": False})
    assert r.status_code == 409


def test_image_reject_default(monkeypatch):
    monkeypatch.setattr(settings, "multimodal_strategy", "reject")
    monkeypatch.setattr(settings, "mode", "dry-run")
    client = TestClient(app)
    r = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.5",
            "stream": False,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,aaa"}]}],
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["message"]["code"] == "local_proxy_vision_unsupported"


def test_image_passthrough_forwards_input_image(monkeypatch):
    monkeypatch.setattr(settings, "multimodal_strategy", "native_passthrough")
    monkeypatch.setattr(settings, "mode", "dry-run")
    monkeypatch.setattr(settings, "store_prompts", True)
    client = TestClient(app)
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    r = client.post(
        "/v1/responses",
        json={
            "model": "gpt-6-astra",
            "stream": False,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{png}"},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 200
    payload = r.json()["payload"]
    assert png in str(payload["input"])
    assert "LOCAL VISION NOTICE" not in str(payload["input"])
