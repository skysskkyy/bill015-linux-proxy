from __future__ import annotations

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
