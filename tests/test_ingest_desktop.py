from __future__ import annotations

from fastapi.testclient import TestClient

from app.ingest import normalize_responses_request
from app.ingest.catalog import build_catalog
from app.main import app


def test_additional_tools_populate_catalog():
    body = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {"type": "function", "name": "shell_command", "description": "run a shell command"},
                    {"type": "custom", "name": "apply_patch", "description": "FREEFORM patch"},
                    {"type": "tool_search", "name": "tool_search", "description": "discover tools"},
                ],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "patch README"}]},
        ],
    }
    turn = normalize_responses_request(body)
    assert turn.responses_lite is True
    names = {spec.name for spec in turn.catalog.specs.values()}
    assert {"shell_command", "apply_patch", "tool_search"} <= names
    assert turn.catalog.specs["apply_patch"].call_type == "custom"
    assert turn.catalog.specs["tool_search"].call_type == "tool_search"


def test_tool_search_output_extends_catalog():
    items = [
        {"type": "message", "role": "user", "content": "open a thread"},
        {
            "type": "tool_search_output",
            "call_id": "search-1",
            "status": "completed",
            "execution": "client",
            "tools": [{"type": "function", "name": "create_thread", "description": "desktop thread tool"}],
        },
    ]
    catalog = build_catalog([], items, query="open a thread")
    assert "create_thread" in catalog.specs


def test_compact_endpoint_returns_output_items(monkeypatch):
    monkeypatch.setattr("app.config.settings.mode", "dry-run")
    client = TestClient(app)
    r = client.post("/v1/responses/compact", json={"model": "gpt-5.5", "input": "summarize"})
    assert r.status_code == 200
    data = r.json()
    assert "output" in data
    assert data["output"][0]["type"] == "message"
    assert data["output"][0]["role"] == "assistant"
