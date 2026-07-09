import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_selftest_script():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "selftest.py")], cwd=str(ROOT), text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_audit_from_result_accepts_tool_calls():
    from app.models import Bill015Result, BridgeToolCall, NormalizedRequest
    from app.upstream import audit_from_result

    request = NormalizedRequest(
        model="gpt-test",
        instructions="",
        user_input="hello",
        want_stream=False,
        client_api="responses",
        is_primary_path=True,
    )
    result = Bill015Result(
        local_request_id="resp_local_test",
        bridge_mode="tool_call",
        tool_calls=[BridgeToolCall(id="call_test", name="shell_command", arguments='{"command":"echo hi"}')],
    )

    record = audit_from_result(result, request, "exploit")

    assert record["tool_calls"][0]["name"] == "shell_command"
    assert record["repeated_tool_call_detected"] is False


def test_chat_json_includes_tool_calls():
    from app.models import Bill015Result, BridgeToolCall, NormalizedRequest
    from app.response_events import chat_json

    request = NormalizedRequest(
        model="gpt-test",
        instructions="",
        user_input="hello",
        want_stream=False,
        client_api="chat.completions",
        is_primary_path=False,
    )
    result = Bill015Result(
        local_request_id="resp_local_test",
        bridge_mode="tool_call",
        tool_calls=[BridgeToolCall(id="call_test", name="shell_command", arguments='{"command":"echo hi"}')],
    )

    body = chat_json(result, request)
    message = body["choices"][0]["message"]

    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["id"] == "call_test"
    assert message["tool_calls"][0]["function"]["name"] == "shell_command"


def test_cors_defaults_are_local_only():
    from app.config import settings

    assert settings.cors_allow_origins == []
    assert "localhost" in settings.cors_allow_origin_regex
    assert "127" in settings.cors_allow_origin_regex


def test_response_annotations_setting_controls_field(monkeypatch):
    from app.config import settings
    from app.models import Bill015Result, NormalizedRequest
    from app.response_events import response_json

    request = NormalizedRequest(
        model="gpt-test",
        instructions="",
        user_input="hello",
        want_stream=False,
        client_api="responses",
        is_primary_path=True,
    )
    result = Bill015Result(local_request_id="resp_local_test", answer="ok")

    monkeypatch.setattr(settings, "responses_emit_annotations", False)
    part_without = response_json(result, request)["output"][0]["content"][0]
    assert "annotations" not in part_without

    monkeypatch.setattr(settings, "responses_emit_annotations", True)
    part_with = response_json(result, request)["output"][0]["content"][0]
    assert part_with["annotations"] == []
