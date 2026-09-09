from __future__ import annotations

from app.context.windows import compact_threshold_tokens, model_context_window


def test_official_windows_differ_by_model(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "model_context_windows", {})
    monkeypatch.setattr(settings, "auto_compact_percent", 90)
    assert model_context_window("gpt-6-astra") == 1_050_000
    assert model_context_window("gpt-5.5") == 1_050_000
    assert model_context_window("gpt-5.1-codex") == 272_000
    assert compact_threshold_tokens("gpt-6-astra") == int(1_050_000 * 0.90)


def test_config_window_override(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "model_context_windows", {"gpt-6-astra": 272000})
    assert model_context_window("gpt-6-astra") == 272000
