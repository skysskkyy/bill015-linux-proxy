from __future__ import annotations

from ..config import Settings, settings
from .tokens import threshold_tokens

# Official API total context (input and output share this budget).
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-6-astra": 1_050_000,
    "gpt-6": 1_050_000,
    "gpt-5.6-sol": 1_050_000,
    "gpt-5.6-terra": 1_050_000,
    "gpt-5.6-luna": 1_050_000,
    "gpt-5.6": 1_050_000,
    "gpt-5.5": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.1-codex-max": 1_050_000,
    "gpt-5.1-codex": 272_000,
    "gpt-5.1": 400_000,
    "gpt-5": 400_000,
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
}


def _lookup(table: dict[str, int], model: str) -> int | None:
    name = (model or "").strip().lower()
    if not name:
        return None
    if name in table:
        return table[name]
    best: int | None = None
    best_len = -1
    for key, value in table.items():
        if name.startswith(key) and len(key) > best_len:
            best = value
            best_len = len(key)
    return best


def model_context_window(model: str | None, cfg: Settings = settings) -> int:
    name = (model or cfg.default_model or "").strip()
    mapped = cfg.map_model(name)
    override = _lookup({str(k).lower(): int(v) for k, v in (cfg.model_context_windows or {}).items() if str(v).strip()}, mapped)
    if override:
        return max(16_000, override)
    found = _lookup(MODEL_CONTEXT_WINDOWS, mapped)
    if found:
        return found
    return max(16_000, cfg.context_window_tokens)


def compact_threshold_tokens(model: str | None, cfg: Settings = settings) -> int:
    window = model_context_window(model, cfg)
    return threshold_tokens(window, cfg.auto_compact_percent)
