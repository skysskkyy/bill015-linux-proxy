from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal

Mode = Literal["exploit", "verify", "normal", "dry-run"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.local.json"


def _load_local_config() -> dict[str, Any]:
    # Default behavior: read local project config. An env override is kept only for tests/advanced use.
    raw_path = os.getenv("BILL015_CONFIG_PATH", "")
    path = Path(raw_path) if raw_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


LOCAL_CONFIG = _load_local_config()


def _cfg(path: str, default: Any = None) -> Any:
    cur: Any = LOCAL_CONFIG
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _env_str(name: str, cfg_path: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is not None:
        return raw
    val = _cfg(cfg_path, default)
    return default if val is None else str(val)


def _env_bool(name: str, cfg_path: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        raw = _cfg(cfg_path, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, cfg_path: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        raw = _cfg(cfg_path, default)
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, cfg_path: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        raw = _cfg(cfg_path, default)
    try:
        return float(raw)
    except Exception:
        return default


def _cfg_dict(path: str, default: Dict[str, str]) -> Dict[str, str]:
    val = _cfg(path, default)
    if isinstance(val, dict):
        return {str(k): str(v) for k, v in val.items()}
    return dict(default)


@dataclass
class Settings:
    config_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)
    host: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_HOST", "server.host", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_PORT", "server.port", 8787))
    mode: Mode = field(default_factory=lambda: _env_str("LOCAL_PROXY_MODE", "mode", "exploit").strip().lower())  # type: ignore[assignment]
    upstream_base_url: str = field(default_factory=lambda: _env_str("PACKY_BASE_URL", "upstream.base_url", "https://packyapi.com").rstrip("/"))
    upstream_api_key_env: str = field(default_factory=lambda: _env_str("PACKY_API_KEY_ENV", "upstream.api_key_env", "PACKY_API_KEY"))
    upstream_api_key_file_value: str = field(default_factory=lambda: _env_str("", "upstream.api_key", ""))
    default_model: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_MODEL", "model.default", "gpt-5.5"))
    model_aliases: Dict[str, str] = field(default_factory=lambda: _cfg_dict("model.aliases", {"codex-gpt55": "gpt-5.5"}))
    function_name: str = field(default_factory=lambda: _env_str("BILL015_FUNCTION_NAME", "bill015.function_name", "emit_value"))
    answer_field: str = field(default_factory=lambda: _env_str("BILL015_ANSWER_FIELD", "bill015.answer_field", "answer"))
    max_output_tokens: int = field(default_factory=lambda: _env_int("BILL015_MAX_OUTPUT_TOKENS", "bill015.max_output_tokens", 2048))
    reasoning_effort: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_REASONING_EFFORT", "reasoning.effort", "high"))
    reasoning_summary: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_REASONING_SUMMARY", "reasoning.summary", "auto"))
    max_answer_chars: int = field(default_factory=lambda: _env_int("BILL015_MAX_ANSWER_CHARS", "bill015.max_answer_chars", 16384))
    max_request_bytes: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_MAX_REQUEST_BYTES", "limits.max_request_bytes", 1048576))
    max_concurrency: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_MAX_CONCURRENCY", "limits.max_concurrency", 2))
    upstream_timeout_seconds: float = field(default_factory=lambda: _env_float("PACKY_TIMEOUT_SECONDS", "upstream.timeout_seconds", 60.0))
    args_done_timeout_ms: int = field(default_factory=lambda: _env_int("BILL015_ARGS_DONE_TIMEOUT_MS", "limits.args_done_timeout_ms", 45000))
    upstream_idle_timeout_ms: int = field(default_factory=lambda: _env_int("BILL015_UPSTREAM_IDLE_TIMEOUT_MS", "limits.upstream_idle_timeout_ms", 15000))
    evidence_dir: Path = field(default_factory=lambda: Path(_env_str("LOCAL_PROXY_LOG_DIR", "logging.dir", str(PROJECT_ROOT / "proxy_evidence"))))
    store_prompts: bool = field(default_factory=lambda: _env_bool("LOCAL_PROXY_STORE_PROMPTS", "logging.store_prompts", False))
    store_answers: bool = field(default_factory=lambda: _env_bool("LOCAL_PROXY_STORE_ANSWERS", "logging.store_answers", True))
    rotate_mb: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_LOG_ROTATE_MB", "logging.rotate_mb", 10))
    admin_token: str = field(default_factory=lambda: _env_str("LOCAL_PROXY_ADMIN_TOKEN", "admin.token", ""))
    circuit_failures: int = field(default_factory=lambda: _env_int("LOCAL_PROXY_CIRCUIT_FAILURES", "limits.circuit_failures", 5))
    packy_cookie: str = field(default_factory=lambda: _env_str("PACKY_COOKIE", "upstream.cookie", ""))
    packy_user_id: str = field(default_factory=lambda: _env_str("PACKY_USER_ID", "upstream.user_id", "192833"))

    @property
    def upstream_api_key(self) -> str:
        # Local config is the normal path now; env remains an optional override for advanced/test use.
        return os.getenv(self.upstream_api_key_env, "") or self.upstream_api_key_file_value

    @property
    def upstream_configured(self) -> bool:
        return bool(self.upstream_api_key)

    def map_model(self, model: str | None) -> str:
        # Codex Desktop may send stale/internal aliases (observed: gpt-5.1)
        # even when the UI displays 5.5. This proxy intentionally targets
        # the configured upstream model, so every unknown client model is
        # normalized to default_model.
        if not model:
            return self.default_model
        return self.model_aliases.get(model, self.default_model)

    def public_model_ids(self) -> list[str]:
        ids = [self.default_model]
        ids.extend(self.model_aliases.keys())
        return list(dict.fromkeys(ids))

    def validate_mode(self) -> None:
        if self.mode not in {"exploit", "verify", "normal", "dry-run"}:
            self.mode = "exploit"


settings = Settings()
settings.validate_mode()
