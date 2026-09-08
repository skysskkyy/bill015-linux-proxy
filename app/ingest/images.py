from __future__ import annotations

from typing import Any

from ..config import Settings, settings

VISION_NOTICE = (
    "[LOCAL VISION NOTICE] This proxy did not inspect image bytes. "
    "Do not claim to have seen screenshots or photos. Ask for a text description or OCR if needed."
)


def _walk_replace(value: Any, *, replacements: list[int], cfg: Settings) -> Any:
    if isinstance(value, list):
        return [_walk_replace(item, replacements=replacements, cfg=cfg) for item in value]
    if not isinstance(value, dict):
        return value
    typ = str(value.get("type") or "")
    if typ in {"input_image", "image_url", "output_image"} or "image_url" in value or value.get("image"):
        replacements[0] += 1
        if cfg.multimodal_strategy == "native_passthrough":
            return value
        return {"type": "input_text", "text": VISION_NOTICE}
    return {key: _walk_replace(val, replacements=replacements, cfg=cfg) for key, val in value.items()}


def sanitize_images(body: dict[str, Any], cfg: Settings = settings) -> tuple[dict[str, Any], int]:
    if cfg.multimodal_strategy == "native_passthrough":
        return body, 0
    replacements = [0]
    cleaned = _walk_replace(body, replacements=replacements, cfg=cfg)
    count = replacements[0]
    if count and cfg.multimodal_strategy == "reject":
        raise ValueError("image inputs are not supported by this local proxy")
    return cleaned, count
