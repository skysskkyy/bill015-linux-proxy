from __future__ import annotations

import base64
import math
from typing import Any

IMAGE_PART_TYPES = {"input_image", "image_url", "output_image", "computer_screenshot"}


def is_image_part(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    typ = str(obj.get("type") or "")
    return typ in IMAGE_PART_TYPES or "image_url" in obj or bool(obj.get("image"))


def _url_and_detail(obj: dict[str, Any]) -> tuple[str, str]:
    detail = str(obj.get("detail") or "high").strip().lower() or "high"
    raw = obj.get("image_url") if obj.get("image_url") is not None else obj.get("image")
    if isinstance(raw, dict):
        detail = str(raw.get("detail") or detail).strip().lower() or "high"
        raw = raw.get("url") or raw.get("image_url") or raw.get("image") or ""
    return str(raw or ""), detail


def _b64_bytes(url: str) -> bytes | None:
    marker = ";base64,"
    if not url.startswith("data:") or marker not in url:
        return None
    blob = url.split(marker, 1)[1].strip()
    try:
        return base64.b64decode(blob, validate=False)
    except Exception:
        return None


def image_size_from_bytes(data: bytes) -> tuple[int, int] | None:
    if len(data) < 10:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return (width, height) if width and height else None
    if data[:6] in {b"GIF87a", b"GIF89a"} and len(data) >= 10:
        width = int.from_bytes(data[6:8], "little")
        height = int.from_bytes(data[8:10], "little")
        return (width, height) if width and height else None
    if data[:2] == b"\xff\xd8":
        return _jpeg_size(data)
    if data[:4] == b"RIFF" and len(data) >= 30 and data[8:12] == b"WEBP":
        return _webp_size(data)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    index = 2
    length = len(data)
    while index + 8 < length:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return (width, height) if width and height else None
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if index + 4 > length:
            break
        segment = int.from_bytes(data[index + 2 : index + 4], "big")
        if segment < 2:
            break
        index += 2 + segment
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    kind = data[12:16]
    if kind == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return (width, height) if width and height else None
    if kind == b"VP8 " and len(data) >= 30:
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return (width, height) if width and height else None
    if kind == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return (width, height) if width and height else None
    return None


def vision_tokens_for_size(width: int, height: int, detail: str = "high") -> int:
    if width <= 0 or height <= 0:
        return 85
    if detail == "low":
        return 85
    longest = max(width, height)
    if longest > 2048:
        scale = 2048 / longest
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
    shortest = min(width, height)
    if shortest > 768:
        scale = 768 / shortest
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
    tiles = math.ceil(width / 512) * math.ceil(height / 512)
    return 85 + 170 * max(1, tiles)


def _tokens_from_nbytes(nbytes: int, detail: str) -> int:
    if detail == "low":
        return 85
    if nbytes < 40_000:
        return 300
    if nbytes < 150_000:
        return 765
    if nbytes < 400_000:
        return 1105
    return min(2805, 1500 + nbytes // 4000)


def estimate_image_part(obj: dict[str, Any]) -> int:
    url, detail = _url_and_detail(obj)
    raw = _b64_bytes(url)
    if raw:
        size = image_size_from_bytes(raw)
        if size:
            return vision_tokens_for_size(size[0], size[1], detail)
        return _tokens_from_nbytes(len(raw), detail)
    return 85 if detail == "low" else 765
