from __future__ import annotations

from app.context.image_tokens import estimate_image_part, vision_tokens_for_size
from app.context.pack import pack_turn_items
from app.context.tokens import item_tokens
from app.ingest import normalize_responses_request

PNG_1x1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_vision_tokens_1x1_is_one_tile():
    assert vision_tokens_for_size(1, 1, "high") == 255
    assert vision_tokens_for_size(1, 1, "low") == 85
    assert vision_tokens_for_size(1024, 1024, "high") == 765


def test_data_url_png_uses_header_size_not_base64_length():
    url = f"data:image/png;base64,{PNG_1x1}"
    tokens = estimate_image_part({"type": "input_image", "image_url": url, "detail": "high"})
    assert tokens == 255
    padded = {"type": "input_image", "image_url": f"data:image/png;base64,{'A' * 80_000}"}
    padded_tokens = item_tokens(padded)
    assert padded_tokens < 2000
    assert padded_tokens >= 85


def test_passthrough_keeps_input_image_in_packed_input():
    url = f"data:image/png;base64,{PNG_1x1}"
    body = {
        "model": "gpt-6-astra",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is in this image"},
                    {"type": "input_image", "image_url": url, "detail": "high"},
                ],
            }
        ],
    }
    turn = normalize_responses_request(body)
    packed = pack_turn_items(turn)
    blob = str(packed.items)
    assert "input_image" in blob
    assert PNG_1x1 in blob
    assert "LOCAL VISION NOTICE" not in blob
    assert packed.tokens >= 255
    assert packed.tokens < 2000
