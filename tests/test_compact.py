from __future__ import annotations

from app.context.compact import merge_compact_output, parse_compact_output


def test_parse_compact_output_reads_output_array():
    items = parse_compact_output(
        {
            "object": "response.compaction",
            "output": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "compaction", "id": "cmp_1", "encrypted_content": "blob"},
            ],
        }
    )
    assert items[1]["type"] == "compaction"
    assert items[1]["encrypted_content"] == "blob"


def test_merge_compact_output_appends_latest_user():
    source = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "latest task"}]},
    ]
    compacted = [{"type": "compaction", "id": "cmp_1", "encrypted_content": "blob"}]
    merged = merge_compact_output(source, compacted)
    assert merged[0]["type"] == "compaction"
    assert any("latest task" in str(item) for item in merged)
