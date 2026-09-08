from __future__ import annotations

from app.upstream.errors import CYBER_POLICY_ERROR_MESSAGE, is_cyber_policy_rotation_error, key_rotation_error_reason
from app.upstream.keys import ApiKeyPool


def test_cyber_policy_matcher_requires_logged_fields():
    ok = {"error": {"code": "cyber_policy", "message": CYBER_POLICY_ERROR_MESSAGE}}
    assert is_cyber_policy_rotation_error(ok)
    assert key_rotation_error_reason(ok) == "cyber_policy"
    nope = {"error": {"code": "cyber_policy", "message": "please rephrase"}}
    assert not is_cyber_policy_rotation_error(nope)
    assert key_rotation_error_reason(nope) is None


def test_key_pool_rotates_and_cools_failed_key():
    pool = ApiKeyPool(["aaa", "bbb", "ccc"])
    first = pool.current()
    assert first is not None and first.key == "aaa"
    nxt = pool.rotate_after_failure("aaa", {"aaa"}, cooldown_seconds=600)
    assert nxt is not None and nxt.key == "bbb"
    again = pool.current()
    assert again is not None and again.key == "bbb"
