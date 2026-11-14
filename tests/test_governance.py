"""Budget and rate-limit enforcement, each on its own throwaway key."""
import time

import pytest


def test_budget_blocks_after_exhaustion(chat, temp_key):
    key = temp_key(max_budget=0.00001)          # less than one request costs
    first = chat(key)
    assert first.status_code == 200, "a fresh key starts under budget"

    # Spend is recorded asynchronously, so allow a few attempts
    for _ in range(5):
        time.sleep(2)
        r = chat(key)
        if r.status_code != 200:
            break
    else:
        pytest.fail("budget was never enforced")

    assert r.status_code in (400, 422)          # 422 on LiteLLM 1.109.1
    message = r.json()["error"]["message"].lower()
    assert "budget" in message, "client/chat.py detects budget errors by this word"


def test_rpm_limit_returns_429_on_the_extra_request(chat, temp_key):
    key = temp_key(rpm_limit=2)
    codes = [chat(key).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
