"""Budget and rate-limit enforcement, each on its own throwaway key."""
import threading
import time

import pytest

from settings import GATEWAY, MODEL, REASONING_MODEL


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


def test_key_cannot_use_an_alias_it_was_not_granted(chat, temp_key):
    key = temp_key(models=[MODEL])
    assert chat(key).status_code == 200
    r = chat(key, model=REASONING_MODEL)
    assert r.status_code in (401, 403), r.text


def test_simultaneous_requests_cannot_jointly_overshoot_the_budget(chat, temp_key, http, master_key):
    """Four requests at once on a key with half a request of budget left: at most one gets in.

    Observed on LiteLLM 1.109.1 (bench/budget_race.py): 1 of 2 and 1 of 4 admitted, so the
    overshoot stays under one request even under concurrency. Guards against an upgrade
    that goes back to check-then-charge without accounting for requests in flight.
    """
    prompt, max_tokens = "Count from 1 to 500, separated by spaces. Output only the numbers.", 64
    admin = {"Authorization": f"Bearer {master_key}"}

    probe = chat(temp_key(max_budget=1.0), prompt, max_tokens=max_tokens)
    assert probe.status_code == 200
    cost = float(probe.headers["x-litellm-response-cost"])

    key = temp_key(max_budget=1.5 * cost)
    assert chat(key, prompt, max_tokens=max_tokens).status_code == 200
    for _ in range(30):                          # spend is recorded asynchronously
        spend = http.get(f"{GATEWAY}/key/info", headers=admin, params={"key": key}).json()["info"]["spend"]
        if spend >= cost * 0.999:
            break
        time.sleep(1)
    else:
        pytest.fail("spend of the first request was never recorded")

    barrier = threading.Barrier(4)
    responses = [None] * 4

    def worker(i: int) -> None:
        barrier.wait()
        responses[i] = chat(key, prompt, max_tokens=max_tokens)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    codes = [r.status_code for r in responses]
    assert codes.count(200) <= 1, f"budget overshot by more than one request: {codes}"
    for r in responses:
        if r.status_code != 200:
            assert r.status_code in (400, 422), r.text
            assert "budget" in r.json()["error"]["message"].lower()
