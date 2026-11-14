"""Health, authentication on both layers, and cost accounting."""
import pytest

from settings import (GATEWAY, INPUT_COST_PER_TOKEN, LLAMA, MODEL,
                      OUTPUT_COST_PER_TOKEN)


def test_readiness_reports_db_connected(http):
    body = http.get(f"{GATEWAY}/health/readiness").json()
    assert body["status"] == "healthy"
    assert body["db"] == "connected"


@pytest.mark.parametrize("auth", [None, "Bearer sk-not-a-real-key"], ids=["missing", "invalid"])
def test_gateway_rejects_missing_or_invalid_key(http, auth):
    headers = {"Authorization": auth} if auth else {}
    r = http.post(f"{GATEWAY}/v1/chat/completions", headers=headers,
                  json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401


def test_llama_server_rejects_requests_without_key(http):
    # If this fails, llama-server was started without scripts/start-llama.ps1
    assert http.get(f"{LLAMA}/v1/models").status_code == 401


def test_llama_server_accepts_the_key_from_env(http, llama_key):
    r = http.get(f"{LLAMA}/v1/models", headers={"Authorization": f"Bearer {llama_key}"})
    assert r.status_code == 200
    assert MODEL in [m["id"] for m in r.json()["data"]]


def test_client_key_can_chat(chat, client_key):
    r = chat(client_key)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"].strip()
    assert body["usage"]["completion_tokens"] > 0


def test_cost_header_matches_notional_pricing(chat, client_key):
    r = chat(client_key)
    assert r.status_code == 200, r.text
    u = r.json()["usage"]
    expected = (u["prompt_tokens"] * INPUT_COST_PER_TOKEN
                + u["completion_tokens"] * OUTPUT_COST_PER_TOKEN)
    assert float(r.headers["x-litellm-response-cost"]) == pytest.approx(expected, rel=1e-6)
