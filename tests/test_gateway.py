"""Health, authentication on both layers, cost accounting, aliases and streaming."""
import json

import pytest

from settings import (BACKEND_MODEL, GATEWAY, INPUT_COST_PER_TOKEN, LLAMA, MODEL,
                      OUTPUT_COST_PER_TOKEN, REASONING_MODEL)


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
    assert BACKEND_MODEL in [m["id"] for m in r.json()["data"]]


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


def test_fast_alias_answers_without_reasoning(chat, client_key):
    r = chat(client_key, content="Is 391 a prime number? Answer in one sentence.", max_tokens=100)
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    assert not msg.get("reasoning_content"), "thinking should be off for " + MODEL
    assert msg["content"].strip()


def test_reasoning_alias_produces_reasoning(chat, client_key):
    # 30 tokens is enough to see reasoning start; the answer itself is not needed
    r = chat(client_key, content="Is 391 a prime number?", max_tokens=30, model=REASONING_MODEL)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"].get("reasoning_content"), \
        "thinking should be on for " + REASONING_MODEL


def test_streaming_sends_chunks_then_usage_then_done(http, client_key):
    """Checks the SSE contract the client relies on, at the HTTP level."""
    pieces, usage, done = [], None, False
    body = {"model": MODEL, "max_tokens": 30, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Count from 1 to 10."}]}
    with http.stream("POST", f"{GATEWAY}/v1/chat/completions",
                     headers={"Authorization": f"Bearer {client_key}"}, json=body) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue                         # blank separators between events
            data = line[len("data: "):]
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                text = (choice.get("delta") or {}).get("content")
                if text:
                    pieces.append(text)

    assert len(pieces) >= 2, "the answer should arrive in several pieces"
    assert usage and usage["completion_tokens"] > 0, "include_usage should add a usage chunk"
    assert done, "the stream should end with [DONE]"
