"""Fixtures for testing the running lab stack (not mocks: these hit the real services)."""
from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest
from dotenv import dotenv_values

from settings import GATEWAY, LLAMA, MODEL, ROOT


class Secret(str):
    """A str that hides its value in repr().

    pytest prints fixture values in failure tracebacks using repr(); this keeps
    keys out of terminal output, logs and pasted error reports. str(), f-strings
    and HTTP headers still get the real value.
    """
    def __repr__(self) -> str:
        return f"<redacted ...{self[-4:]}>"


def _secret(path: Path, name: str) -> Secret:
    value = dotenv_values(path).get(name)
    if not value:
        pytest.exit(f"{name} missing from {path}", returncode=2)
    return Secret(value)


@pytest.fixture(scope="session")
def master_key() -> Secret:
    return _secret(ROOT / ".env", "LITELLM_MASTER_KEY")


@pytest.fixture(scope="session")
def llama_key() -> Secret:
    return _secret(ROOT / ".env", "LLAMACPP_API_KEY")


@pytest.fixture(scope="session")
def client_key() -> Secret:
    return _secret(ROOT / "client" / ".env", "LITELLM_API_KEY")


@pytest.fixture(scope="session")
def http():
    with httpx.Client(timeout=120) as client:
        yield client


@pytest.fixture(scope="session", autouse=True)
def stack_is_up(http):
    """Stop the whole run with one clear message instead of 20 confusing failures."""
    for name, url in [("LiteLLM", f"{GATEWAY}/health/liveliness"),
                      ("llama-server", f"{LLAMA}/health")]:
        try:
            http.get(url, timeout=5).raise_for_status()
        except httpx.HTTPError as e:
            pytest.exit(f"{name} not reachable at {url} ({type(e).__name__}) - start the stack first",
                        returncode=2)


@pytest.fixture(scope="session")
def chat(http):
    """Send one small chat request with the given key; returns the raw httpx response."""
    def _chat(key: str, content: str = "Reply with just: OK /no_think", max_tokens: int = 8):
        return http.post(
            f"{GATEWAY}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": MODEL,
                  "messages": [{"role": "user", "content": content}],
                  "max_tokens": max_tokens},
        )
    return _chat


@pytest.fixture
def temp_key(http, master_key):
    """Factory for throwaway virtual keys with custom limits, deleted after the test.

    Tests never spend the real client key's budget, and stay repeatable.
    """
    admin = {"Authorization": f"Bearer {master_key}"}
    created: list[str] = []

    def _make(**limits) -> str:
        body = {"key_alias": f"pytest-{uuid.uuid4().hex[:8]}", "models": [MODEL], **limits}
        r = http.post(f"{GATEWAY}/key/generate", headers=admin, json=body)
        r.raise_for_status()
        created.append(Secret(r.json()["key"]))
        return created[-1]

    yield _make
    if created:
        http.post(f"{GATEWAY}/key/delete", headers=admin, json={"keys": created})
