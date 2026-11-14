"""End-to-end tests of client/chat.py exit codes, run as a real subprocess.

Each behaviour is checked in both output modes (streaming and --no-stream),
because they use different code paths in the client.
"""
import os
import subprocess
import sys
import time

import pytest

from settings import ROOT

CLIENT = ROOT / "client" / "chat.py"
MODES = [pytest.param([], id="stream"), pytest.param(["--no-stream"], id="blocking")]


def run_client(*args, **env_overrides):
    # UTF-8 both ways: Windows pipes default to cp1252 and choke on characters like emoji
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **env_overrides}
    return subprocess.run([sys.executable, str(CLIENT), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          env=env, timeout=180)


@pytest.mark.parametrize("mode", MODES)
def test_success_exits_0(mode):
    p = run_client("Reply with just: OK", *mode)
    assert p.returncode == 0, p.stderr
    assert "OK" in p.stdout


@pytest.mark.parametrize("mode", MODES)
def test_truncated_answer_exits_6(mode):
    p = run_client("Count from 1 to 50, separated by commas.", "--max-tokens", "5", *mode)
    assert p.returncode == 6, p.stderr
    assert "WARNING" in p.stderr


@pytest.mark.parametrize("mode", MODES)
def test_invalid_key_exits_4(mode):
    # An env var takes precedence over client/.env (load_dotenv does not override)
    p = run_client("hi", *mode, LITELLM_API_KEY="sk-not-a-real-key")
    assert p.returncode == 4, p.stderr


@pytest.mark.parametrize("mode", MODES)
def test_exhausted_budget_exits_3(mode, chat, temp_key):
    key = temp_key(max_budget=0.00001)
    chat(key)                                   # spend it
    time.sleep(3)                               # let spend be recorded
    p = run_client("hi", *mode, LITELLM_API_KEY=key)
    assert p.returncode == 3, p.stderr


@pytest.mark.slow
def test_unreachable_gateway_exits_5():
    p = run_client("hi", LITELLM_BASE_URL="http://127.0.0.1:4999/v1")
    assert p.returncode == 5, p.stderr
