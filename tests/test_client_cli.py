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
    # The client must produce UTF-8 on pipes by itself, so no PYTHONIOENCODING here
    env = {**os.environ, **env_overrides}
    return subprocess.run([sys.executable, str(CLIENT), *args],
                          capture_output=True, text=True,
                          # errors="replace": keep diagnostics readable even if the
                          # child emits something that is not valid UTF-8
                          encoding="utf-8", errors="replace",
                          env=env, timeout=180)


def test_reads_key_from_client_env_file():
    # The only CLI test without a key override: proves client/.env is used (1 request)
    p = run_client("Reply with just: OK")
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("mode", MODES)
def test_success_exits_0(mode, suite_key):
    p = run_client("Reply with just: OK", *mode, LITELLM_API_KEY=suite_key)
    assert p.returncode == 0, p.stderr
    assert "OK" in p.stdout


@pytest.mark.parametrize("mode", MODES)
def test_truncated_answer_exits_6(mode, suite_key):
    p = run_client("Count from 1 to 50, separated by commas.", "--max-tokens", "5", *mode,
                   LITELLM_API_KEY=suite_key)
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


def _fits_cp1252(text: str) -> bool:
    try:
        text.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


@pytest.mark.parametrize("mode", MODES)
def test_non_cp1252_output_survives_a_pipe(mode, suite_key):
    # PYTHONIOENCODING=cp1252 simulates a default Windows machine, where Python
    # encodes piped output with the ANSI code page
    p = run_client("Reply with only the check mark emoji and nothing else.", *mode,
                   PYTHONIOENCODING="cp1252", LITELLM_API_KEY=suite_key)
    assert p.returncode == 0, p.stderr
    assert not _fits_cp1252(p.stdout), \
        f"inconclusive: the model replied {p.stdout!r}, which cp1252 can encode anyway"
