"""Minimal LiteLLM client: one prompt in, one answer out.

Talks to the LiteLLM proxy through the OpenAI-compatible API. It only knows the
gateway's public aliases (local-fast, local-reasoning); which model sits behind
them, and how thinking is switched on or off, is configured in LiteLLM.
Reads LITELLM_API_KEY from client/.env.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

# 127.0.0.1, not "localhost": Docker publishes the port on IPv4 only, and on
# Windows "localhost" tries IPv6 (::1) first, costing ~2s per connection.
BASE_URL = os.getenv("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1")
ENV_FILE = Path(__file__).resolve().parent / ".env"

# Distinct exit codes so scripts and tests can tell failures apart
EXIT_OK = 0
EXIT_OTHER = 1
EXIT_RATE_LIMIT = 2
EXIT_BUDGET = 3
EXIT_AUTH = 4
EXIT_UNREACHABLE = 5
EXIT_TRUNCATED = 6

FAST_MODEL = "local-fast"
REASONING_MODEL = "local-reasoning"

DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_TOKENS_REASONING = 2048  # reasoning blocks are long


def make_client() -> OpenAI:
    load_dotenv(ENV_FILE)
    key = os.getenv("LITELLM_API_KEY")
    if not key:
        sys.exit(f"LITELLM_API_KEY not found (expected in {ENV_FILE})")
    # The SDK itself retries 408/409/429/5xx with exponential backoff
    # (honouring Retry-After). It never retries other 4xx, e.g. budget errors.
    return OpenAI(base_url=BASE_URL, api_key=key, max_retries=3, timeout=120)


def is_budget_error(err: openai.APIStatusError) -> bool:
    # LiteLLM signals an exhausted budget with a 4xx whose message mentions it
    return "budget" in str(err).lower()


def main() -> int:
    p = argparse.ArgumentParser(description="Send one prompt through LiteLLM.")
    p.add_argument("prompt")
    p.add_argument("--model", default=None,
                   help=f"gateway alias (default {FAST_MODEL})")
    p.add_argument("--reasoning", action="store_true",
                   help=f"shortcut for --model {REASONING_MODEL} (slow, costs more tokens)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help=f"default {DEFAULT_MAX_TOKENS}, or {DEFAULT_MAX_TOKENS_REASONING} with --reasoning")
    p.add_argument("--show-thinking", action="store_true",
                   help="also print the reasoning block")
    args = p.parse_args()

    if args.reasoning and args.model:
        p.error("use either --reasoning or --model, not both")
    model = REASONING_MODEL if args.reasoning else (args.model or FAST_MODEL)
    max_tokens = args.max_tokens or (DEFAULT_MAX_TOKENS_REASONING if args.reasoning else DEFAULT_MAX_TOKENS)
    client = make_client()

    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=max_tokens,
        )
    except openai.RateLimitError:  # 429 - must come before APIStatusError
        print("Rate limited (429) even after retries - slow down or raise rpm_limit.",
              file=sys.stderr)
        return EXIT_RATE_LIMIT
    except openai.APIStatusError as e:  # any other HTTP error status
        if is_budget_error(e):
            print(f"Budget exhausted (HTTP {e.status_code}) - retrying will not help.",
                  file=sys.stderr)
            return EXIT_BUDGET
        if e.status_code == 401:
            print("Key rejected (401) - check LITELLM_API_KEY.", file=sys.stderr)
            return EXIT_AUTH
        print(f"API error HTTP {e.status_code}: {e.message}", file=sys.stderr)
        return EXIT_OTHER
    except openai.APIConnectionError:  # no HTTP response at all
        print(f"Cannot reach LiteLLM at {BASE_URL} - is the container running?",
              file=sys.stderr)
        return EXIT_UNREACHABLE

    resp = raw.parse()
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)  # non-standard field, passed through by LiteLLM

    if args.show_thinking and reasoning:
        print("--- thinking ---")
        print(reasoning.strip())
        print("--- answer ---")
    print((msg.content or "").strip())

    u = resp.usage
    finish = resp.choices[0].finish_reason
    cost = raw.headers.get("x-litellm-response-cost", "n/a")
    print(
        f"\n[{resp.model} | tokens in/out {u.prompt_tokens}/{u.completion_tokens} | "
        f"notional cost ${cost} | finish={finish}]",
        file=sys.stderr,
    )

    # A cut-off answer is not a success - the tokens were still paid for
    if finish == "length":
        hint = " (the reasoning used them all - no answer produced)" if not msg.content else ""
        print(f"WARNING: hit max_tokens={max_tokens}{hint}. Retry with a higher --max-tokens.",
              file=sys.stderr)
        return EXIT_TRUNCATED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
