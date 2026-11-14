"""Minimal LiteLLM client: one prompt in, one answer out.

Talks to the LiteLLM proxy through the OpenAI-compatible API. It only knows the
gateway's public aliases (local-fast, local-reasoning); which model sits behind
them, and how thinking is switched on or off, is configured in LiteLLM.
Reads LITELLM_API_KEY from client/.env.

Streams tokens as they arrive by default; --no-stream waits for the full answer.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
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
EXIT_INTERRUPTED = 130  # conventional code for Ctrl+C

FAST_MODEL = "local-fast"
REASONING_MODEL = "local-reasoning"

DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_TOKENS_REASONING = 2048  # reasoning blocks are long


@dataclass
class Result:
    """What both modes report back, so the summary and exit code are computed once."""
    model: str | None = None
    content: str = ""
    finish: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: str = "n/a"
    started: float = 0.0
    first_token: float | None = None  # time of first streamed token (streaming only)
    ended: float = 0.0


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


def run_blocking(client: OpenAI, req: dict, show_thinking: bool) -> Result:
    res = Result(started=time.perf_counter())
    raw = client.chat.completions.with_raw_response.create(**req)
    resp = raw.parse()
    res.ended = time.perf_counter()

    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)  # non-standard, passed through by LiteLLM
    if show_thinking and reasoning:
        print("--- thinking ---")
        print(reasoning.strip())
        print("--- answer ---")
    res.content = (msg.content or "").strip()
    print(res.content)

    res.model = resp.model
    res.finish = resp.choices[0].finish_reason
    res.prompt_tokens = resp.usage.prompt_tokens
    res.completion_tokens = resp.usage.completion_tokens
    res.cost = raw.headers.get("x-litellm-response-cost", "n/a")
    return res


def run_streaming(client: OpenAI, req: dict, show_thinking: bool) -> Result:
    res = Result(started=time.perf_counter())
    progress = sys.stderr.isatty()  # live "thinking..." counter only on a real console
    reasoning_chunks = 0
    printed_answer = False

    # include_usage: the server sends one extra final chunk with token counts
    stream = client.chat.completions.create(**req, stream=True,
                                            stream_options={"include_usage": True})
    with stream:  # closes the HTTP connection even on Ctrl+C or errors
        for chunk in stream:
            res.model = chunk.model or res.model
            if chunk.usage:  # the final usage-only chunk has no choices
                res.prompt_tokens = chunk.usage.prompt_tokens
                res.completion_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            res.finish = choice.finish_reason or res.finish
            delta = choice.delta

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                res.first_token = res.first_token or time.perf_counter()
                reasoning_chunks += 1
                if show_thinking:
                    if reasoning_chunks == 1:
                        print("--- thinking ---")
                    print(reasoning, end="", flush=True)
                elif progress:
                    print(f"\r(thinking... {reasoning_chunks} chunks)", end="",
                          file=sys.stderr, flush=True)

            if delta.content:
                res.first_token = res.first_token or time.perf_counter()
                if not printed_answer:
                    if show_thinking and reasoning_chunks:
                        print("\n--- answer ---")
                    elif progress and reasoning_chunks:
                        print("\r" + " " * 40 + "\r", end="", file=sys.stderr, flush=True)
                    printed_answer = True
                res.content += delta.content
                print(delta.content, end="", flush=True)

    res.ended = time.perf_counter()
    print()
    # Response headers are sent before generation starts, so the gateway cannot
    # report this request's cost in them; it is still recorded in LiteLLM's logs.
    res.cost = "n/a (streamed; see gateway logs)"
    return res


def summarise(res: Result) -> None:
    total = res.ended - res.started
    parts = [res.model or "?"]
    if res.prompt_tokens is not None:
        parts.append(f"tokens in/out {res.prompt_tokens}/{res.completion_tokens}")
    if res.first_token is not None:
        ttft = res.first_token - res.started
        gen = res.ended - res.first_token
        rate = f" | {res.completion_tokens / gen:.1f} tok/s" if res.completion_tokens and gen > 0 else ""
        parts.append(f"first token {ttft:.2f}s{rate}")
    parts.append(f"total {total:.2f}s")
    parts.append(f"cost ${res.cost}" if not res.cost.startswith("n/a") else f"cost {res.cost}")
    parts.append(f"finish={res.finish}")
    print("\n[" + " | ".join(parts) + "]", file=sys.stderr)


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
    p.add_argument("--no-stream", action="store_true",
                   help="wait for the complete answer instead of streaming it")
    args = p.parse_args()

    if args.reasoning and args.model:
        p.error("use either --reasoning or --model, not both")
    model = REASONING_MODEL if args.reasoning else (args.model or FAST_MODEL)
    max_tokens = args.max_tokens or (DEFAULT_MAX_TOKENS_REASONING if args.reasoning else DEFAULT_MAX_TOKENS)
    req = {"model": model,
           "messages": [{"role": "user", "content": args.prompt}],
           "max_tokens": max_tokens}

    client = make_client()
    run = run_blocking if args.no_stream else run_streaming

    try:
        res = run(client, req, args.show_thinking)
    # HTTP errors arrive before any token is streamed, so both modes share this
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
    except openai.APIConnectionError:  # no HTTP response at all, or connection lost
        print(f"\nCannot reach LiteLLM at {BASE_URL} (or the connection dropped).",
              file=sys.stderr)
        return EXIT_UNREACHABLE
    except openai.APIError as e:  # e.g. an error event in the middle of a stream
        print(f"\nStream failed: {e}", file=sys.stderr)
        return EXIT_OTHER
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED

    summarise(res)

    # A cut-off answer is not a success - the tokens were still paid for
    if res.finish == "length":
        hint = " (the reasoning used them all - no answer produced)" if not res.content else ""
        print(f"WARNING: hit max_tokens={max_tokens}{hint}. Retry with a higher --max-tokens.",
              file=sys.stderr)
        return EXIT_TRUNCATED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
