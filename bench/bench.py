"""Benchmark the gateway aliases: time to first token, generation speed, total time.

Every option (A, B, C, baseline) is measured with the same prompts and settings,
so the numbers are comparable. Results are printed and appended to
bench/results.jsonl with a label.

Usage (from the project root, .venv active):
  python bench\\bench.py --label baseline                    # warm, grouped by alias
  python bench\\bench.py --label B --order interleaved       # alternate aliases (forces swaps in B)
  python bench\\bench.py --label A --concurrent              # both aliases at the same time
  python bench\\bench.py --label B1 --concurrent --reverse   # same, but local-reasoning starts first
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import dotenv_values
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = "http://127.0.0.1:4000"
RESULTS = Path(__file__).resolve().parent / "results.jsonl"

# Fixed workloads: never change these between options, or the numbers stop being comparable
WORKLOADS = {
    "local-fast": {"prompt": "Explain what an API gateway does, in three sentences.",
                   "max_tokens": 200},
    "local-reasoning": {"prompt": "Is 391 a prime number?",
                        "max_tokens": 512},
}


def create_bench_key() -> tuple[str, callable]:
    """A throwaway key with generous limits, so benchmarks never touch real keys."""
    master = dotenv_values(ROOT / ".env")["LITELLM_MASTER_KEY"]
    admin = {"Authorization": f"Bearer {master}"}
    r = httpx.post(f"{GATEWAY}/key/generate", headers=admin, timeout=30,
                   json={"key_alias": f"bench-{uuid.uuid4().hex[:8]}",
                         "models": list(WORKLOADS), "rpm_limit": 120, "max_budget": 0.20})
    r.raise_for_status()
    key = r.json()["key"]

    def delete() -> None:
        httpx.post(f"{GATEWAY}/key/delete", headers=admin, json={"keys": [key]}, timeout=30)

    return key, delete


def one_request(client: OpenAI, alias: str) -> dict:
    w = WORKLOADS[alias]
    t0 = time.perf_counter()
    first = usage = finish = None
    stream = client.chat.completions.create(
        model=alias, max_tokens=w["max_tokens"], stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": w["prompt"]}])
    with stream:
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # First token = first answer OR reasoning text, whichever comes first
            if first is None and (delta.content or getattr(delta, "reasoning_content", None)):
                first = time.perf_counter()
            finish = chunk.choices[0].finish_reason or finish
    t1 = time.perf_counter()

    tokens = usage.completion_tokens if usage else None
    gen_s = (t1 - first) if first else None
    return {"alias": alias,
            "ttft_s": round(first - t0, 3) if first else None,
            "total_s": round(t1 - t0, 3),
            "completion_tokens": tokens,
            "gen_tok_s": round(tokens / gen_s, 2) if tokens and gen_s else None,
            "finish": finish}


def run_concurrent(client: OpenAI, aliases: list[str]) -> list[dict]:
    """Start one request per alias at the same moment and wait for all."""
    results: list[dict] = []
    lock = threading.Lock()

    def worker(alias: str) -> None:
        r = one_request(client, alias)
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(a,)) for a in aliases]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def summarise(rows: list[dict]) -> None:
    print(f"\n{'alias':<17}{'n':>3}{'ttft med':>10}{'ttft max':>10}{'tok/s med':>11}{'total med':>11}")
    for alias in WORKLOADS:
        rs = [r for r in rows if r["alias"] == alias]
        if not rs:
            continue
        med = lambda k: statistics.median(r[k] for r in rs if r[k] is not None)
        print(f"{alias:<17}{len(rs):>3}{med('ttft_s'):>9.2f}s"
              f"{max(r['ttft_s'] for r in rs if r['ttft_s'] is not None):>9.2f}s"
              f"{med('gen_tok_s'):>11.1f}{med('total_s'):>10.2f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="which option/setup this run measures")
    p.add_argument("--runs", type=int, default=3, help="measured requests per alias")
    p.add_argument("--order", choices=["grouped", "interleaved"], default="grouped",
                   help="grouped: fast x N then reasoning x N, after a warm-up; "
                        "interleaved: alternate aliases, no warm-up (measures switching)")
    p.add_argument("--concurrent", action="store_true",
                   help="send both aliases at the same time (measures contention)")
    p.add_argument("--reverse", action="store_true",
                   help="reverse the alias order (local-reasoning first); in --concurrent mode its "
                        "thread starts first, so it wins the race for a single model slot")
    args = p.parse_args()

    key, delete_key = create_bench_key()
    client = OpenAI(base_url=f"{GATEWAY}/v1", api_key=key, max_retries=0, timeout=600)
    aliases = list(WORKLOADS)[::-1] if args.reverse else list(WORKLOADS)
    rows: list[dict] = []
    try:
        if args.concurrent:
            run_concurrent(client, aliases)                     # warm-up, not recorded
            for i in range(args.runs):
                print(f"concurrent round {i + 1}/{args.runs}", flush=True)
                rows += run_concurrent(client, aliases)
        elif args.order == "grouped":
            for alias in aliases:
                one_request(client, alias)                      # warm-up, not recorded
                for i in range(args.runs):
                    print(f"{alias} {i + 1}/{args.runs}", flush=True)
                    rows.append(one_request(client, alias))
        else:
            for i in range(args.runs):
                for alias in aliases:
                    print(f"{alias} {i + 1}/{args.runs}", flush=True)
                    rows.append(one_request(client, alias))
    finally:
        delete_key()

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    mode = ("concurrent" if args.concurrent else args.order) + ("-reverse" if args.reverse else "")
    with RESULTS.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"time": stamp, "label": args.label, "mode": mode, **r}) + "\n")

    print(f"\n== {args.label} ({mode}) ==")
    summarise(rows)
    print(f"\nAppended {len(rows)} rows to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
