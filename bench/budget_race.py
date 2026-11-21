"""Budget race: are simultaneous requests admitted when a key is just below its budget?

LiteLLM checks the budget BEFORE a request runs and adds the cost AFTER it finishes.
Sequentially that means at most one request of overshoot. This script tests the
concurrent case:

  1. measure the cost c of one fixed request (throwaway key A)
  2. create key B with max_budget = 1.5 * c, spend c on it -> 0.5 * c left
  3. fire N requests on key B at the same moment
  4. fire one more request -> should be blocked
  5. report status codes and final spend vs budget

A second mode (--max-tokens) asks whether requests in flight are counted by their
max_tokens (worst case) or by something smaller. Budget $0.004, below the worst case of a
2000-token request ($0.008) but far above the real cost of a short answer:

  part 1: one short-answer request with max_tokens=2000 on a fresh key
          -> rejected means even a lone request is judged by its worst case
  part 2: a long request (max_tokens=2000) runs; 3 s later a tiny request on the same key
          -> tiny one rejected means the long one counts as its max_tokens while in flight

Usage (from the project root, .venv active, stack up):
  python bench\\budget_race.py            # N = 2
  python bench\\budget_race.py --n 4
  python bench\\budget_race.py --max-tokens
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = "http://127.0.0.1:4000"
RESULTS = Path(__file__).resolve().parent / "budget_race.jsonl"
MODEL = "local-fast"
MAX_TOKENS = 64
# A prompt that always runs past MAX_TOKENS, so every request costs the same
PROMPT = "Count from 1 to 500, separated by spaces. Output only the numbers."

ADMIN = {"Authorization": f"Bearer {dotenv_values(ROOT / '.env')['LITELLM_MASTER_KEY']}"}


def new_key(max_budget: float) -> str:
    r = httpx.post(f"{GATEWAY}/key/generate", headers=ADMIN, timeout=30,
                   json={"key_alias": f"budget-race-{uuid.uuid4().hex[:8]}",
                         "models": [MODEL], "rpm_limit": 120, "max_budget": max_budget})
    r.raise_for_status()
    return r.json()["key"]


def delete_keys(keys: list[str]) -> None:
    httpx.post(f"{GATEWAY}/key/delete", headers=ADMIN, json={"keys": keys}, timeout=30)


def chat(key: str, prompt: str = PROMPT, max_tokens: int = MAX_TOKENS) -> dict:
    """One blocking request. Returns status, cost header, tokens and a short error."""
    r = httpx.post(f"{GATEWAY}/v1/chat/completions", timeout=300,
                   headers={"Authorization": f"Bearer {key}"},
                   json={"model": MODEL, "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": prompt}]})
    out = {"status": r.status_code, "cost": None, "completion_tokens": None, "error": None}
    if r.status_code == 200:
        out["cost"] = float(r.headers.get("x-litellm-response-cost", "nan"))
        out["completion_tokens"] = r.json()["usage"]["completion_tokens"]
    else:
        out["error"] = r.text[:160]
    return out


def key_spend(key: str) -> tuple[float, float]:
    r = httpx.get(f"{GATEWAY}/key/info", headers=ADMIN, params={"key": key}, timeout=30)
    r.raise_for_status()
    info = r.json()["info"]
    return float(info.get("spend") or 0.0), float(info.get("max_budget") or 0.0)


def wait_for_spend(key: str, at_least: float, timeout: float = 60) -> float:
    """Spend is written asynchronously; poll until it reaches the expected value."""
    deadline = time.monotonic() + timeout
    while True:
        spend, _ = key_spend(key)
        if spend >= at_least * 0.999 or time.monotonic() > deadline:
            return spend
        time.sleep(1)


def burst(key: str, n: int) -> list[dict]:
    """n requests released at the same instant by a barrier."""
    barrier = threading.Barrier(n)
    results: list[dict] = [None] * n

    def worker(i: int) -> None:
        barrier.wait()
        results[i] = chat(key)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def max_tokens_mode() -> None:
    budget, big = 0.004, 2000
    short = "Reply with just: OK"
    worst = big * 0.000004
    print(f"budget ${budget:.4f}; worst case of a max_tokens={big} request >= ${worst:.4f}")
    keys: list[str] = []
    try:
        # part 1: a lone short request with a big max_tokens
        k1 = new_key(budget); keys.append(k1)
        r1 = chat(k1, short, big)
        print(f"\npart 1, lone request, max_tokens={big}: {r1['status']}  "
              f"cost={r1['cost']}  tokens={r1['completion_tokens']}  {r1['error'] or ''}")

        # part 2: tiny request while a long one is in flight
        k2 = new_key(budget); keys.append(k2)
        long_result: dict = {}
        t = threading.Thread(target=lambda: long_result.update(chat(k2, PROMPT, big)))
        t0 = time.monotonic()
        t.start()
        time.sleep(3)
        tiny = chat(k2, short, 8)
        tiny_at = time.monotonic() - t0
        t.join()
        long_s = time.monotonic() - t0
        print(f"\npart 2, long request in flight (max_tokens={big}):")
        print(f"  tiny request at {tiny_at:.1f} s: {tiny['status']}  cost={tiny['cost']}  {tiny['error'] or ''}")
        print(f"  long request: {long_result['status']}  tokens={long_result['completion_tokens']}  "
              f"cost={long_result['cost']}  took {long_s:.0f} s  {long_result['error'] or ''}")
        if tiny_at > long_s:
            print("WARNING: the long request finished before the tiny one was sent; part 2 is invalid")
        spend2, _ = key_spend(k2)
        print(f"  key spend now ${spend2:.6f} (may lag a few seconds)")

        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "mode": "max-tokens", "budget": budget, "max_tokens": big,
                                "part1_status": r1["status"], "part1_cost": r1["cost"],
                                "part2_tiny_status": tiny["status"], "part2_tiny_at_s": round(tiny_at, 1),
                                "part2_long_status": long_result["status"],
                                "part2_long_tokens": long_result["completion_tokens"],
                                "part2_long_cost": long_result["cost"]}) + "\n")
    finally:
        delete_keys(keys)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=2, help="simultaneous requests in the burst")
    p.add_argument("--max-tokens", action="store_true",
                   help="test whether in-flight requests count by their max_tokens")
    args = p.parse_args()
    if args.max_tokens:
        max_tokens_mode()
        return

    keys: list[str] = []
    try:
        # 1. cost of one request
        a = new_key(1.0); keys.append(a)
        probe = chat(a)
        if probe["status"] != 200:
            raise SystemExit(f"probe failed: {probe}")
        c = probe["cost"]
        print(f"cost of one request: ${c:.6f} ({probe['completion_tokens']} completion tokens)")

        # 2. key B: budget 1.5c, spend c
        budget = round(1.5 * c, 8)
        b = new_key(budget); keys.append(b)
        first = chat(b)
        if first["status"] != 200:
            raise SystemExit(f"first request on key B failed: {first}")
        spend_before = wait_for_spend(b, c)
        print(f"key B: budget ${budget:.6f}, spend before burst ${spend_before:.6f} "
              f"({spend_before / budget:.0%} of budget)")

        # 3. the burst
        results = burst(b, args.n)
        # 4. one more, after the burst
        after = chat(b)

        admitted = [r for r in results if r["status"] == 200]
        expected = spend_before + sum(r["cost"] for r in admitted)
        spend_after = wait_for_spend(b, expected)
        time.sleep(5)                                   # let any late write land
        spend_after, _ = key_spend(b)

        print(f"\nburst of {args.n}:")
        for i, r in enumerate(results, 1):
            print(f"  request {i}: {r['status']}  tokens={r['completion_tokens']}  {r['error'] or ''}")
        print(f"request after burst: {after['status']}  {after['error'] or ''}")
        print(f"\nadmitted in burst: {len(admitted)}/{args.n}")
        print(f"final spend ${spend_after:.6f} = {spend_after / budget:.0%} of budget "
              f"(overshoot = {(spend_after - budget) / c:+.2f} requests)")
        uneven = [r for r in admitted + [first, probe] if r["completion_tokens"] != MAX_TOKENS]
        if uneven:
            print("WARNING: some requests stopped before max_tokens, so costs are not identical")

        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "n": args.n, "cost_per_request": c, "budget": budget,
                                "spend_before": spend_before, "spend_after": spend_after,
                                "burst_status": [r["status"] for r in results],
                                "after_status": after["status"]}) + "\n")
    finally:
        delete_keys(keys)


if __name__ == "__main__":
    main()
