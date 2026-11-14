"""How much latency does the LiteLLM gateway add?

Sends the same request, alternately, straight to llama.cpp and through LiteLLM,
with the same HTTP client and the same sampling settings, and compares time to
first token (TTFT) and total time. Alternating cancels out slow drift (heat,
background load); llama.cpp's prompt cache sees the same prompt on both paths.

Usage (from the project root, .venv active, stack running):
  python bench\\gateway_overhead.py                 # 20 pairs
  python bench\\gateway_overhead.py --pairs 40
Results are printed and appended to bench/overhead.jsonl.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = "http://127.0.0.1:4000"
LLAMA = "http://127.0.0.1:8080"
RESULTS = Path(__file__).resolve().parent / "overhead.jsonl"

PROMPT = "Explain what an API gateway does, in two sentences."
MAX_TOKENS = 32

# What LiteLLM sends for the local-fast alias (config/config.yaml), so both paths
# ask llama.cpp for exactly the same thing
DIRECT_BODY = {
    "model": "Qwen3-4B-Q4_K_M",
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
GATEWAY_BODY = {"model": "local-fast"}


def one_request(client: httpx.Client, url: str, key: str, extra: dict) -> dict:
    body = {**extra, "max_tokens": MAX_TOKENS, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": PROMPT}]}
    t0 = time.perf_counter()
    first = None
    tokens = None
    with client.stream("POST", f"{url}/v1/chat/completions",
                       headers={"Authorization": f"Bearer {key}"}, json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                tokens = event["usage"].get("completion_tokens")
            for choice in event.get("choices", []):
                if first is None and (choice.get("delta") or {}).get("content"):
                    first = time.perf_counter()
    t1 = time.perf_counter()
    return {"ttft_ms": round((first - t0) * 1000, 1) if first else None,
            "total_ms": round((t1 - t0) * 1000, 1), "tokens": tokens}


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=int, default=20)
    args = p.parse_args()

    env = dotenv_values(ROOT / ".env")
    llama_key = env["LLAMACPP_API_KEY"]
    admin = {"Authorization": f"Bearer {env['LITELLM_MASTER_KEY']}"}

    with httpx.Client(timeout=120) as client:
        r = client.post(f"{GATEWAY}/key/generate", headers=admin,
                        json={"key_alias": f"overhead-{uuid.uuid4().hex[:8]}",
                              "models": ["local-fast"], "rpm_limit": 600, "max_budget": 0.10})
        r.raise_for_status()
        gw_key = r.json()["key"]
        rows = []
        try:
            for _ in range(3):  # warm-up both paths, not recorded
                one_request(client, LLAMA, llama_key, DIRECT_BODY)
                one_request(client, GATEWAY, gw_key, GATEWAY_BODY)
            for i in range(args.pairs):
                print(f"pair {i + 1}/{args.pairs}", flush=True)
                order = [("direct", LLAMA, llama_key, DIRECT_BODY), ("gateway", GATEWAY, gw_key, GATEWAY_BODY)]
                if i % 2:  # alternate which path goes first
                    order.reverse()
                for path, url, key, extra in order:
                    rows.append({"path": path, **one_request(client, url, key, extra)})
        finally:
            client.post(f"{GATEWAY}/key/delete", headers=admin, json={"keys": [gw_key]})

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with RESULTS.open("a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"time": stamp, **r}) + "\n")

    print(f"\n{'path':<9}{'n':>4}{'ttft med':>11}{'ttft p90':>11}{'total med':>12}{'tokens med':>12}")
    med = {}
    for path in ("direct", "gateway"):
        rs = [r for r in rows if r["path"] == path]
        ttft = [r["ttft_ms"] for r in rs if r["ttft_ms"] is not None]
        total = [r["total_ms"] for r in rs]
        toks = [r["tokens"] for r in rs if r["tokens"] is not None]
        med[path] = (statistics.median(ttft), statistics.median(total))
        print(f"{path:<9}{len(rs):>4}{statistics.median(ttft):>9.1f}ms{pct(ttft, 0.9):>9.1f}ms"
              f"{statistics.median(total):>10.1f}ms{statistics.median(toks) if toks else 0:>12}")
    print(f"\ngateway overhead (median): TTFT +{med['gateway'][0] - med['direct'][0]:.1f} ms, "
          f"total +{med['gateway'][1] - med['direct'][1]:.1f} ms")
    print(f"Appended {len(rows)} rows to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
