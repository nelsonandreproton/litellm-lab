"""Shared constants for the test suite."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GATEWAY = "http://127.0.0.1:4000"   # 127.0.0.1, not localhost (IPv6 fallback costs ~2 s on Windows)
LLAMA = "http://127.0.0.1:8080"
# Public gateway aliases (what clients and keys use)
MODEL = "local-fast"
REASONING_MODEL = "local-reasoning"
# Models the llama.cpp router may serve (allow-list in config/models.ini)
BACKEND_MODELS = {"Qwen3-4B-Q4_K_M", "Qwen3-8B-Q4_K_M"}

# Must match the notional pricing in config/config.yaml
INPUT_COST_PER_TOKEN = 0.000001
OUTPUT_COST_PER_TOKEN = 0.000004
