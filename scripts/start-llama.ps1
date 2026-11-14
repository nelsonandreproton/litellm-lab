# Starts llama-server for the LiteLLM lab.
# Reads the shared API key from .env so the server never starts unprotected.
# Usage (from anywhere):  C:\agent-system\litellm-lab\scripts\start-llama.ps1

$ErrorActionPreference = 'Stop'

$root    = Split-Path -Parent $PSScriptRoot          # project root (parent of scripts\)
$envFile = Join-Path $root '.env'
$model   = Join-Path $root 'models\Qwen3-4B-Q4_K_M.gguf'

# --- Load the API key from .env (never printed) ---
$line = Get-Content $envFile | Where-Object { $_ -match '^\s*LLAMACPP_API_KEY\s*=' }
if (-not $line) { throw "LLAMACPP_API_KEY not found in $envFile" }
$key = ($line -replace '^\s*LLAMACPP_API_KEY\s*=\s*', '').Trim()
if ($key.Length -lt 32) { throw "LLAMACPP_API_KEY looks too short - refusing to start" }

if (-not (Test-Path $model)) { throw "Model not found: $model" }

# --- Run the server; clear the key from this session when it stops ---
try {
    $env:LLAMA_API_KEY = $key
    & llama-server -m $model `
        --alias qwen3-4b `
        --host 127.0.0.1 `
        --port 8080 `
        -ngl 99 `
        -c 8192 `
        --jinja
}
finally {
    Remove-Item Env:LLAMA_API_KEY -ErrorAction SilentlyContinue
    Remove-Variable key -ErrorAction SilentlyContinue
}
