# Starts llama-server for the LiteLLM lab.
# Reads the shared API key from .env so the server never starts unprotected.
#
# Usage (from anywhere):
#   C:\agent-system\litellm-lab\scripts\start-llama.ps1                       # default: Qwen3-4B on :8080
#   ...\start-llama.ps1 -Model Qwen3-8B-Q4_K_M.gguf -Alias qwen3-8b            # another model
#   ...\start-llama.ps1 -Model Qwen3-8B-Q4_K_M.gguf -Alias qwen3-8b -Port 8081 # second server

param(
    [string]$Model = 'Qwen3-4B-Q4_K_M.gguf',   # file name inside models\
    [string]$Alias = 'qwen3-4b',               # model id the server reports; LiteLLM's backend model name
    [int]$Port     = 8080,
    [int]$Ctx      = 8192
)

$ErrorActionPreference = 'Stop'

$root      = Split-Path -Parent $PSScriptRoot          # project root (parent of scripts\)
$envFile   = Join-Path $root '.env'
$modelPath = Join-Path $root "models\$Model"

# --- Load the API key from .env (never printed) ---
$line = Get-Content $envFile | Where-Object { $_ -match '^\s*LLAMACPP_API_KEY\s*=' }
if (-not $line) { throw "LLAMACPP_API_KEY not found in $envFile" }
$key = ($line -replace '^\s*LLAMACPP_API_KEY\s*=\s*', '').Trim()
if ($key.Length -lt 32) { throw "LLAMACPP_API_KEY looks too short - refusing to start" }

if (-not (Test-Path $modelPath)) { throw "Model not found: $modelPath" }

Write-Host "Starting llama-server: $Model as '$Alias' on 127.0.0.1:$Port (ctx $Ctx)"

# --- Run the server; clear the key from this session when it stops ---
try {
    $env:LLAMA_API_KEY = $key
    & llama-server -m $modelPath `
        --alias $Alias `
        --host 127.0.0.1 `
        --port $Port `
        -ngl 99 `
        -c $Ctx `
        --jinja
}
finally {
    Remove-Item Env:LLAMA_API_KEY -ErrorAction SilentlyContinue
    Remove-Variable key -ErrorAction SilentlyContinue
}
