# Starts the llama.cpp router (option B2): one endpoint, models loaded on demand,
# both Qwen3 models kept resident. Chosen by measurement - see bench/ and README.
#
# Usage (from anywhere):  C:\agent-system\litellm-lab\scripts\start-router.ps1
#
# Security / governance:
#   - binds 127.0.0.1 only; requires LLAMACPP_API_KEY (children inherit it)
#   - serves ONLY the models in config\models.ini: the router also scans llama.cpp's
#     download cache and has no flag to disable that, so LLAMA_CACHE is pointed at an
#     empty folder for this process (the real cache is untouched)

param(
    [int]$MaxModels = 2,      # 2 = B2 (both resident); 1 = B1 (swap)
    [int]$Port      = 8080
)

$ErrorActionPreference = 'Stop'

$root    = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root '.env'
$preset  = Join-Path $root 'config\models.ini'

# --- Load the API key from .env (never printed) ---
$line = Get-Content $envFile | Where-Object { $_ -match '^\s*LLAMACPP_API_KEY\s*=' }
if (-not $line) { throw "LLAMACPP_API_KEY not found in $envFile" }
$key = ($line -replace '^\s*LLAMACPP_API_KEY\s*=\s*', '').Trim()
if ($key.Length -lt 32) { throw "LLAMACPP_API_KEY looks too short - refusing to start" }

if (-not (Test-Path $preset)) { throw "Preset file not found: $preset" }

# --- Empty cache folder so only presets are served ---
$emptyCache = Join-Path $env:LOCALAPPDATA 'litellm-lab\empty-llama-cache'
New-Item -ItemType Directory -Force $emptyCache | Out-Null
if (Get-ChildItem $emptyCache -Force | Select-Object -First 1) {
    throw "$emptyCache is not empty - refusing to start (the router would serve its contents)"
}

Write-Host "Starting llama.cpp router on 127.0.0.1:$Port (models-max $MaxModels, presets: config\models.ini)"

# Preset model paths are relative to the project root
Push-Location $root
$oldCache = $env:LLAMA_CACHE
try {
    $env:LLAMA_API_KEY = $key
    $env:LLAMA_CACHE   = $emptyCache
    & llama-server --models-preset $preset `
        --models-max $MaxModels `
        --host 127.0.0.1 `
        --port $Port
}
finally {
    Remove-Item Env:LLAMA_API_KEY -ErrorAction SilentlyContinue
    if ($oldCache) { $env:LLAMA_CACHE = $oldCache } else { Remove-Item Env:LLAMA_CACHE -ErrorAction SilentlyContinue }
    Remove-Variable key -ErrorAction SilentlyContinue
    Pop-Location
}
