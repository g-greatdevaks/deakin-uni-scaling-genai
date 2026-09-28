# ==============================================================================
# Teardown & Cleanup Script (Windows PowerShell)
# ==============================================================================

param (
    [switch]$All
)

Set-Location -Path $PSScriptRoot

Write-Host "🧹 Cleaning up demo artifacts and temporary checkpoints..." -ForegroundColor Yellow

# Remove generated model files, canaries, and tool caches
Remove-Item -Path "vulnerable_model.bin" -ErrorAction SilentlyContinue
Remove-Item -Path "safe_model.safetensors" -ErrorAction SilentlyContinue
Remove-Item -Path "HACKED_DEMO_CANARY.txt" -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Include __pycache__, .ruff_cache, .pytest_cache -Recurse -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

if ($All) {
    Write-Host "🗑 Removing .venv..." -ForegroundColor Red
    Remove-Item -Path ".venv" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "✔ Full cleanup complete (including virtual environment)." -ForegroundColor Green
} else {
    Write-Host "✔ Demo artifacts removed. (.venv preserved. Use '.\cleanup.ps1 -All' to delete .venv)." -ForegroundColor Green
}
