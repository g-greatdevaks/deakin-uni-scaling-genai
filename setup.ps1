# ==============================================================================
# Setup Script for Deakin University Workshop (Windows PowerShell)
# Isolated Python Virtual Environment Bootstrap (Central Processing Unit [CPU]-Native)
# ==============================================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🚀 Bootstrapping Python Virtual Environment (.venv on Windows)..." -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Verify python availability and minimum version (3.11+)
try {
    $pyVersion = python --version
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Error: Python 3.11+ is required, but found $pyVersion."
        exit 1
    }
    Write-Host "✔ Found Python: $pyVersion" -ForegroundColor Green
} catch {
    Write-Error "❌ Error: Python 3.11+ is required and must be available in PATH (install from python.org or Microsoft Store)."
    exit 1
}

# 2. Create isolated virtual environment
if (Test-Path -Path ".venv") {
    Write-Host "ℹ Existing .venv detected. Reusing existing environment." -ForegroundColor Yellow
} else {
    Write-Host "📦 Creating virtual environment at .venv..." -ForegroundColor Yellow
    python -m venv .venv
}

# 3. Activate virtual environment
Write-Host "🔌 Activating .venv..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# 4. Upgrade pip
Write-Host "⬆ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# 5. Install dependencies from requirements.txt
Write-Host "📥 Installing dependencies from requirements.txt..." -ForegroundColor Yellow
python -m pip install --quiet --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 6. Verify environment health
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "✔ Environment Verified Successfully!" -ForegroundColor Green
python -c "import torch, safetensors, rich; print(f'  • PyTorch:     {torch.__version__} (Device: {torch.device(\"cpu\")})'); print(f'  • SafeTensors: {safetensors.__version__}'); print(f'  • Rich:        {rich.__version__}')"
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "🎯 Environment bootstrap complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
