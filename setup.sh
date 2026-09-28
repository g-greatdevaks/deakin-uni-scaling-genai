#!/usr/bin/env bash
# ==============================================================================
# Setup Script for Deakin University Workshop (Linux and macOS)
# Isolated Python Virtual Environment Bootstrap (Central Processing Unit [CPU]-Native)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "🚀 Bootstrapping Python Virtual Environment (.venv)..."
echo "======================================================================"

# 1. Verify python3 availability and minimum version (3.11+)
if ! command -v python3 &>/dev/null; then
  echo "❌ Error: python3 is not installed or not in PATH."
  echo "Python 3.11+ is required (install via Homebrew on macOS or Advanced Package Tool [apt] on Linux)."
  exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "❌ Error: Python 3.11+ is required, but found Python ${PYTHON_VERSION}."
  exit 1
fi
echo "✔ Found Python: $(python3 --version) (Version: ${PYTHON_VERSION})"

# 2. Create isolated virtual environment
if [ -d ".venv" ]; then
  echo "ℹ Existing .venv detected. Reusing existing environment."
else
  echo "📦 Creating virtual environment at .venv..."
  python3 -m venv .venv
fi

# 3. Activate virtual environment
echo "🔌 Activating .venv..."
# shellcheck source=/dev/null
source .venv/bin/activate

# 4. Upgrade pip quietly
echo "⬆ Upgrading pip..."
pip install --upgrade pip --quiet

# 5. Install dependencies from requirements.txt
echo "📥 Installing dependencies from requirements.txt..."
# On Linux, append the PyTorch CPU wheel index to avoid downloading multi-gigabyte Compute Unified Device Architecture (CUDA) packages
if [[ "$(uname -s)" == "Linux" ]]; then
  pip install --quiet --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
else
  # macOS natively installs Metal Performance Shaders (MPS) and CPU wheels directly from the Python Package Index (PyPI)
  pip install --quiet -r requirements.txt
fi

# 6. Verify environment health
echo ""
echo "======================================================================"
echo "✔ Environment Verified Successfully!"
python3 -c "
import importlib.metadata, torch, safetensors, rich
print(f'  • PyTorch:     {torch.__version__} (Device: {torch.device(\"cpu\")})')
print(f'  • SafeTensors: {safetensors.__version__}')
print(f'  • Rich:        {importlib.metadata.version(\"rich\")}')
"
echo "======================================================================"
echo "🎯 Environment bootstrap complete. Activate with: source .venv/bin/activate"
echo "======================================================================"
