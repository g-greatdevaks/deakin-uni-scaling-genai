#!/usr/bin/env bash
# ==============================================================================
# Teardown & Cleanup Script (Linux & macOS)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "🧹 Cleaning up demo artifacts and temporary checkpoints..."

# Remove generated model files, canaries, and tool caches
rm -f vulnerable_model.bin
rm -f safe_model.safetensors
rm -f HACKED_DEMO_CANARY.txt
rm -rf __pycache__ demo/*/__pycache__ .ruff_cache .pytest_cache

# Optional: Remove .venv when --all is passed
if [ "${1:-}" == "--all" ]; then
  echo "🗑 Removing .venv..."
  rm -rf .venv
  echo "✔ Full cleanup complete (including virtual environment)."
else
  echo "✔ Demo artifacts removed. (.venv preserved. Use './cleanup.sh --all' to delete .venv)."
fi
