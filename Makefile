# ==============================================================================
# Makefile for Deakin University Workshop Repository
# ==============================================================================

SHELL := /bin/bash
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python3

.PHONY: help setup ensure-venv prefix-caching-demo supply-chain-security-demo exploit-pickle-demo harden-safetensors-demo cleanup cleanup-all

help:
	@echo "=========================================================================================="
	@echo "🎯 Deakin University Generative AI Workshop: Execution Targets"
	@echo "=========================================================================================="
	@echo "  make setup                      : Bootstrap isolated Python virtual environment (.venv) and install dependencies"
	@echo "  make prefix-caching-demo        : Run Demo 1 (Automatic Prefix Caching vs. Naive Prefill A/B Comparison)"
	@echo "  make supply-chain-security-demo : Run Demo 2 (Pickle Deserialization Exploit vs. SafeTensors Defense)"
	@echo "  make exploit-pickle-demo        : Run Demo 2A individually (PyTorch legacy pickle Remote Code Execution [RCE] exploit)"
	@echo "  make harden-safetensors-demo    : Run Demo 2B individually (SafeTensors zero-copy memory-mapped defense)"
	@echo "  make cleanup                    : Clean up generated model checkpoints and security canary files"
	@echo "  make cleanup-all                : Complete teardown including the Python virtual environment (.venv)"
	@echo "=========================================================================================="

ensure-venv:
	@if [ ! -f "$(PYTHON)" ]; then \
		echo "❌ Virtual environment not found. Running 'make setup' first..."; \
		./setup.sh; \
	fi

setup:
	@chmod +x setup.sh cleanup.sh
	@./setup.sh

prefix-caching-demo: ensure-venv
	@$(PYTHON) demo/01-inference-scaling/benchmark_prefix_caching.py

exploit-pickle-demo: ensure-venv
	@echo "--- Step 2A: The Vulnerable PyTorch Pickle Deserialization Exploit ---"
	@$(PYTHON) demo/02-supply-chain-security/exploit_pickle.py

harden-safetensors-demo: ensure-venv
	@echo "--- Step 2B: The SafeTensors Architectural Defense ---"
	@$(PYTHON) demo/02-supply-chain-security/harden_safetensors.py

supply-chain-security-demo: exploit-pickle-demo harden-safetensors-demo

cleanup:
	@./cleanup.sh

cleanup-all:
	@./cleanup.sh --all
