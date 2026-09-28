"""
================================================================================
Shared Utilities & Constants for Supply Chain Security Demonstrations
Provides centralized logging, canary lifecycle management, and base tensor state
for Demo 2A (Pickle Deserialization) and Demo 2B (SafeTensors Hardening).
================================================================================
Deakin University Workshop: Building and Scaling Generative AI
Author: Anmol Krishan Sachdeva (@greatdevaks)
Disclaimer: For technical education and academic demonstration only.
================================================================================
"""

import logging
import os
import sys

import torch

CANARY_FILE = "HACKED_DEMO_CANARY.txt"
VULNERABLE_CHECKPOINT = "vulnerable_model.bin"
SAFE_CHECKPOINT = "safe_model.safetensors"


def configure_logger(name: str) -> logging.Logger:
    """Configures and returns a structured standard-output (stdout) logger for security modules."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def purge_canary_file(logger: logging.Logger, canary_path: str = CANARY_FILE) -> None:
    """Removes any residual security canary file prior to demonstration execution."""
    if os.path.exists(canary_path):
        try:
            os.remove(canary_path)
        except OSError as exc:
            logger.warning("Could not remove existing canary file %s: %s", canary_path, exc)


def build_reference_weights() -> dict[str, torch.Tensor]:
    """Constructs synthetic neural network projection tensors shared across Architecture A and B tests."""
    return {
        "model.embed_tokens.weight": torch.randn(64, 128),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
    }
