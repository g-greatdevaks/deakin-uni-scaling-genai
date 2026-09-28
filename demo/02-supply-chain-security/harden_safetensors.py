#!/usr/bin/env python3
"""
================================================================================
Demo 2B: The SafeTensors Architectural Defense
Hardened Architecture: Zero-Copy Memory-Mapped Deserialization
References:
  - Hugging Face SafeTensors Specification (https://huggingface.co/docs/safetensors)
  - Open Worldwide Application Security Project (OWASP) Top 10 for Large Language
    Model (LLM) Applications (LLM03: Supply Chain).
================================================================================
Deakin University Workshop: Building and Scaling Generative AI
Author: Anmol Krishan Sachdeva (@greatdevaks)
Disclaimer: For technical education and academic demonstration only.
================================================================================
"""

import json
import os
import struct
import sys
from typing import Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from safetensors.torch import load_file, save_file
from security_common import (
    CANARY_FILE,
    SAFE_CHECKPOINT,
    build_reference_weights,
    configure_logger,
    purge_canary_file,
)

logger = configure_logger("safetensors_hardening_demo")
console = Console()


# ------------------------------------------------------------------------------
# 1. Custom Exceptions
# ------------------------------------------------------------------------------
class HardeningDemoError(Exception):
    """Base exception for errors during the SafeTensors hardening demonstration."""


class SafeTensorsSaveError(HardeningDemoError):
    """Raised when converting tensors to the SafeTensors binary format fails."""


class SafeTensorsLoadError(HardeningDemoError):
    """Raised when inspecting or loading the SafeTensors binary fails."""


# ------------------------------------------------------------------------------
# 2. Hardened Model Serialization
# ------------------------------------------------------------------------------
def generate_safetensors_checkpoint(target_path: str = SAFE_CHECKPOINT) -> int:
    """
    Serializes genuine neural network tensors into the SafeTensors binary format.

    Args:
        target_path: Filesystem path for the resulting .safetensors file.

    Returns:
        The total byte size of the written file.

    Raises:
        SafeTensorsSaveError: If serialization encounters a filesystem or format error.
    """
    try:
        console.print(
            f"[bold yellow]Step 1: Serializing tensors into SafeTensors format: '{target_path}'...[/bold yellow]"
        )

        model_weights: dict[str, torch.Tensor] = build_reference_weights()

        save_file(model_weights, target_path)
        file_size = os.path.getsize(target_path)
        logger.info("Saved SafeTensors checkpoint: %s (Size: %d bytes)", target_path, file_size)
        console.print(f"✔ Checkpoint written to [cyan]{target_path}[/cyan] ({file_size:,} bytes).\n")
        return file_size
    except Exception as exc:
        logger.error("Failed to serialize model to SafeTensors: %s", exc)
        raise SafeTensorsSaveError("SafeTensors serialization routine failed.") from exc


# ------------------------------------------------------------------------------
# 3. Binary Layout Inspection
# ------------------------------------------------------------------------------
def inspect_binary_layout(target_path: str = SAFE_CHECKPOINT) -> tuple[int, dict[str, Any]]:
    """
    Parses and displays the internal structure of the SafeTensors file.

    Args:
        target_path: Path to the .safetensors file.

    Returns:
        A tuple containing the header byte size and the parsed JavaScript Object Notation (JSON)
        metadata dictionary.

    Raises:
        SafeTensorsLoadError: If header unpacking fails.
    """
    try:
        console.print("[bold white]Step 2: Inspecting SafeTensors Binary Layout (Under the Hood):[/bold white]")
        file_size = os.path.getsize(target_path)

        with open(target_path, "rb") as file_handle:
            # First 8 bytes store unsigned 64-bit integer specifying JSON header length
            header_size_bytes = file_handle.read(8)
            if len(header_size_bytes) < 8:
                raise ValueError("Corrupt file: Header contains fewer than 8 bytes.")
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_json = file_handle.read(header_size).decode("utf-8")
            parsed_metadata = json.loads(header_json)

        table = Table(title="SafeTensors Binary Layout (Declarative Header & Raw Byte Buffers)")
        table.add_column("Byte Range", style="dim", justify="right")
        table.add_column("Segment", style="bold cyan")
        table.add_column("Description & Technical Role", style="white")

        table.add_row(
            "0 - 7",
            "Header Length (8 bytes)",
            f"Little-endian Unsigned 64-bit Integer (uint64) (Value: {header_size} bytes)",
        )
        table.add_row(
            f"8 - {8 + header_size - 1}",
            "Metadata (JSON)",
            "Strict tensor schemas (dtype, shape, offset pointers in JavaScript Object Notation / JSON)",
        )
        table.add_row(
            f"{8 + header_size} - {file_size - 1}",
            "Raw Binary Buffers",
            "Continuous float data mapped directly via Operating System (OS) memory-mapped files (mmap)",
        )

        console.print(table)
        preview_text = json.dumps(parsed_metadata, indent=2)[:320]
        console.print(f"\n[dim]Schema Preview:[/dim]\n[cyan]{preview_text}...[/cyan]\n")
        return header_size, parsed_metadata
    except Exception as exc:
        logger.error("Failed to parse binary layout of %s: %s", target_path, exc)
        raise SafeTensorsLoadError("Failed to unpack SafeTensors header.") from exc


# ------------------------------------------------------------------------------
# 4. Ingestion & Deserialization Verification
# ------------------------------------------------------------------------------
def verify_hardened_ingestion(target_path: str = SAFE_CHECKPOINT) -> None:
    """
    Loads weights using SafeTensors and verifies that no deserialization hooks execute.

    Args:
        target_path: Path to the .safetensors file.

    Raises:
        SafeTensorsLoadError: If loading tensors fails.
    """
    try:
        console.print("[bold green]Step 3: Loading tensors via 'safetensors.torch.load_file()' ...[/bold green]")
        loaded_tensors = load_file(target_path)
        logger.info("Successfully loaded %d tensors via memory mapping", len(loaded_tensors))

        console.print(f"✔ Successfully memory-mapped {len(loaded_tensors)} tensors:")
        for name, tensor in loaded_tensors.items():
            console.print(f"   • [cyan]{name}[/cyan] | Shape: {list(tensor.shape)} | Dtype: {tensor.dtype}")

        # Confirm that no arbitrary execution canary was triggered
        if os.path.exists(CANARY_FILE):
            logger.error("Security canary unexpectedly detected!")
            console.print("[bold red]Anomalous canary detected during SafeTensors load![/bold red]")
        else:
            console.print(Panel(
                "[bold green]VERIFICATION PASSED: NO DESERIALIZATION HOOKS EXECUTED[/bold green]\n\n"
                "[bold white]Architectural Principles of SafeTensors:[/bold white]\n"
                "  1. [bold]Absence of an Execution Virtual Machine (VM):[/bold] "
                "The parser evaluates only declarative JSON metadata and raw numerical bytes.\n"
                "  2. [bold]Zero-Copy Memory Mapping (mmap):[/bold] File byte ranges are mapped directly to virtual\n"
                "     memory pages, avoiding intermediate Python heap copies during initialization.\n"
                "  3. [bold]Header Bounds Validation:[/bold] Out-of-bounds byte offsets or invalid data types\n"
                "     trigger a schema validation error before tensor memory buffers are accessed.",
                title="[bold green]Supply Chain Hardening Summary[/bold green]",
                border_style="green",
            ))
    except Exception as exc:
        logger.error("Failed during SafeTensors load verification: %s", exc)
        raise SafeTensorsLoadError("Verification routine encountered an error.") from exc


def main() -> None:
    """Coordinates the SafeTensors creation, structural analysis, and ingestion verification."""
    try:
        console.print(Panel.fit(
            "[bold green]DEMO 2B: SAFETENSORS ARCHITECTURAL DEFENSE[/bold green]\n"
            "[dim]Architecture B: Memory-Mapped Deserialization & Structural Hardening[/dim]",
            border_style="green",
        ))

        purge_canary_file(logger, CANARY_FILE)
        generate_safetensors_checkpoint(SAFE_CHECKPOINT)
        inspect_binary_layout(SAFE_CHECKPOINT)
        verify_hardened_ingestion(SAFE_CHECKPOINT)
    except HardeningDemoError as exc:
        logger.error("Hardening demonstration halted: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Unexpected failure in hardening routine: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
