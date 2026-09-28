#!/usr/bin/env python3
"""
================================================================================
Benchmark Automatic Prefix Caching (APC) vs. Naive Prefill Recomputation
Architecture A (Naive Recomputation) vs. Architecture B (Key-Value / KV-Cache Reuse)
References:
  - Kwon et al. (2023), "Efficient Memory Management for Large Language Model
    Serving with PagedAttention", Association for Computing Machinery Symposium
    on Operating Systems Principles (ACM SOSP '23).
  - Zheng et al. (2024), "SGLang: Efficient Execution of Structured Language
    Model Programs" (RadixAttention), arXiv:2312.07104.
================================================================================
Deakin University Workshop: Building and Scaling Generative AI
Author: Anmol Krishan Sachdeva (@greatdevaks)
Disclaimer: For technical education and academic demonstration only.
================================================================================
"""

import logging
import math
import sys
import time
from dataclasses import dataclass

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch import nn

# Configure structured logging
logger = logging.getLogger("inference_scaling_benchmark")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)

console = Console()


# ------------------------------------------------------------------------------
# 1. Custom Exceptions
# ------------------------------------------------------------------------------
class BenchmarkError(Exception):
    """Base exception for errors encountered during benchmark execution."""


class ModelInitializationError(BenchmarkError):
    """Raised when the attention module fails to initialize."""


class EvaluationExecutionError(BenchmarkError):
    """Raised when an inference pass encounters a runtime computation error."""


# ------------------------------------------------------------------------------
# 2. Configuration & Metric Dataclasses
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable configuration defining model dimensions and workload scales."""

    hidden_dim: int = 512
    num_heads: int = 8
    prefix_length: int = 1000
    query_length: int = 32
    num_requests: int = 4
    seed: int = 42

    @property
    def head_dim(self) -> int:
        """Computes per-head dimension."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})")
        return self.hidden_dim // self.num_heads


@dataclass
class RequestMetric:
    """
    Encapsulates latency and computational metrics (including Time-To-First-Token / TTFT)
    for an individual query.
    """

    query_id: int
    mode: str
    ttft_ms: float
    tokens_evaluated: int
    cache_status: str


# ------------------------------------------------------------------------------
# 3. Architectural Module (Gemma-style Multi-Head Attention with Key-Value / KV Caching)
# ------------------------------------------------------------------------------
class LightweightAttention(nn.Module):
    """
    Implements a single-layer multi-head scaled dot-product self-attention module
    with explicit Key-Value (KV) cache management to simulate Prefill vs. Decode
    computational dynamics.
    """

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Executes forward attention calculation.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, hidden_dim].
            past_kv: Optional tuple containing previously cached (Key, Value) tensors.

        Returns:
            Tuple containing projected attention outputs and updated (Key, Value) cache.
        """
        try:
            batch_size, seq_len, _ = x.shape
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            if past_kv is not None:
                cached_k, cached_v = past_kv
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
            return self.out_proj(out), (k, v)
        except Exception as exc:
            logger.error("Runtime exception during attention matrix computation: %s", exc)
            raise EvaluationExecutionError("Attention forward pass encountered a tensor failure.") from exc


# ------------------------------------------------------------------------------
# 4. Benchmark Harness
# ------------------------------------------------------------------------------
class BenchmarkHarness:
    """Manages synthetic workload generation and comparative metric recording."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.device = torch.device("cpu")
        try:
            torch.manual_seed(self.config.seed)
            self.model = LightweightAttention(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
            ).to(self.device)
            self.model.eval()
            logger.info("Initialized LightweightAttention module on %s", self.device)
        except Exception as exc:
            raise ModelInitializationError("Failed to initialize benchmark attention model.") from exc

    def generate_workload(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Generates synthetic input tensors representing a shared prefix and distinct user queries."""
        try:
            prefix = torch.randn(1, self.config.prefix_length, self.config.hidden_dim, device=self.device)
            queries = [
                torch.randn(1, self.config.query_length, self.config.hidden_dim, device=self.device)
                for _ in range(self.config.num_requests)
            ]
            return prefix, queries
        except Exception as exc:
            logger.error("Failed to generate synthetic tensor workload: %s", exc)
            raise EvaluationExecutionError("Workload generation failed.") from exc

    def evaluate_naive(self, prefix: torch.Tensor, queries: list[torch.Tensor]) -> list[RequestMetric]:
        """Evaluates requests under Architecture A (naive full-prefill recomputation with 0% cache reuse)."""
        logger.info("Starting Architecture A evaluation (Naive Recomputation)...")
        results: list[RequestMetric] = []
        with torch.no_grad():
            for req_idx, query in enumerate(queries):
                try:
                    full_input = torch.cat([prefix, query], dim=1)
                    start_time = time.perf_counter()
                    _, _ = self.model(full_input)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                    results.append(
                        RequestMetric(
                            query_id=req_idx + 1,
                            mode="Naive",
                            ttft_ms=elapsed_ms,
                            tokens_evaluated=self.config.prefix_length + self.config.query_length,
                            cache_status="0% (Miss)",
                        )
                    )
                except Exception as exc:
                    raise EvaluationExecutionError(f"Error executing naive query #{req_idx + 1}") from exc
        return results

    def evaluate_cached(self, prefix: torch.Tensor, queries: list[torch.Tensor]) -> list[RequestMetric]:
        """Evaluates requests under Architecture B (prefix-cached evaluation using Automatic Prefix Caching / APC)."""
        logger.info("Starting Architecture B evaluation (Automatic Prefix Caching)...")
        results: list[RequestMetric] = []
        prefix_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        with torch.no_grad():
            for req_idx, query in enumerate(queries):
                try:
                    start_time = time.perf_counter()
                    if prefix_cache is None:
                        # Warm up prefix cache on initial request
                        _, prefix_cache = self.model(prefix)
                        _, _ = self.model(query, past_kv=prefix_cache)
                        tokens_evaluated = self.config.prefix_length + self.config.query_length
                        status = "0% (Warmup)"
                    else:
                        # Subsequent requests reuse cached Key-Value blocks
                        _, _ = self.model(query, past_kv=prefix_cache)
                        tokens_evaluated = self.config.query_length
                        status = "100% (Hit)"

                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    results.append(
                        RequestMetric(
                            query_id=req_idx + 1,
                            mode="Cached",
                            ttft_ms=elapsed_ms,
                            tokens_evaluated=tokens_evaluated,
                            cache_status=status,
                        )
                    )
                except Exception as exc:
                    raise EvaluationExecutionError(f"Error executing cached query #{req_idx + 1}") from exc
        return results


# ------------------------------------------------------------------------------
# 5. Presentation Layer
# ------------------------------------------------------------------------------
def display_results(
    config: BenchmarkConfig,
    naive_metrics: list[RequestMetric],
    cached_metrics: list[RequestMetric],
) -> None:
    """Renders formatted tables and performance summaries to the terminal."""
    console.print(Panel.fit(
        "[bold cyan]DEMO 1: SCALABILITY BENCHMARK[/bold cyan]\n"
        "[dim]Architecture A (Naive Recomputation) vs. Architecture B (Automatic Prefix Caching)[/dim]\n"
        "[yellow]Execution Mode: Central Processing Unit (CPU) Native Evaluation "
        "(Arithmetic Intensity & Key-Value Cache Reuse)[/yellow]",
        border_style="cyan",
    ))

    console.print("\n[bold white]Workload Parameters:[/bold white]")
    console.print(f"  • Shared System Prompt Prefix: [green]{config.prefix_length} tokens[/green]")
    console.print(f"  • Unique Query Length:         [green]{config.query_length} tokens[/green]")
    console.print(f"  • Cumulative Request Count:    [green]{config.num_requests} requests[/green]\n")

    table = Table(title="[bold yellow]Empirical Comparison: Time-To-First-Token (TTFT)[/bold yellow]")
    table.add_column("Query ID", justify="center", style="bold white")
    table.add_column("Arch A: Naive TTFT", justify="right", style="red")
    table.add_column("Arch B: Cached TTFT", justify="right", style="green")
    table.add_column("Speedup Factor", justify="center", style="bold cyan")
    table.add_column("Tokens Saved", justify="right", style="magenta")

    total_naive_time = sum(m.ttft_ms for m in naive_metrics)
    total_cached_time = sum(m.ttft_ms for m in cached_metrics)
    total_tokens_saved = 0

    for i in range(len(naive_metrics)):
        a = naive_metrics[i]
        b = cached_metrics[i]
        speedup = f"{a.ttft_ms / b.ttft_ms:.1f}x"
        tokens_saved = a.tokens_evaluated - b.tokens_evaluated
        total_tokens_saved += tokens_saved
        table.add_row(
            f"Query #{a.query_id}",
            f"{a.ttft_ms:.2f} ms",
            f"{b.ttft_ms:.2f} ms ({b.cache_status})",
            speedup,
            f"{tokens_saved} tokens",
        )

    console.print(table)
    overall_speedup = total_naive_time / total_cached_time if total_cached_time > 0 else 0.0

    console.print(Panel(
        f"[bold white]Systems Scaling Analysis:[/bold white]\n"
        f"  • Cumulative Prefill Latency (Naive):  [red]{total_naive_time:.2f} ms[/red]\n"
        f"  • Cumulative Prefill Latency (Cached): [green]{total_cached_time:.2f} ms[/green]\n"
        f"  • [bold]Net Prefill Speedup Ratio:             [bold cyan]{overall_speedup:.1f}x[/bold cyan][/bold]\n"
        f"  • Redundant Computations Avoided:      [magenta]{total_tokens_saved:,} tokens[/magenta]\n\n"
        f"[dim]Technical Principle: By retaining and reusing precomputed Key-Value (KV) projection\n"
        f"tensors for static prompt prefixes, Automatic Prefix Caching avoids repeated O(N²)\n"
        f"self-attention prefill evaluations on subsequent requests.[/dim]",
        title="[bold green]Architectural Summary[/bold green]",
        border_style="green",
    ))


def main() -> None:
    """Entrypoint executing the prefix caching evaluation harness."""
    try:
        config = BenchmarkConfig()
        harness = BenchmarkHarness(config)
        prefix, queries = harness.generate_workload()

        naive_metrics = harness.evaluate_naive(prefix, queries)
        cached_metrics = harness.evaluate_cached(prefix, queries)

        display_results(config, naive_metrics, cached_metrics)
    except BenchmarkError as exc:
        logger.error("Benchmark terminated with a controlled error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Unexpected runtime failure: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
