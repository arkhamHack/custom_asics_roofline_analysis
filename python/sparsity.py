"""Compute and memory savings calculator for attention variants."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ComputeStats:
    """Unified stats for one variant at one config."""
    name: str
    seq_len: int
    n_heads: int
    head_dim: int
    batch_size: int

    total_flops: int
    active_flops: int

    total_memory: int
    active_memory: int

    # Derived
    @property
    def compute_saved_ratio(self) -> float:
        return 1.0 - (self.active_flops / self.total_flops) if self.total_flops > 0 else 0.0

    @property
    def memory_saved_ratio(self) -> float:
        return 1.0 - (self.active_memory / self.total_memory) if self.total_memory > 0 else 0.0

    @property
    def flops_saved(self) -> int:
        return self.total_flops - self.active_flops

    @property
    def memory_saved(self) -> int:
        return self.total_memory - self.active_memory


def dense_stats(B: int, H: int, N: int, D: int, dtype_bytes: int = 2) -> ComputeStats:
    """Full dense attention: O(N²) scores, all heads active."""
    flops = 4 * B * H * N * N * D  # QK^T + AV, each 2*B*H*N*N*D
    mem = (3 * B * H * N * D + B * H * N * N + B * H * N * D) * dtype_bytes  # Q+K+V+scores+output
    return ComputeStats(
        name="dense", seq_len=N, n_heads=H, head_dim=D, batch_size=B,
        total_flops=flops, active_flops=flops,
        total_memory=mem, active_memory=mem,
    )


def sliding_window_stats(
    B: int, H: int, N: int, D: int,
    window_size: int = 128,
    dtype_bytes: int = 2,
) -> ComputeStats:
    """Sliding window: O(N·w) scores instead of O(N²)."""
    dense_flops = 4 * B * H * N * N * D
    # Effective pairs: each row attends to at most window_size keys
    w = min(window_size, N)
    # For causal: average window is w/2 for first w tokens, w for rest
    effective_pairs = N * w - (w * (w - 1)) // 2  # exact for causal window
    ratio = effective_pairs / (N * N)

    active_flops = int(dense_flops * ratio)
    dense_mem = (3 * B * H * N * D + B * H * N * N + B * H * N * D) * dtype_bytes
    active_mem = (3 * B * H * N * D + B * H * effective_pairs + B * H * N * D) * dtype_bytes

    return ComputeStats(
        name="sliding_window", seq_len=N, n_heads=H, head_dim=D, batch_size=B,
        total_flops=dense_flops, active_flops=active_flops,
        total_memory=dense_mem, active_memory=active_mem,
    )


def sparse_block_stats(
    B: int, H: int, N: int, D: int,
    block_size: int = 64,
    n_selected: int = 2,
    dtype_bytes: int = 2,
) -> ComputeStats:
    """Block-sparse attention: each query block attends to n_selected KV blocks."""
    dense_flops = 4 * B * H * N * N * D
    num_q_blocks = N // block_size
    # Each query block attends to n_selected * block_size keys
    effective_pairs = num_q_blocks * block_size * n_selected * block_size
    ratio = effective_pairs / (N * N)

    active_flops = int(dense_flops * ratio)
    dense_mem = (3 * B * H * N * D + B * H * N * N + B * H * N * D) * dtype_bytes
    active_mem = (3 * B * H * N * D + B * H * effective_pairs + B * H * N * D) * dtype_bytes

    return ComputeStats(
        name="block_sparse", seq_len=N, n_heads=H, head_dim=D, batch_size=B,
        total_flops=dense_flops, active_flops=active_flops,
        total_memory=dense_mem, active_memory=active_mem,
    )


def gqa_stats(
    B: int, H: int, N: int, D: int,
    n_kv_heads: int = 2,
    dtype_bytes: int = 2,
) -> ComputeStats:
    """GQA: KV heads shared → reduced KV memory, same compute."""
    dense_flops = 4 * B * H * N * N * D  # compute is same (KV is broadcast)
    # Memory: Q still has H heads, but KV only has n_kv_heads
    dense_mem = (3 * B * H * N * D + B * H * N * N + B * H * N * D) * dtype_bytes
    gqa_mem = (
        B * H * N * D +         # Q
        2 * B * n_kv_heads * N * D +  # K + V (reduced)
        B * H * N * N +         # scores (still full)
        B * H * N * D           # output
    ) * dtype_bytes

    return ComputeStats(
        name=f"gqa_{H}q_{n_kv_heads}kv", seq_len=N, n_heads=H, head_dim=D, batch_size=B,
        total_flops=dense_flops, active_flops=dense_flops,  # compute unchanged
        total_memory=dense_mem, active_memory=gqa_mem,
    )


def moe_stats(
    B: int, H: int, N: int, D: int,
    n_experts: int = 8,
    top_k: int = 2,
    ffn_expand: int = 4,
    dtype_bytes: int = 2,
) -> ComputeStats:
    """MoE: attention always runs, but FFN only uses top_k/n_experts fraction."""
    attn_flops = 4 * B * H * N * N * D
    dense_ffn_flops = 2 * B * N * D * (ffn_expand * D) * 2  # up + down
    total_flops = attn_flops + dense_ffn_flops
    active_ffn_flops = int(dense_ffn_flops * (top_k / n_experts))
    active_flops = attn_flops + active_ffn_flops

    dense_mem = (3 * B * H * N * D + B * H * N * N + B * H * N * D) * dtype_bytes
    # FFN weights: all experts stored, but only top_k accessed per token
    ffn_weight_mem = n_experts * D * ffn_expand * D * 2 * dtype_bytes  # W1 + W2
    active_ffn_weight_mem = top_k * D * ffn_expand * D * 2 * dtype_bytes
    total_mem = dense_mem + ffn_weight_mem
    active_mem = dense_mem + active_ffn_weight_mem

    return ComputeStats(
        name=f"moe_{n_experts}e_top{top_k}", seq_len=N, n_heads=H, head_dim=D, batch_size=B,
        total_flops=total_flops, active_flops=active_flops,
        total_memory=total_mem, active_memory=active_mem,
    )


def paged_stats(
    B: int, H: int, N: int, D: int,
    page_size: int = 16,
    avg_seq_len: Optional[int] = None,
    dtype_bytes: int = 2,
) -> ComputeStats:
    """Paged attention: same compute, reduced memory waste from fragmentation."""
    if avg_seq_len is None:
        avg_seq_len = N * 3 // 4  # assume 75% utilization on average

    dense_flops = 4 * B * H * N * N * D
    # Contiguous: allocates max_seq_len for every sequence
    contiguous_kv_mem = 2 * B * H * N * D * dtype_bytes  # K + V
    # Paged: allocates only pages needed for actual length
    pages_needed = (avg_seq_len + page_size - 1) // page_size
    paged_kv_mem = 2 * B * H * pages_needed * page_size * D * dtype_bytes

    dense_mem = (3 * B * H * N * D + B * H * N * N + B * H * N * D) * dtype_bytes
    # Savings come from KV cache allocation, not compute
    saved_kv = contiguous_kv_mem - paged_kv_mem
    active_mem = dense_mem - saved_kv

    return ComputeStats(
        name="paged", seq_len=N, n_heads=H, head_dim=D, batch_size=B,
        total_flops=dense_flops, active_flops=dense_flops,  # same compute
        total_memory=dense_mem, active_memory=max(active_mem, 0),
    )


def compute_all_savings(
    B: int = 2, H: int = 8, N: int = 512, D: int = 64,
) -> List[ComputeStats]:
    """Compute savings for all variants at a given configuration."""
    return [
        dense_stats(B, H, N, D),
        sliding_window_stats(B, H, N, D, window_size=128),
        sparse_block_stats(B, H, N, D, block_size=64, n_selected=2),
        gqa_stats(B, H, N, D, n_kv_heads=2),
        gqa_stats(B, H, N, D, n_kv_heads=4),
        paged_stats(B, H, N, D, page_size=16),
        moe_stats(B, H, N, D, n_experts=4, top_k=2),
        moe_stats(B, H, N, D, n_experts=8, top_k=2),
        moe_stats(B, H, N, D, n_experts=16, top_k=2),
    ]


def print_savings_table(stats_list: List[ComputeStats]) -> None:
    """Print a formatted savings summary."""
    header = (
        f"{'Variant':<22} {'N':>5} {'Compute Saved':>14} "
        f"{'Memory Saved':>13} {'Active GFLOP':>12}"
    )
    print(header)
    print("─" * len(header))
    for s in stats_list:
        print(
            f"{s.name:<22} {s.seq_len:>5} "
            f"{s.compute_saved_ratio*100:>13.1f}% "
            f"{s.memory_saved_ratio*100:>12.1f}% "
            f"{s.active_flops/1e9:>12.2f}"
        )
