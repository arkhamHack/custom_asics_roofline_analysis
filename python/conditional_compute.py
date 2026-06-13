"""Compute/memory savings calculator and conditional compute orchestrator.
"""
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from python.benchmark import QKVFixture


# ══════════════════════════════════════════════════════════════════════════════
# Savings calculator (formerly sparsity.py)
# ══════════════════════════════════════════════════════════════════════════════

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
    """Full dense attention: O(N^2) scores, all heads active."""
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
    """Sliding window: O(N*w) scores instead of O(N^2)."""
    dense_flops = 4 * B * H * N * N * D
    w = min(window_size, N)
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
    """GQA: KV heads shared -> reduced KV memory, same compute."""
    dense_flops = 4 * B * H * N * N * D
    dense_mem = (3 * B * H * N * D + B * H * N * N + B * H * N * D) * dtype_bytes
    gqa_mem = (
        B * H * N * D +         # Q
        2 * B * n_kv_heads * N * D +  # K + V (reduced)
        B * H * N * N +         # scores (still full)
        B * H * N * D           # output
    ) * dtype_bytes

    return ComputeStats(
        name=f"gqa_{H}q_{n_kv_heads}kv", seq_len=N, n_heads=H, head_dim=D, batch_size=B,
        total_flops=dense_flops, active_flops=dense_flops,
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
    ffn_weight_mem = n_experts * D * ffn_expand * D * 2 * dtype_bytes
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
        avg_seq_len = N * 3 // 4

    dense_flops = 4 * B * H * N * N * D
    contiguous_kv_mem = 2 * B * H * N * D * dtype_bytes
    pages_needed = (avg_seq_len + page_size - 1) // page_size
    paged_kv_mem = 2 * B * H * pages_needed * page_size * D * dtype_bytes

    dense_mem = (3 * B * H * N * D + B * H * N * N + B * H * N * D) * dtype_bytes
    saved_kv = contiguous_kv_mem - paged_kv_mem
    active_mem = dense_mem - saved_kv

    return ComputeStats(
        name="paged", seq_len=N, n_heads=H, head_dim=D, batch_size=B,
        total_flops=dense_flops, active_flops=dense_flops,
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
    print("-" * len(header))
    for s in stats_list:
        print(
            f"{s.name:<22} {s.seq_len:>5} "
            f"{s.compute_saved_ratio*100:>13.1f}% "
            f"{s.memory_saved_ratio*100:>12.1f}% "
            f"{s.active_flops/1e9:>12.2f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Conditional compute orchestrator
# ══════════════════════════════════════════════════════════════════════════════


def run_conditional_compute(
    B: int = 2,
    H: int = 8,
    N: int = 512,
    D: int = 64,
    fixture: Optional[QKVFixture] = None,
) -> Dict:
    """
    Run all conditional compute variants and collect metrics.

    Returns dict with:
        - 'savings': list of ComputeStats (theoretical)
        - 'runtime': dict of actual JAX runtime results
        - 'moe_sweep': MoE results across expert counts
        - 'gqa_sweep': GQA results across group ratios
        - 'sliding_window_sweep': results across window sizes
    """
    from python.attention.flash import flash_attention
    from python.attention.sliding_window import sliding_window_attention
    from python.attention.paged_attention import paged_attention
    from python.attention.gqa import grouped_query_attention
    from python.attention.moe_attention import moe_attention, make_attention_experts

    key = jax.random.PRNGKey(0)
    if fixture is not None:
        Q, K, V, x = fixture.Q, fixture.K, fixture.V, fixture.x
    else:
        k1, k2, k3 = jax.random.split(key, 3)
        Q = jax.random.normal(k1, (B, H, N, D))
        K = jax.random.normal(k2, (B, H, N, D))
        V = jax.random.normal(k3, (B, H, N, D))
        x = (Q + K + V) / 3.0

    results = {}

    results["savings"] = compute_all_savings(B, H, N, D)

    sw_results = []
    for w in [64, 128, 256]:
        if w <= N:
            out, stats = sliding_window_attention(Q, K, V, window_size=w, is_causal=True)
            jax.block_until_ready(out)
            sw_results.append({"window_size": w, **stats})
    results["sliding_window_sweep"] = sw_results


    gqa_results = []
    for n_kv in [1, 2, 4, H]:
        out, stats = grouped_query_attention(
            Q, K[:, :n_kv], V[:, :n_kv], n_kv_heads=n_kv
        )
        jax.block_until_ready(out)
        gqa_results.append(stats)
    results["gqa_sweep"] = gqa_results

    # ── Runtime: Paged attention ──────────────────────────────────────────
    seq_lens = jnp.array([N * 3 // 4] * B)  # 75% utilization
    out_paged, paged_stats = paged_attention(Q, K, V, page_size=16, seq_lens=seq_lens)
    jax.block_until_ready(out_paged)
    results["paged"] = paged_stats


    moe_results = []
    for n_exp in [4, 8, 16]:
        Wq, Wk, Wv, Wo, W_gate = make_attention_experts(
            n_experts=n_exp, n_heads=H, head_dim=D, key=key
        )
        out_moe, moe_stats_result = moe_attention(
            x, Wq, Wk, Wv, Wo, W_gate, top_k=2, is_causal=True
        )
        jax.block_until_ready(out_moe)
        moe_results.append(moe_stats_result)
    results["moe_sweep"] = moe_results

    return results


def print_conditional_compute_summary(results: Dict) -> None:
    """Print a formatted summary of conditional compute analysis."""
    print("\n" + "=" * 70)
    print("  Conditional Compute Explorer — What Actually Fires?")
    print("=" * 70)

    # Theoretical savings
    print("\n── Theoretical Savings (vs Dense) ──")
    print_savings_table(results["savings"])

    # Sliding window
    print("\n── Sliding Window Attention ──")
    for sw in results["sliding_window_sweep"]:
        print(f"  window={sw['window_size']:>4}  "
              f"sparsity={sw['sparsity_ratio']*100:.1f}%  "
              f"FLOPs ratio={sw['flops_vs_dense']*100:.1f}%")

    # GQA
    print("\n── Grouped Query Attention (KV Cache Savings) ──")
    for g in results["gqa_sweep"]:
        print(f"  {g['n_heads']}Q/{g['n_kv_heads']}KV (group={g['group_size']})  "
              f"KV memory saved={g['kv_memory_saved_ratio']*100:.0f}%  "
              f"BW reduction={g['kv_bandwidth_reduction']:.0f}×")

    # Paged
    print("\n── Paged Attention (vLLM-style) ──")
    p = results["paged"]
    print(f"  page_size={p['page_size']}  "
          f"pages_alloc={p['pages_allocated']}  "
          f"frag={p['internal_fragmentation']*100:.1f}%  "
          f"mem_saved={p['memory_saved_vs_contiguous']*100:.1f}%")

    # MoE
    print("\n── Mixture of Experts (Token-Level Routing) ──")
    for m in results["moe_sweep"]:
        print(f"  {m['n_experts']} experts, top-{m['top_k']}  "
              f"compute_saved={m['compute_saved_ratio']*100:.0f}%  "
              f"utilization={m['expert_utilization']*100:.0f}%  "
              f"gate_entropy={m['gate_entropy']:.2f}  "
              f"L_aux={m['load_balancing_loss']:.4f}")

    # Visual summary
    print("\n── What Users See ──")
    print("\n  Dense:    ", "█" * 20)
    if results["sliding_window_sweep"]:
        sw = results["sliding_window_sweep"][0]
        active = int(20 * sw["flops_vs_dense"])
        print(f"  Window:   ", "█" * active + "░" * (20 - active))
    if results["moe_sweep"]:
        m = results["moe_sweep"][1]  # 8 experts
        active = int(20 * m["compute_fraction"])
        print(f"  MoE(8e):  ", "█" * active + "░" * (20 - active))

    # Expert routing display
    if results["moe_sweep"]:
        m = results["moe_sweep"][1]
        n_exp = m["n_experts"]
        top_k = m["top_k"]
        print(f"\n  Expert Routing ({n_exp} experts, top-{top_k}):")
        # Simulate which experts fire (from stats)
        for i in range(n_exp):
            status = "✅" if i < top_k else "❌"
            print(f"    Expert {i+1} {status}")
        print(f"\n  Compute Saved: {m['compute_saved_ratio']*100:.0f}%")
        print(f"  Memory Saved:  (FFN weights) {(1-top_k/n_exp)*100:.0f}%")
