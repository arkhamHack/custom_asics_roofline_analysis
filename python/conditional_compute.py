"""Conditional compute orchestrator: runs all attention variants and collects metrics."""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple

from python.sparsity import (
    compute_all_savings, print_savings_table, ComputeStats,
    dense_stats, moe_stats,
)


def run_conditional_compute(
    B: int = 2,
    H: int = 8,
    N: int = 512,
    D: int = 64,
    dtype=jnp.float32,
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
    from python.attention.naive import naive_attention
    from python.attention.flash import flash_attention
    from python.attention.sliding_window import sliding_window_attention
    from python.attention.paged_attention import paged_attention
    from python.attention.gqa import grouped_query_attention
    from python.attention.moe_attention import moe_attention, make_expert_weights

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    Q = jax.random.normal(k1, (B, H, N, D), dtype=dtype)
    K = jax.random.normal(k2, (B, H, N, D), dtype=dtype)
    V = jax.random.normal(k3, (B, H, N, D), dtype=dtype)

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
        K_gqa = jax.random.normal(k2, (B, n_kv, N, D), dtype=dtype)
        V_gqa = jax.random.normal(k3, (B, n_kv, N, D), dtype=dtype)
        out, stats = grouped_query_attention(Q, K_gqa, V_gqa, n_kv_heads=n_kv)
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
        W1, W2, W_gate = make_expert_weights(n_exp, D, expand=4, key=key)
        out_moe, moe_stats_result = moe_attention(
            Q, K, V, W1, W2, W_gate, top_k=2
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
              f"gate_entropy={m['gate_entropy']:.2f}")

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
