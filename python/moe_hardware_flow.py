"""
End-to-end MoE training → FPGA-style hardware evaluation pipeline.

Flow:
  1. Train Global-KV MoE attention (L_task + α·L_aux)
  2. Benchmark trained forward pass across sequence lengths
  3. Roofline + FPGA latency model (fpga_like.yaml peaks)
  4. Per-config hardware stats (sparse FLOPs, DRAM, routing)
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from python.attention.moe_attention import (
    MoeParams,
    make_moe_params,
    make_training_batch,
    moe_attention_from_params,
    moe_dram_traffic,
    train_moe,
)
from python.benchmark import BenchmarkResult
from python.hardware import FPGA_CLOCK_FREQ_GHZ
from python.hardware_benchmark import benchmark_variant_timeloop
from python.roofline import analyze_roofline, print_roofline_summary


HW_PEAK_GFLOPS = 200.0
HW_PEAK_BW_GB_S = 25.6


@dataclass
class MoeEvalPoint:
    seq_len: int
    latency_ms: float
    sparse_attn_flops: int
    throughput_gflops: float
    dram_bytes: int
    arithmetic_intensity: float
    fpga_latency_ms: float
    fpga_utilization_pct: float
    region: str
    moe_stats: Dict


@dataclass
class MoeHardwareFlowResult:
    params: MoeParams
    n_experts: int
    top_k: int
    batch_size: int = 2
    n_heads: int = 8
    head_dim: int = 64
    training_history: List[Dict[str, float]] = field(default_factory=list)
    eval_points: List[MoeEvalPoint] = field(default_factory=list)
    final_train_stats: Dict = field(default_factory=dict)


def _time_moe_forward(
    params: MoeParams,
    x: jnp.ndarray,
    top_k: int,
    n_warmup: int = 3,
    n_runs: int = 8,
) -> float:
    def _run():
        out, _ = moe_attention_from_params(x, params, top_k=top_k, is_causal=True)
        return out

    for _ in range(n_warmup):
        jax.block_until_ready(_run())

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(_run())
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def _fpga_latency_ms(
    sparse_flops: int,
    dram_bytes: int,
    hw_peak_gflops: float = HW_PEAK_GFLOPS,
    hw_peak_bw_gb_s: float = HW_PEAK_BW_GB_S,
) -> float:
    """Roofline-style FPGA latency (compute vs memory bound)."""
    compute_ms = (sparse_flops / (hw_peak_gflops * 1e9)) * 1e3
    mem_ms = (dram_bytes / (hw_peak_bw_gb_s * 1e9)) * 1e3
    return max(compute_ms, mem_ms)


def evaluate_trained_moe(
    params: MoeParams,
    n_experts: int,
    top_k: int,
    batch_size: int = 2,
    n_heads: int = 8,
    head_dim: int = 64,
    seq_lens: List[int] = (128, 256, 512, 1024),
    key: jax.random.PRNGKey = jax.random.PRNGKey(1),
) -> List[MoeEvalPoint]:
    """Benchmark trained MoE on FPGA-style metrics at multiple sequence lengths."""
    ridge = HW_PEAK_GFLOPS / HW_PEAK_BW_GB_S
    points: List[MoeEvalPoint] = []

    for i, N in enumerate(seq_lens):
        key, kx = jax.random.split(key)
        x, _ = make_training_batch(kx, batch_size, n_heads, N, head_dim)

        _, stats = moe_attention_from_params(x, params, top_k=top_k, is_causal=True)

        try:
            tl = benchmark_variant_timeloop(
                "moe", batch_size, n_heads, N, head_dim, timeout=300
            )
            latency_ms = tl.latency_ms
            dram_bytes = tl.total_dram_bytes
            sparse_flops = int(
                tl.throughput_gflops * 1e9 * (latency_ms / 1e3)
            )
            throughput = tl.throughput_gflops
            ai = tl.arithmetic_intensity
            fpga_ms = latency_ms
            fpga_util = getattr(tl, "utilization_pct", 0.0)
            region = "memory-bound" if ai < ridge else "compute-bound"
            hw_source = "timeloop"
        except Exception as exc:
            print(f"    [warn] Timeloop moe failed ({exc}); using analytical model")
            latency_ms = _time_moe_forward(params, x, top_k)
            traffic = moe_dram_traffic(
                batch_size, n_heads, N, head_dim, n_experts,
                stats["tokens_per_expert"],
            )
            dram_bytes = sum(traffic.values())
            sparse_flops = stats["sparse_attn_flops"]
            throughput = (sparse_flops / 1e9) / (latency_ms / 1e3)
            ai = sparse_flops / max(dram_bytes, 1)
            fpga_ms = _fpga_latency_ms(sparse_flops, dram_bytes)
            fpga_util = 0.0
            region = "memory-bound" if ai < ridge else "compute-bound"
            hw_source = "analytical"

        points.append(MoeEvalPoint(
            seq_len=N,
            latency_ms=latency_ms,
            sparse_attn_flops=sparse_flops,
            throughput_gflops=throughput,
            dram_bytes=dram_bytes,
            arithmetic_intensity=ai,
            fpga_latency_ms=fpga_ms,
            fpga_utilization_pct=fpga_util,
            region=region,
            moe_stats={**stats, "hw_source": hw_source},
        ))
        print(
            f"  N={N:>4}  FPGA={latency_ms:6.1f}ms ({hw_source})  "
            f"util={fpga_util:5.1f}%  "
            f"GFLOP/s={throughput:6.1f}  "
            f"AI={ai:5.1f}  {region}  "
            f"active_experts={stats['active_experts']}/{n_experts}  "
            f"L_aux={stats['load_balancing_loss']:.3f}"
        )

    return points


def run_moe_hardware_flow(
    batch_size: int = 2,
    n_heads: int = 8,
    head_dim: int = 64,
    train_seq_len: int = 128,
    n_experts: int = 8,
    top_k: int = 2,
    n_train_steps: int = 100,
    lr: float = 1e-3,
    aux_weight: float = 0.01,
    seq_lens: Optional[List[int]] = None,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> MoeHardwareFlowResult:
    """
    Full pipeline: train MoE end-to-end, then evaluate on FPGA-style hardware.
    """
    if seq_lens is None:
        seq_lens = [128, 256, 512, 1024]

    print("\n" + "=" * 70)
    print("  MoE End-to-End Training → FPGA Hardware Evaluation")
    print("=" * 70)

    print(f"\n[1/3] Training Global-KV MoE  "
          f"({n_experts} experts, top-{top_k}, {n_train_steps} steps) …")
    print(f"      L = L_task (denoising MSE) + {aux_weight} · L_aux (load balance)")

    params = make_moe_params(n_experts, n_heads, head_dim, key)
    params, history = train_moe(
        params, key,
        batch_size=batch_size,
        n_heads=n_heads,
        seq_len=train_seq_len,
        head_dim=head_dim,
        top_k=top_k,
        n_steps=n_train_steps,
        lr=lr,
        aux_weight=aux_weight,
        log_every=max(1, n_train_steps // 10),
    )

    key, k_eval = jax.random.split(key)
    x_final, _ = make_training_batch(
        k_eval, batch_size, n_heads, train_seq_len, head_dim
    )
    _, final_stats = moe_attention_from_params(
        x_final, params, top_k=top_k, is_causal=True, aux_weight=aux_weight
    )

    print(f"\n[2/3] Post-training routing @ N={train_seq_len}:")
    print(f"      task-ready loss history: {history[-1]['total_loss']:.5f}")
    print(f"      L_aux={final_stats['load_balancing_loss']:.4f}  "
          f"gate_entropy={final_stats['gate_entropy']:.2f}  "
          f"compute_saved={final_stats['compute_saved_ratio']*100:.0f}%")
    print(f"      tokens/expert: {final_stats['tokens_per_expert']}")

    print(f"\n[3/3] FPGA-style hardware eval (peak={HW_PEAK_GFLOPS} GFLOP/s, "
          f"BW={HW_PEAK_BW_GB_S} GB/s, clock={FPGA_CLOCK_FREQ_GHZ*1e3:.0f} MHz) …")
    eval_points = evaluate_trained_moe(
        params, n_experts, top_k,
        batch_size=batch_size,
        n_heads=n_heads,
        head_dim=head_dim,
        seq_lens=seq_lens,
        key=key,
    )

    return MoeHardwareFlowResult(
        params=params,
        n_experts=n_experts,
        top_k=top_k,
        batch_size=batch_size,
        n_heads=n_heads,
        head_dim=head_dim,
        training_history=history,
        eval_points=eval_points,
        final_train_stats=final_stats,
    )


def moe_eval_to_benchmark_results(
    eval_points: List[MoeEvalPoint],
    batch_size: int,
    n_heads: int,
    head_dim: int,
    n_experts: int,
) -> List[BenchmarkResult]:
    """Convert MoE eval points to BenchmarkResult for roofline plots."""
    results = []
    for pt in eval_points:
        t = moe_dram_traffic(
            batch_size, n_heads, pt.seq_len, head_dim,
            n_experts,
            pt.moe_stats["tokens_per_expert"],
        )
        results.append(BenchmarkResult(
            name="moe_trained",
            seq_len=pt.seq_len,
            head_dim=head_dim,
            n_heads=n_heads,
            batch_size=batch_size,
            dtype="float32",
            latency_ms=pt.latency_ms,
            throughput_gflops=pt.throughput_gflops,
            dram_qkv_bytes=t["qkv"],
            dram_scores_bytes=t["scores"],
            dram_output_bytes=t["output"],
        ))
    return results


def print_moe_hardware_summary(result: MoeHardwareFlowResult) -> None:
    """Print consolidated training + hardware summary."""
    print("\n" + "=" * 70)
    print("  MoE Hardware Flow — Summary")
    print("=" * 70)

    h0 = result.training_history[0]
    h1 = result.training_history[-1]
    print(f"\n── Training ({len(result.training_history)} steps) ──")
    print(f"  L_total:  {h0['total_loss']:.5f} → {h1['total_loss']:.5f}")
    print(f"  L_task:   {h0['task_loss']:.5f} → {h1['task_loss']:.5f}")
    print(f"  L_aux:    {h0['aux_loss']:.4f} → {h1['aux_loss']:.4f}")

    print(f"\n── FPGA Eval (trained weights) ──")
    header = (
        f"{'N':>5} {'CPU(ms)':>8} {'FPGA(ms)':>9} "
        f"{'GFLOP/s':>8} {'DRAM(KB)':>9} {'AI':>6} {'Region':<14} {'Util%':>6}"
    )
    print(header)
    print("-" * len(header))
    for pt in result.eval_points:
        print(
            f"{pt.seq_len:>5} {pt.latency_ms:>8.1f} {pt.fpga_latency_ms:>9.1f} "
            f"{pt.throughput_gflops:>8.1f} {pt.dram_bytes/1e3:>9.1f} "
            f"{pt.arithmetic_intensity:>6.1f} {pt.region:<14} "
            f"{pt.fpga_utilization_pct:>5.0f}%"
        )

    bench = moe_eval_to_benchmark_results(
        result.eval_points,
        batch_size=result.batch_size,
        n_heads=result.n_heads,
        head_dim=result.head_dim,
        n_experts=result.n_experts,
    )
    roofline = analyze_roofline(bench, HW_PEAK_GFLOPS, HW_PEAK_BW_GB_S)
    print("\n── Roofline (trained MoE) ──")
    print_roofline_summary(roofline)

    fs = result.final_train_stats
    print(f"\n── Learned Routing @ N={result.eval_points[0].seq_len if result.eval_points else '?'} ──")
    print(f"  Experts active: {fs['active_experts']}/{fs['n_experts']}")
    print(f"  Routed queries: {fs['routed_queries']}")
    print(f"  Compute saved:  {fs['compute_saved_ratio']*100:.0f}%")
    print(f"  SRAM reads est: {fs['estimated_sram_reads']/1e3:.1f} KB")
