"""FlashAccel experiment runner.

Usage:
  python run_experiment.py                    # Timeloop hardware sim (default)
  python run_experiment.py --host-benchmark   # legacy host JAX timing
  python run_experiment.py --moe-train        # MoE train + analytical HW eval
  python run_experiment.py --skip-moe         # skip conditional compute step

Default flow: JAX verifies attention algorithms; Timeloop (fpga_like.yaml)
reports latency, DRAM traffic, utilization, and energy for each variant.
"""
import sys
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
HW_PEAK_GFLOPS = 200.0
HW_PEAK_BW_GB_S = 25.6
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_platform_name", "cpu")

from python.attention.moe_attention import (
    MoeParams,
    make_moe_params,
    make_training_batch,
    moe_attention_from_params,
    moe_dram_traffic,
    train_moe,
)
from python.benchmark import BenchmarkResult, make_qkv_fixtures
from python.benchmark import (
    run_timeloop_benchmark_suite,
    print_timeloop_table,
    benchmark_variant_timeloop,
)
# from python.compiler  import analyze_attention_hlo, print_analysis
from python.hardware  import FPGA_CLOCK_FREQ_GHZ
from python.roofline  import analyze_roofline, print_roofline_summary
from python.plots     import make_all_plots
from python.conditional_compute import run_conditional_compute, print_conditional_compute_summary

B, H, D   = 2, 8, 64
SEQ_LENS  = [128, 256, 512, 1024]
HLO_N     = 512
PLOTS_DIR = Path(__file__).parent / "workspace/outputs/plots"
HLO_DIR   = Path(__file__).parent / "workspace/outputs/hlo"
QKV       = make_qkv_fixtures(B, H, D, SEQ_LENS, key=0)


def main():
    parser = argparse.ArgumentParser(description="FlashAccel experiment runner")
    parser.add_argument(
        "--host-benchmark", action="store_true",
        help="Use host JAX timing instead of Timeloop hardware simulation (legacy)",
    )
    parser.add_argument("--skip-timeloop", action="store_true", help="(deprecated) same as --host-benchmark")
    parser.add_argument("--skip-moe", action="store_true", help="Skip MoE/conditional compute step")
    parser.add_argument(
        "--moe-train", action="store_true",
        help="Run end-to-end MoE training + Custom Asics evaluation",
    )
    parser.add_argument(
        "--moe-steps", type=int, default=100,
        help="Training steps for --moe-train (default: 100)",
    )
    args = parser.parse_args()

    if args.moe_train:
        moe_result = run_moe_hardware_flow(
            batch_size=B, n_heads=H, head_dim=D,
            train_seq_len=128, n_experts=8, top_k=2,
            n_train_steps=args.moe_steps,
            seq_lens=SEQ_LENS,
        )
        print_moe_hardware_summary(moe_result)
        return

    print("=" * 70)
    print("  FlashAccel — Transformer Attention on FPGA-Style Arch")
    print("  Conditional Compute Explorer")
    print("=" * 70)

    use_host = args.host_benchmark or args.skip_timeloop

    if use_host:
        from python.benchmark import run_benchmark_suite, print_table
        print("\n[1/6] Running JAX host benchmarks (legacy mode) …")
        bench_results = run_benchmark_suite(
            B=B, H=H, D=D, seq_lens=SEQ_LENS, fixtures=QKV,
            n_warmup=3, n_runs=8,
        )
        print("\nBenchmark summary:")
        print_table(bench_results)
    else:
        print("\n[1/6] Simulating JAX attention variants on fpga_like hardware (Timeloop) …")
        print("      JAX verifies algorithms; Timeloop reports latency/DRAM/utilization")
        print("      (requires Docker with timeloopaccelergy image)")
        try:
            bench_results = run_timeloop_benchmark_suite(
                B=B, H=H, D=D, seq_lens=SEQ_LENS, fixtures=QKV,
                timeout=300, verify_jax=True,
            )
            print("\nHardware benchmark summary (Timeloop):")
            print_timeloop_table(bench_results)
        except Exception as e:
            print(f"[WARNING] Timeloop benchmark failed: {e}")
            print("          Falling back to host JAX benchmarks …")
            from python.benchmark import run_benchmark_suite, print_table
            bench_results = run_benchmark_suite(
                B=B, H=H, D=D, seq_lens=SEQ_LENS, fixtures=QKV,
                n_warmup=3, n_runs=8,
            )
            print_table(bench_results)

    print("\n[1b/6] Roofline analysis …")
    roofline = analyze_roofline(bench_results)
    print_roofline_summary(roofline)

    # print("\n[2/6] Analysing StableHLO IR …")
    # HLO_DIR.mkdir(parents=True, exist_ok=True)
    # hlo_fixture = QKV[HLO_N]
    # hlo_analysis = analyze_attention_hlo(
    #     B=B, H=H, N=HLO_N, D=D,
    #     sample=(hlo_fixture.Q, hlo_fixture.K, hlo_fixture.V),
    #     dump_dir=HLO_DIR,
    # )
    # print_analysis(hlo_analysis)

    cc_results = None
    if not args.skip_moe:
        print("\n[3/6] Running Conditional Compute Explorer …")
        try:
            cc_results = run_conditional_compute(
                B=B, H=H, N=HLO_N, D=D, fixture=QKV[HLO_N],
            )
            print_conditional_compute_summary(cc_results)
        except Exception as e:
            print(f"[WARNING] Conditional compute step failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[3/6] Conditional compute skipped (--skip-moe)")

    print(f"\n[4/6] Generating plots → {PLOTS_DIR}")
    make_all_plots(
        bench_results, hw_stats=None, cc_results=cc_results,
        save_dir=str(PLOTS_DIR),
    )

    print("\n[5/6] Done.")
    print("      Open the HTML files in workspace/outputs/plots/ to view results.")
    if cc_results:
        moe_8 = next((m for m in cc_results.get("moe_sweep", []) if m["n_experts"] == 8), None)
        if moe_8:
            print(f"\n  ┌─ Key Insight ─────────────────────────────────┐")
            print(f"  │  MoE (8 experts, top-2):                      │")
            print(f"  │    Compute Saved: {moe_8['compute_saved_ratio']*100:.0f}%"
                  f"{'':>24}│")
            print(f"  │    Expert Utilization: {moe_8['expert_utilization']*100:.0f}%"
                  f"{'':>20}│")
            print(f"  └────────────────────────────────────────────────┘")





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
    train_seq_len: int = 128
    training_history: List[Dict[str, float]] = field(default_factory=list)
    eval_points: List[MoeEvalPoint] = field(default_factory=list)
    final_train_stats: Dict = field(default_factory=dict)


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

    for N in seq_lens:
        key, kx = jax.random.split(key)
        x, _ = make_training_batch(kx, batch_size, n_heads, N, head_dim)
        _, stats = moe_attention_from_params(x, params, top_k=top_k, is_causal=True)

        try:
            tl = benchmark_variant_timeloop("moe", batch_size, n_heads, N, head_dim, timeout=300)
            latency_ms = tl.latency_ms
            dram_bytes = tl.total_dram_bytes
            sparse_flops = int(tl.throughput_gflops * 1e9 * (latency_ms / 1e3))
            throughput = tl.throughput_gflops
            ai = tl.arithmetic_intensity
            fpga_util = getattr(tl, "utilization_pct", 0.0)
        except Exception as exc:
            print(f"    [warn] Timeloop moe failed ({exc}); using analytical model")
            # Time forward pass on host
            fn = lambda: jax.block_until_ready(
                moe_attention_from_params(x, params, top_k=top_k, is_causal=True)[0]
            )
            for _ in range(3):
                fn()
            times = []
            for _ in range(8):
                t0 = time.perf_counter()
                fn()
                times.append((time.perf_counter() - t0) * 1e3)
            latency_ms = float(np.median(times))
            # Analytical DRAM model
            traffic = moe_dram_traffic(
                batch_size, n_heads, N, head_dim, n_experts, stats["tokens_per_expert"],
            )
            dram_bytes = sum(traffic.values())
            sparse_flops = stats["sparse_attn_flops"]
            throughput = (sparse_flops / 1e9) / (latency_ms / 1e3)
            ai = sparse_flops / max(dram_bytes, 1)
            # Roofline latency estimate
            latency_ms = max(
                (sparse_flops / (HW_PEAK_GFLOPS * 1e9)) * 1e3,
                (dram_bytes / (HW_PEAK_BW_GB_S * 1e9)) * 1e3,
            )
            fpga_util = 0.0

        region = "memory-bound" if ai < ridge else "compute-bound"
        points.append(MoeEvalPoint(
            seq_len=N, latency_ms=latency_ms, sparse_attn_flops=sparse_flops,
            throughput_gflops=throughput, dram_bytes=dram_bytes,
            arithmetic_intensity=ai, fpga_latency_ms=latency_ms,
            fpga_utilization_pct=fpga_util, region=region, moe_stats=stats,
        ))
        print(
            f"  N={N:>4}  FPGA={latency_ms:6.1f}ms  "
            f"util={fpga_util:5.1f}%  GFLOP/s={throughput:6.1f}  "
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
    """Full pipeline: train MoE end-to-end, then evaluate on FPGA-style hardware."""
    if seq_lens is None:
        seq_lens = [128, 256, 512, 1024]

    print("\n" + "=" * 70)
    print("  MoE End-to-End Training -> Custom Asics Evaluation")
    print("=" * 70)
    print(f"\n[1/3] Training Windowed Local-KV MoE  "
          f"({n_experts} experts, top-{top_k}, {n_train_steps} steps) ...")

    params = make_moe_params(n_experts, n_heads, head_dim, key)
    params, history = train_moe(
        params, key,
        batch_size=batch_size, n_heads=n_heads, seq_len=train_seq_len,
        head_dim=head_dim, top_k=top_k, n_steps=n_train_steps,
        lr=lr, aux_weight=aux_weight,
        log_every=max(1, n_train_steps // 10),
    )

    key, k_eval = jax.random.split(key)
    x_final, _ = make_training_batch(k_eval, batch_size, n_heads, train_seq_len, head_dim)
    _, final_stats = moe_attention_from_params(
        x_final, params, top_k=top_k, is_causal=True, aux_weight=aux_weight
    )

    print(f"\n[2/3] Post-training routing @ N={train_seq_len}:")
    print(f"      L_aux={final_stats['load_balancing_loss']:.4f}  "
          f"gate_entropy={final_stats['gate_entropy']:.2f}  "
            f"local_window_saved={final_stats['compute_saved_ratio']*100:.0f}%")

    print(f"\n[3/3] FPGA-style hardware eval (peak={HW_PEAK_GFLOPS} GFLOP/s, "
          f"BW={HW_PEAK_BW_GB_S} GB/s) ...")
    eval_points = evaluate_trained_moe(
        params, n_experts, top_k,
        batch_size=batch_size, n_heads=n_heads, head_dim=head_dim,
        seq_lens=seq_lens, key=key,
    )

    return MoeHardwareFlowResult(
        params=params, n_experts=n_experts, top_k=top_k,
        batch_size=batch_size, n_heads=n_heads, head_dim=head_dim,
        train_seq_len=train_seq_len,
        training_history=history, eval_points=eval_points,
        final_train_stats=final_stats,
    )


def print_moe_hardware_summary(result: MoeHardwareFlowResult) -> None:
    """Print consolidated training + hardware summary."""
    print("\n" + "=" * 70)
    print("  MoE Hardware Flow -- Summary")
    print("=" * 70)

    h0, h1 = result.training_history[0], result.training_history[-1]
    print(f"\n-- Training ({len(result.training_history)} steps) --")
    print(f"  L_total:  {h0['total_loss']:.5f} -> {h1['total_loss']:.5f}")
    print(f"  L_task:   {h0['task_loss']:.5f} -> {h1['task_loss']:.5f}")
    print(f"  L_aux:    {h0['aux_loss']:.4f} -> {h1['aux_loss']:.4f}")

    print(f"\n-- FPGA Eval (trained weights) --")
    header = (
        f"{'N':>5} {'FPGA(ms)':>9} {'GFLOP/s':>8} "
        f"{'DRAM(KB)':>9} {'AI':>6} {'Region':<14} {'Util%':>6} {'Saved%':>7}"
    )
    print(header)
    print("-" * len(header))
    for pt in result.eval_points:
        saved_pct = pt.moe_stats.get("compute_saved_ratio", 0.0) * 100
        print(
            f"{pt.seq_len:>5} {pt.fpga_latency_ms:>9.1f} "
            f"{pt.throughput_gflops:>8.1f} {pt.dram_bytes/1e3:>9.1f} "
            f"{pt.arithmetic_intensity:>6.1f} {pt.region:<14} "
            f"{pt.fpga_utilization_pct:>5.0f}% {saved_pct:>6.0f}%"
        )

    print("\n-- Roofline (trained MoE) --")
    ridge = HW_PEAK_GFLOPS / HW_PEAK_BW_GB_S
    print(f"\nRoofline Analysis (ridge point = {ridge:.1f} F/B)")
    print(f"Hardware: {HW_PEAK_GFLOPS} GFLOP/s peak, {HW_PEAK_BW_GB_S} GB/s DRAM BW")
    header = f"{'Variant':<12} {'N':>6} {'AI(F/B)':>9} {'Region':<15} {'% Peak':>8}"
    print(header)
    print("─" * len(header))
    for pt in result.eval_points:
        attainable = min(HW_PEAK_BW_GB_S * pt.arithmetic_intensity, HW_PEAK_GFLOPS)
        pct_peak = (pt.throughput_gflops / attainable * 100) if attainable > 0 else 0.0
        print(
            f"{'moe_trained':<12} {pt.seq_len:>6} {pt.arithmetic_intensity:>9.1f} "
            f"{pt.region:<15} {pct_peak:>7.1f}%"
        )

    fs = result.final_train_stats
    print(f"\n-- Learned Routing --")
    print(f"  Experts active: {fs['active_experts']}/{fs['n_experts']}")
    print(
        f"  Local-window compute saved @ train N={result.train_seq_len}: "
        f"{fs['compute_saved_ratio']*100:.0f}%"
    )


if __name__ == "__main__":
    main()
