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
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
jax.config.update("jax_platform_name", "cpu")

from python.benchmark import make_qkv_fixtures
from python.compiler  import analyze_attention_hlo, print_analysis
from python.hardware_benchmark import (
    run_timeloop_benchmark_suite,
    print_timeloop_table,
)
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
        help="Run end-to-end MoE training + FPGA hardware evaluation",
    )
    parser.add_argument(
        "--moe-steps", type=int, default=100,
        help="Training steps for --moe-train (default: 100)",
    )
    args = parser.parse_args()

    if args.moe_train:
        from python.moe_hardware_flow import run_moe_hardware_flow, print_moe_hardware_summary
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

    print("\n[2/6] Analysing StableHLO IR …")
    HLO_DIR.mkdir(parents=True, exist_ok=True)
    hlo_fixture = QKV[HLO_N]
    hlo_analysis = analyze_attention_hlo(
        B=B, H=H, N=HLO_N, D=D,
        sample=(hlo_fixture.Q, hlo_fixture.K, hlo_fixture.V),
        dump_dir=HLO_DIR,
    )
    print_analysis(hlo_analysis)

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


if __name__ == "__main__":
    main()
