"""FlashAccel experiment runner.

Usage: python run_experiment.py [--skip-timeloop] [--skip-moe]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
jax.config.update("jax_platform_name", "cpu")

from python.benchmark import run_benchmark_suite, print_table
from python.compiler  import analyze_attention_hlo, print_analysis
from python.hardware  import run_all, print_summary
from python.roofline  import analyze_roofline, print_roofline_summary
from python.plots     import make_all_plots
from python.conditional_compute import run_conditional_compute, print_conditional_compute_summary

B, H, D   = 2, 8, 64
SEQ_LENS  = [128, 256, 512, 1024]
PLOTS_DIR = Path(__file__).parent / "workspace/outputs/plots"
HLO_DIR   = Path(__file__).parent / "workspace/outputs/hlo"


def main():
    parser = argparse.ArgumentParser(description="FlashAccel experiment runner")
    parser.add_argument("--skip-timeloop", action="store_true", help="Skip Docker/Timeloop step")
    parser.add_argument("--skip-moe", action="store_true", help="Skip MoE/conditional compute step")
    args = parser.parse_args()

    print("=" * 70)
    print("  FlashAccel — Transformer Attention on FPGA-Style Arch")
    print("  Conditional Compute Explorer")
    print("=" * 70)

    print("\n[1/6] Running JAX benchmarks …")
    bench_results = run_benchmark_suite(
        B=B, H=H, D=D, seq_lens=SEQ_LENS, n_warmup=3, n_runs=8,
    )
    print("\nBenchmark summary:")
    print_table(bench_results)

    print("\n[1b/6] Roofline analysis …")
    roofline = analyze_roofline(bench_results)
    print_roofline_summary(roofline)

    print("\n[2/6] Analysing StableHLO IR …")
    HLO_DIR.mkdir(parents=True, exist_ok=True)
    hlo_analysis = analyze_attention_hlo(B=B, H=H, N=512, D=D, dump_dir=HLO_DIR)
    print_analysis(hlo_analysis)

    hw_stats = None
    if not args.skip_timeloop:
        print("\n[3/6] Running Timeloop/Accelergy via Docker …")
        print("      (requires Docker to be running with the arm64 image pulled)")
        try:
            hw_stats = run_all(timeout=300)
            print("\nTimeloop summary:")
            print_summary(hw_stats)
        except Exception as e:
            print(f"[WARNING] Timeloop step failed: {e}")
            print("          Re-run after fixing Docker / YAML issues.")
    else:
        print("\n[3/6] Timeloop skipped (--skip-timeloop)")

    cc_results = None
    if not args.skip_moe:
        print("\n[4/6] Running Conditional Compute Explorer …")
        try:
            cc_results = run_conditional_compute(B=B, H=H, N=512, D=D)
            print_conditional_compute_summary(cc_results)
        except Exception as e:
            print(f"[WARNING] Conditional compute step failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[4/6] Conditional compute skipped (--skip-moe)")

    print(f"\n[5/6] Generating plots → {PLOTS_DIR}")
    make_all_plots(
        bench_results, hw_stats=hw_stats, cc_results=cc_results,
        save_dir=str(PLOTS_DIR),
    )

    print("\n[6/6] Done.")
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
