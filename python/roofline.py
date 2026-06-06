"""Roofline analysis: classifies attention variants as memory-bound or compute-bound."""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RooflinePoint:
    name: str
    seq_len: int
    arithmetic_intensity: float  # FLOP / DRAM byte
    throughput_gflops: float
    region: str                  # "memory-bound" or "compute-bound"
    percent_of_peak: float       # % of attainable peak in its region


@dataclass
class RooflineAnalysis:
    hw_peak_gflops: float
    hw_peak_bw_gb_s: float
    ridge_point: float           # AI where memory roof meets compute roof
    points: List[RooflinePoint]


def analyze_roofline(
    benchmark_results,
    hw_peak_gflops: float = 200.0,
    hw_peak_bw_gb_s: float = 25.6,
) -> RooflineAnalysis:
    """
    Classify each benchmark result as memory-bound or compute-bound.

    Ridge point = peak_gflops / peak_bw_gb_s (FLOP/byte).
    If AI < ridge → memory-bound (limited by bandwidth).
    If AI >= ridge → compute-bound (limited by ALUs).
    """
    ridge = hw_peak_gflops / hw_peak_bw_gb_s

    points = []
    for r in benchmark_results:
        ai = r.arithmetic_intensity
        attainable = min(hw_peak_bw_gb_s * ai, hw_peak_gflops)
        region = "memory-bound" if ai < ridge else "compute-bound"
        pct = (r.throughput_gflops / attainable * 100) if attainable > 0 else 0.0

        points.append(RooflinePoint(
            name=r.name,
            seq_len=r.seq_len,
            arithmetic_intensity=ai,
            throughput_gflops=r.throughput_gflops,
            region=region,
            percent_of_peak=pct,
        ))

    return RooflineAnalysis(
        hw_peak_gflops=hw_peak_gflops,
        hw_peak_bw_gb_s=hw_peak_bw_gb_s,
        ridge_point=ridge,
        points=points,
    )


def print_roofline_summary(analysis: RooflineAnalysis) -> None:
    """Print a formatted roofline classification table."""
    print(f"\nRoofline Analysis (ridge point = {analysis.ridge_point:.1f} F/B)")
    print(f"Hardware: {analysis.hw_peak_gflops} GFLOP/s peak, "
          f"{analysis.hw_peak_bw_gb_s} GB/s DRAM BW")
    header = f"{'Variant':<12} {'N':>6} {'AI(F/B)':>9} {'Region':<15} {'% Peak':>8}"
    print(header)
    print("─" * len(header))
    for p in analysis.points:
        print(
            f"{p.name:<12} {p.seq_len:>6} {p.arithmetic_intensity:>9.1f} "
            f"{p.region:<15} {p.percent_of_peak:>7.1f}%"
        )
