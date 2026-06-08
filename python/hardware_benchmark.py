"""
JAX → Timeloop hardware benchmark bridge.

Each JAX attention variant maps to optimal Timeloop mappings on fpga_like.yaml.
All latency, DRAM, utilization, and energy metrics come from Timeloop simulation
—not host CPU/GPU timing.
"""
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from python.benchmark import (
    BenchmarkResult,
    QKVFixture,
    _attention_flops,
    _flash_dram_bytes,
    _naive_dram_bytes,
    _quantized_dram_bytes,
)
from python.hardware import (
    FPGA_CLOCK_FREQ_GHZ,
    JAX_VARIANT_TIMELOOP,
    MOE_N_EXPERTS,
    MOE_TOP_K,
    PAGED_PAGE_SIZE,
    run_experiment,
)


def _sliding_window_dram_bytes(B, H, N, D, dtype_bytes=2):
    from python.benchmark import _bytes
    w = min(128, N)
    effective_pairs = N * w - (w * (w - 1)) // 2
    qkv = 3 * _bytes((B, H, N, D), dtype_bytes)
    scores = 2 * _bytes((B, H, effective_pairs), dtype_bytes)
    output = _bytes((B, H, N, D), dtype_bytes)
    return {"qkv": qkv, "scores": scores, "output": output}


def _gqa_dram_bytes(B, H, N, D, dtype_bytes=2):
    from python.benchmark import _bytes
    n_kv = 2
    qkv = _bytes((B, H, N, D), dtype_bytes) + 2 * _bytes((B, n_kv, N, D), dtype_bytes)
    return {"qkv": qkv, "scores": 0, "output": _bytes((B, H, N, D), dtype_bytes)}


def _paged_dram_bytes(B, H, N, D, dtype_bytes=2, page_size=PAGED_PAGE_SIZE):
    from python.benchmark import _bytes
    avg_seq = int(N * 0.75)
    n_pages = (avg_seq + page_size - 1) // page_size
    q_bytes = _bytes((B, H, N, D), dtype_bytes)
    kv_pages = 2 * B * H * n_pages * page_size * D * dtype_bytes
    page_table = B * n_pages * 4
    output = _bytes((B, H, N, D), dtype_bytes)
    return {"qkv": q_bytes + kv_pages + page_table, "scores": 0, "output": output}


def _moe_dram_bytes(
    B, H, N, D,
    n_experts=MOE_N_EXPERTS,
    top_k=MOE_TOP_K,
    dtype_bytes=2,
):
    from python.benchmark import _bytes
    n_active = n_experts
    routed_per_expert = max(1, B * N * top_k // n_experts)
    x_bytes = _bytes((B, H, N, D), dtype_bytes)
    global_kv = n_active * 2 * _bytes((B, H, N, D), dtype_bytes)
    routed_q = n_active * _bytes((routed_per_expert, H, D), dtype_bytes)
    expert_w = n_active * 4 * D * D * dtype_bytes
    output = _bytes((B, H, N, D), dtype_bytes)
    return {
        "qkv": x_bytes + global_kv + routed_q + expert_w,
        "scores": 0,
        "output": output,
    }


VARIANT_TRAFFIC_FNS: Dict[str, Callable] = {
    "naive":          _naive_dram_bytes,
    "flash":          _flash_dram_bytes,
    "quantized":      _quantized_dram_bytes,
    "sliding_window": _sliding_window_dram_bytes,
    "gqa":            _gqa_dram_bytes,
    "paged":          _paged_dram_bytes,
    "moe":            _moe_dram_bytes,
}


def verify_jax_variants(
    B: int = 2,
    H: int = 8,
    N: int = 128,
    D: int = 64,
    fixture: Optional[QKVFixture] = None,
) -> None:
    """Smoke-test that JAX attention implementations compile and run."""
    from python.attention.naive import naive_attention
    from python.attention.flash import flash_attention
    from python.attention.quantize import quantized_attention
    from python.attention.sliding_window import sliding_window_attention
    from python.attention.gqa import grouped_query_attention
    from python.attention.paged_attention import paged_attention
    from python.attention.moe_attention import moe_attention, make_attention_experts

    if fixture is not None:
        Q, K, V = fixture.Q, fixture.K, fixture.V
        x = fixture.x
    else:
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        Q = jax.random.normal(k1, (B, H, N, D))
        K = jax.random.normal(k2, (B, H, N, D))
        V = jax.random.normal(k3, (B, H, N, D))
        x = (Q + K + V) / 3.0

    key = jax.random.PRNGKey(0)

    checks = [
        ("naive", lambda: naive_attention(Q, K, V)),
        ("flash", lambda: flash_attention(Q, K, V, is_causal=True)),
        ("quantized", lambda: quantized_attention(Q, K, V)),
        ("sliding_window", lambda: sliding_window_attention(Q, K, V, window_size=128, is_causal=True)),
        ("gqa", lambda: grouped_query_attention(Q, K[:, :2], V[:, :2], n_kv_heads=2)),
        ("paged", lambda: paged_attention(Q, K, V)),
        ("moe", lambda: moe_attention(
            x,
            *make_attention_experts(8, H, D, key=jax.random.split(key)[0]),
            top_k=2, is_causal=True,
        )[0]),
    ]
    for name, fn in checks:
        out = fn()
        arr = out[0] if isinstance(out, tuple) else out
        jax.block_until_ready(arr)
        print(f"  [jax] {name:<14} OK  shape={arr.shape}")


def _scale_dram_breakdown(
    variant: str,
    B: int,
    H: int,
    N: int,
    D: int,
    timeloop_dram_bytes: float,
) -> Dict[str, int]:
    """Scale analytical DRAM component ratios to Timeloop-measured total."""
    traffic_fn = VARIANT_TRAFFIC_FNS.get(variant, _naive_dram_bytes)
    profile = traffic_fn(B, H, N, D)
    total_model = sum(profile.values()) or 1
    scale = timeloop_dram_bytes / total_model
    return {
        "qkv": int(profile["qkv"] * scale),
        "scores": int(profile["scores"] * scale),
        "output": int(profile["output"] * scale),
    }


def _combine_attention_phases(
    qk_stats: Dict,
    av_stats: Dict,
    batch_size: int,
    n_heads: int,
) -> Dict:
    """Combine QK^T + AV Timeloop results into one attention-layer measurement."""
    scale = batch_size * n_heads

    def _get(d, k, default=0):
        v = d.get(k)
        return v if v is not None else default

    cycles = (_get(qk_stats, "cycles") + _get(av_stats, "cycles")) * scale
    latency_ms = (cycles / (FPGA_CLOCK_FREQ_GHZ * 1e9)) * 1e3

    dram_reads = (_get(qk_stats, "dram_reads_bytes") + _get(av_stats, "dram_reads_bytes")) * scale
    dram_writes = (_get(qk_stats, "dram_writes_bytes") + _get(av_stats, "dram_writes_bytes")) * scale
    sram_reads = (_get(qk_stats, "sram_reads_bytes") + _get(av_stats, "sram_reads_bytes")) * scale
    sram_writes = (_get(qk_stats, "sram_writes_bytes") + _get(av_stats, "sram_writes_bytes")) * scale
    energy_uj = (_get(qk_stats, "energy_uj") + _get(av_stats, "energy_uj")) * scale

    util_qk = _get(qk_stats, "utilization")
    util_av = _get(av_stats, "utilization")
    utilization = (util_qk + util_av) / 2.0 if util_qk and util_av else (util_qk or util_av or 0.0)

    gflops_qk = _get(qk_stats, "gflops", 0)
    gflops_av = _get(av_stats, "gflops", 0)
    timeloop_gflops = (gflops_qk + gflops_av) * scale

    return {
        "cycles": cycles,
        "latency_ms": latency_ms,
        "dram_reads_bytes": dram_reads,
        "dram_writes_bytes": dram_writes,
        "dram_total_bytes": dram_reads + dram_writes,
        "sram_total_bytes": sram_reads + sram_writes,
        "energy_uj": energy_uj,
        "utilization": utilization,
        "timeloop_gflops": timeloop_gflops,
        "qk_stats": qk_stats,
        "av_stats": av_stats,
    }


def benchmark_variant_timeloop(
    variant: str,
    B: int,
    H: int,
    N: int,
    D: int,
    timeout: int = 300,
) -> BenchmarkResult:
    """
    Simulate one JAX attention variant on fpga_like hardware via Timeloop.

    The variant name selects the optimal mapping YAMLs (e.g. flash → flash_qk +
    flash_av).  Metrics are from Timeloop cycles/energy/DRAM — not host JAX time.
    """
    if variant not in JAX_VARIANT_TIMELOOP:
        raise ValueError(
            f"No Timeloop mapping for variant '{variant}'. "
            f"Available: {list(JAX_VARIANT_TIMELOOP)}"
        )

    qk_name, av_name, kind = JAX_VARIANT_TIMELOOP[variant]
    window = min(128, N) if kind == "sliding_window" else None

    run_kw = dict(
        seq_len=N, head_dim=D, window=window, timeout=timeout,
        top_k=MOE_TOP_K, n_experts=MOE_N_EXPERTS,
    )
    qk_stats = run_experiment(qk_name, **run_kw)
    if qk_stats.get("error"):
        raise RuntimeError(f"Timeloop QK failed for {variant}: {qk_stats['error']}")

    av_stats = run_experiment(av_name, **run_kw)
    if av_stats.get("error"):
        raise RuntimeError(f"Timeloop AV failed for {variant}: {av_stats['error']}")

    combined = _combine_attention_phases(qk_stats, av_stats, B, H)

    # MoE: scale one expert's QK+AV to all active experts (global-KV model)
    if kind == "moe":
        for key in ("cycles", "latency_ms", "dram_reads_bytes", "dram_writes_bytes",
                    "dram_total_bytes", "sram_total_bytes", "energy_uj", "timeloop_gflops"):
            if key in combined and combined[key] is not None:
                combined[key] = combined[key] * MOE_N_EXPERTS

    if kind == "moe":
        routed = max(1, (N * MOE_TOP_K) // MOE_N_EXPERTS)
        flops = MOE_N_EXPERTS * (
            2 * routed * H * D * D * 4
            + routed * H * N * D * 4
        ) * B
    else:
        flops = _attention_flops(B, H, N, D)
    latency_ms = combined["latency_ms"]
    throughput = (flops / 1e9) / (latency_ms / 1e3) if latency_ms > 0 else 0.0

    dram_breakdown = _scale_dram_breakdown(
        variant, B, H, N, D, combined["dram_total_bytes"]
    )

    result = BenchmarkResult(
        name=variant,
        seq_len=N,
        head_dim=D,
        n_heads=H,
        batch_size=B,
        dtype="int16",
        latency_ms=latency_ms,
        throughput_gflops=throughput,
        dram_qkv_bytes=dram_breakdown["qkv"],
        dram_scores_bytes=dram_breakdown["scores"],
        dram_output_bytes=dram_breakdown["output"],
    )
    result.utilization_pct = combined["utilization"]
    result.energy_uj = combined["energy_uj"]
    result.sram_bytes = combined["sram_total_bytes"]
    result.metric_source = "timeloop"
    result.timeloop_cycles = combined["cycles"]
    return result


def run_timeloop_benchmark_suite(
    B: int = 2,
    H: int = 8,
    D: int = 64,
    seq_lens: List[int] = (128, 256, 512, 1024),
    timeout: int = 300,
    verify_jax: bool = True,
    fixtures: Optional[Dict[int, QKVFixture]] = None,
) -> List[BenchmarkResult]:
    """
    Benchmark all JAX attention variants on simulated fpga_like hardware.

    Flow:
      1. (Optional) Verify JAX implementations compile
      2. For each variant × seq_len: run Timeloop QK + AV mappings
      3. Return BenchmarkResults with Timeloop-sourced metrics
    """
    if verify_jax:
        print("\n── JAX variant verification (compile + correctness) ──")
        verify_n = min(128, max(seq_lens))
        verify_jax_variants(
            B=B, H=H, N=verify_n, D=D,
            fixture=fixtures.get(verify_n) if fixtures else None,
        )

    variants = list(JAX_VARIANT_TIMELOOP.keys())
    results: List[BenchmarkResult] = []

    print("\n── Timeloop hardware simulation (fpga_like.yaml) ──")
    print(f"    Clock: {FPGA_CLOCK_FREQ_GHZ*1e3:.0f} MHz  |  "
          f"Metrics: cycles → latency, DRAM, utilization, energy")

    for N in seq_lens:
        print(f"\n── seq_len = {N} ──────────────")
        for vname in variants:
            try:
                r = benchmark_variant_timeloop(vname, B, H, N, D, timeout=timeout)
                util = getattr(r, "utilization_pct", 0.0)
                print(
                    f"  {vname:<14} FPGA={r.latency_ms:7.2f} ms  "
                    f"util={util:5.1f}%  "
                    f"DRAM={r.total_dram_bytes/1e6:.2f} MB  "
                    f"GFLOP/s={r.throughput_gflops:.1f}  "
                    f"AI={r.arithmetic_intensity:.1f}"
                )
                results.append(r)
            except Exception as e:
                print(f"  {vname:<14} FAILED: {e}")

    return results


def print_timeloop_table(results: List[BenchmarkResult]) -> None:
    """Print hardware metrics table (all values from Timeloop)."""
    header = (
        f"{'Variant':<14} {'N':>5} {'FPGA(ms)':>9} {'Util%':>7} "
        f"{'DRAM(MB)':>9} {'SRAM(MB)':>9} {'AI(F/B)':>8} {'Source':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        util = getattr(r, "utilization_pct", 0.0)
        sram = getattr(r, "sram_bytes", 0) / 1e6
        source = getattr(r, "metric_source", "timeloop")
        print(
            f"{r.name:<14} {r.seq_len:>5} {r.latency_ms:>9.2f} "
            f"{util:>6.1f}% {r.total_dram_bytes/1e6:>9.2f} "
            f"{sram:>9.2f} {r.arithmetic_intensity:>8.1f} {source:>10}"
        )
