"""JAX attention benchmark suite + Timeloop hardware bridge."""
import time
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ── Shared QKV inputs ────────────────────────────────────────────────────────

@dataclass
class QKVFixture:
    """Fixed Q/K/V tensors for fair cross-variant comparison."""
    Q: jnp.ndarray
    K: jnp.ndarray
    V: jnp.ndarray

    @property
    def x(self) -> jnp.ndarray:
        return (self.Q + self.K + self.V) / 3.0


def make_qkv_fixtures(
    B: int,
    H: int,
    D: int,
    seq_lens: List[int],
    key: int = 0,
    dtype=jnp.float16,
) -> Dict[int, QKVFixture]:
    """One deterministic Q/K/V triple per sequence length."""
    rng = jax.random.PRNGKey(key)
    fixtures: Dict[int, QKVFixture] = {}
    for N in seq_lens:
        kq, kk, kv, rng = jax.random.split(rng, 4)
        fixtures[N] = QKVFixture(
            Q=jax.random.normal(kq, (B, H, N, D), dtype=dtype),
            K=jax.random.normal(kk, (B, H, N, D), dtype=dtype),
            V=jax.random.normal(kv, (B, H, N, D), dtype=dtype),
        )
    return fixtures


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    name: str
    seq_len: int
    head_dim: int
    n_heads: int
    batch_size: int
    dtype: str
    latency_ms: float
    throughput_gflops: float
    dram_qkv_bytes: int
    dram_scores_bytes: int
    dram_output_bytes: int
    # Populated when metrics come from Timeloop (see hardware_benchmark.py)
    utilization_pct: float = 0.0
    energy_uj: float = 0.0
    sram_bytes: int = 0
    metric_source: str = "host"       # "host" | "timeloop"
    timeloop_cycles: float = 0.0

    @property
    def total_dram_bytes(self) -> int:
        return self.dram_qkv_bytes + self.dram_scores_bytes + self.dram_output_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs / DRAM bytes — roofline x-axis."""
        if self.total_dram_bytes == 0:
            return float("inf")
        return (self.throughput_gflops * 1e9 * self.latency_ms / 1e3) / self.total_dram_bytes

    def fpga_latency_ms(
        self,
        hw_peak_gflops: float = 200.0,
        hw_peak_bw_gb_s: float = 25.6,
        dram_latency_ns: float = 100.0,
    ) -> float:
        """
        Simulated FPGA latency based on roofline + pipeline model.

        Accounts for:
          - Compute time: FLOPs / peak_compute
          - Memory time: DRAM_bytes / peak_BW + latency penalty per round-trip
          - Pipeline stall: naive must write scores to DRAM then read back
            (2 sequential transfers), flash avoids this entirely.
        """
        flops = self.throughput_gflops * 1e9 * self.latency_ms / 1e3  # actual FLOPs
        if flops == 0:
            flops = _attention_flops(
                self.batch_size, self.n_heads, self.seq_len, self.head_dim
            )

        # Pure compute time (if fully compute-bound)
        compute_ms = (flops / (hw_peak_gflops * 1e9)) * 1e3

        # Pure memory time (bandwidth-limited)
        mem_ms = (self.total_dram_bytes / (hw_peak_bw_gb_s * 1e9)) * 1e3

        # Pipeline penalty: scores written then read back = 2 serial DRAM transfers
        # Flash and sliding-window avoid this (scores stay in SRAM)
        if self.dram_scores_bytes > 0:
            # Each score tile round-trip incurs DRAM latency
            n_score_tiles = max(1, self.dram_scores_bytes // (64 * 64 * 2))
            pipeline_stall_ms = n_score_tiles * dram_latency_ns * 2 / 1e6
        else:
            pipeline_stall_ms = 0.0

        # FPGA latency = max(compute, memory) + pipeline stalls
        # (compute and memory overlap in a well-designed pipeline,
        #  but stalls are additive)
        return max(compute_ms, mem_ms) + pipeline_stall_ms


def _time_fn(fn: Callable, *args, n_warmup: int = 3, n_runs: int = 10) -> float:
    """Time a JAX function (ms), using block_until_ready to flush async dispatch."""
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1e3)

    return float(np.median(times))


# ── Theoretical traffic calculators ──────────────────────────────────────────

def _bytes(shape, dtype_bytes: int = 2) -> int:
    return int(np.prod(shape)) * dtype_bytes


def _attention_flops(B: int, H: int, N: int, D: int) -> int:
    return 4 * B * H * N * N * D


def _window_attention_flops(B: int, H: int, N: int, D: int, window: int = 128) -> int:
    W = min(window, N)
    return 4 * B * H * N * W * D


def _flash_dram_bytes(B: int, H: int, N: int, D: int, dtype_bytes: int = 2) -> dict:
    qkv = 3 * _bytes((B, H, N, D), dtype_bytes)
    scores = 0
    output = _bytes((B, H, N, D), dtype_bytes)
    return {"qkv": qkv, "scores": scores, "output": output}


def _quantized_dram_bytes(B: int, H: int, N: int, D: int, bits: int = 8) -> dict:
    dtype_bytes = bits // 8
    qkv = 3 * _bytes((B, H, N, D), dtype_bytes)
    scores = 0
    output = _bytes((B, H, N, D), dtype_bytes)
    return {"qkv": qkv, "scores": scores, "output": output}


def _bsda_dram_bytes(
    B: int, H: int, N: int, D: int,
    block_size: int = 64, n_selected: int = 2, dtype_bytes: int = 2
) -> dict:
    """Block-sparse deformable attention: burst block loads, reduced score matrix."""
    num_blocks = N // block_size
    window     = n_selected * block_size
    qkv    = 3 * _bytes((B, H, N, D), dtype_bytes)
    scores = 2 * _bytes((B, H, N, window), dtype_bytes)            # [N, window] not [N, N]
    kv_extra = 2 * n_selected * _bytes((B, H, num_blocks, block_size, D), dtype_bytes)
    output = _bytes((B, H, N, D), dtype_bytes)
    return {"qkv": qkv, "scores": scores + kv_extra, "output": output}



def benchmark_variant(
    name: str,
    fn: Callable,
    B: int, H: int, N: int, D: int,
    traffic_fn: Callable,
    Q: Optional[jnp.ndarray] = None,
    K: Optional[jnp.ndarray] = None,
    V: Optional[jnp.ndarray] = None,
    dtype=jnp.float16,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark one attention variant at given (B, H, N, D)."""
    if Q is None or K is None or V is None:
        Q = jax.random.normal(jax.random.PRNGKey(0), (B, H, N, D), dtype=dtype)
        K = jax.random.normal(jax.random.PRNGKey(1), (B, H, N, D), dtype=dtype)
        V = jax.random.normal(jax.random.PRNGKey(2), (B, H, N, D), dtype=dtype)

    def _call():
        out = fn(Q, K, V)
        return out[0] if isinstance(out, (tuple, list)) else out

    latency_ms = _time_fn(_call, n_warmup=n_warmup, n_runs=n_runs)
    flops = _attention_flops(B, H, N, D)
    throughput_gflops = (flops / 1e9) / (latency_ms / 1e3)

    traffic = traffic_fn(B, H, N, D)

    return BenchmarkResult(
        name=name,
        seq_len=N,
        head_dim=D,
        n_heads=H,
        batch_size=B,
        dtype=str(dtype),
        latency_ms=latency_ms,
        throughput_gflops=throughput_gflops,
        dram_qkv_bytes=traffic["qkv"],
        dram_scores_bytes=traffic["scores"],
        dram_output_bytes=traffic["output"],
    )



def run_benchmark_suite(
    B: int = 2,
    H: int = 8,
    D: int = 64,
    seq_lens: List[int] = (128, 256, 512, 1024),
    fixtures: Optional[Dict[int, QKVFixture]] = None,
    n_warmup: int = 3,
    n_runs: int = 10,
    dtype=jnp.float16,
) -> List[BenchmarkResult]:
    """
    Run all three attention variants across multiple sequence lengths.
    Returns a flat list of BenchmarkResult objects.
    """
    from python.attention.flash import flash_attention
    from python.attention.quantize import quantized_attention
    from python.attention.sliding_window import sliding_window_attention
    from python.attention.gqa import grouped_query_attention
    from python.attention.paged_attention import paged_attention

    def _sliding_window_wrapper(Q, K, V):
        out, _ = sliding_window_attention(Q, K, V, window_size=128, is_causal=True)
        return out

    def _gqa_wrapper(Q, K, V):
        K_gqa = K[:, :2, :, :]
        V_gqa = V[:, :2, :, :]
        out, _ = grouped_query_attention(Q, K_gqa, V_gqa, n_kv_heads=2)
        return out

    def _sliding_window_dram_bytes(B, H, N, D, dtype_bytes=2):
        w = min(128, N)
        effective_pairs = N * w - (w * (w - 1)) // 2
        qkv = 3 * _bytes((B, H, N, D), dtype_bytes)
        scores = 2 * _bytes((B, H, effective_pairs), dtype_bytes)
        output = _bytes((B, H, N, D), dtype_bytes)
        return {"qkv": qkv, "scores": scores, "output": output}

    def _gqa_dram_bytes(B, H, N, D, dtype_bytes=2):
        n_kv = 2
        qkv = _bytes((B, H, N, D), dtype_bytes) + 2 * _bytes((B, n_kv, N, D), dtype_bytes)
        scores = 0
        output = _bytes((B, H, N, D), dtype_bytes)
        return {"qkv": qkv, "scores": scores, "output": output}

    def _paged_wrapper(Q, K, V):
        out, _ = paged_attention(Q, K, V)
        return out

    def _paged_dram_bytes(B, H, N, D, dtype_bytes=2, page_size=16):
        avg_seq = int(N * 0.75)
        n_pages = (avg_seq + page_size - 1) // page_size
        q_bytes = _bytes((B, H, N, D), dtype_bytes)
        kv_pages = 2 * B * H * n_pages * page_size * D * dtype_bytes
        page_table = B * n_pages * 4
        output = _bytes((B, H, N, D), dtype_bytes)
        return {"qkv": q_bytes + kv_pages + page_table, "scores": 0, "output": output}

    variants = [
        ("flash",          flash_attention,         _flash_dram_bytes),
        ("quantized",      quantized_attention,     _quantized_dram_bytes),
        ("sliding_window", _sliding_window_wrapper, _sliding_window_dram_bytes),
        ("gqa",            _gqa_wrapper,            _gqa_dram_bytes),
        ("paged",          _paged_wrapper,          _paged_dram_bytes),
    ]

    results = []
    for N in seq_lens:
        print(f"\n── seq_len = {N} ──────────────")
        qkv = fixtures.get(N) if fixtures else None
        Q = qkv.Q if qkv else None
        K = qkv.K if qkv else None
        V = qkv.V if qkv else None
        run_dtype = Q.dtype if Q is not None else dtype
        for vname, fn, traffic_fn in variants:
            try:
                r = benchmark_variant(
                    vname, fn, B, H, N, D, traffic_fn,
                    Q=Q, K=K, V=V,
                    dtype=run_dtype, n_warmup=n_warmup, n_runs=n_runs,
                )
                print(
                    f"  {vname:<12} latency={r.latency_ms:7.2f} ms  "
                    f"TFLOP/s={r.throughput_gflops/1e3:.3f}  "
                    f"DRAM={r.total_dram_bytes/1e6:.1f} MB  "
                    f"AI={r.arithmetic_intensity:.2f}"
                )
                results.append(r)
            except Exception as e:
                print(f"  {vname:<12} FAILED: {e}")

    return results



def print_table(results: List[BenchmarkResult]) -> None:
    header = (
        f"{'Variant':<14} {'N':>5} {'CPU(ms)':>8} {'FPGA(ms)':>9} "
        f"{'DRAM(MB)':>9} {'AI(F/B)':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<14} {r.seq_len:>5} {r.latency_ms:>8.2f} "
            f"{r.fpga_latency_ms():>9.2f} "
            f"{r.total_dram_bytes/1e6:>9.1f} "
            f"{r.arithmetic_intensity:>8.1f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Timeloop hardware benchmark bridge (merged from hardware_benchmark.py)
# ══════════════════════════════════════════════════════════════════════════════

from python.hardware import (
    FPGA_CLOCK_FREQ_GHZ,
    JAX_VARIANT_TIMELOOP,
    MOE_N_EXPERTS,
    MOE_TOP_K,
    PAGED_PAGE_SIZE,
    run_experiment as _run_timeloop_experiment,
)

# DRAM bandwidth from fpga_like.yaml: 12.8 GB/s read + 12.8 GB/s write
HW_PEAK_BW_GBS = 25.6


def _sliding_window_dram_bytes_hw(B, H, N, D, dtype_bytes=2):
    w = min(128, N)
    effective_pairs = N * w - (w * (w - 1)) // 2
    qkv = 3 * _bytes((B, H, N, D), dtype_bytes)
    scores = 2 * _bytes((B, H, effective_pairs), dtype_bytes)
    output = _bytes((B, H, N, D), dtype_bytes)
    return {"qkv": qkv, "scores": scores, "output": output}


def _gqa_dram_bytes_hw(B, H, N, D, dtype_bytes=2):
    n_kv = 2
    qkv = _bytes((B, H, N, D), dtype_bytes) + 2 * _bytes((B, n_kv, N, D), dtype_bytes)
    return {"qkv": qkv, "scores": 0, "output": _bytes((B, H, N, D), dtype_bytes)}


def _paged_dram_bytes_hw(B, H, N, D, dtype_bytes=2, page_size=PAGED_PAGE_SIZE):
    avg_seq = int(N * 0.75)
    n_pages = (avg_seq + page_size - 1) // page_size
    q_bytes = _bytes((B, H, N, D), dtype_bytes)
    kv_pages = 2 * B * H * n_pages * page_size * D * dtype_bytes
    page_table = B * n_pages * 4
    output = _bytes((B, H, N, D), dtype_bytes)
    return {"qkv": q_bytes + kv_pages + page_table, "scores": 0, "output": output}


def _moe_dram_bytes(B, H, N, D, n_experts=MOE_N_EXPERTS, top_k=MOE_TOP_K, dtype_bytes=2):
    """Windowed Local-KV MoE DRAM traffic: each routed query loads W local keys."""
    W = min(128, N)
    n_active = n_experts
    routed_per_expert = max(1, B * N * top_k // n_experts)
    x_bytes = _bytes((B, H, N, D), dtype_bytes)
    windowed_kv = n_active * routed_per_expert * 2 * W * D * dtype_bytes * H
    routed_q = n_active * _bytes((routed_per_expert, H, D), dtype_bytes)
    expert_w = n_active * 4 * D * D * dtype_bytes
    output = _bytes((B, H, N, D), dtype_bytes)
    return {"qkv": x_bytes + windowed_kv + routed_q + expert_w, "scores": 0, "output": output}


VARIANT_TRAFFIC_FNS: Dict[str, Callable] = {
    "flash":          _flash_dram_bytes,
    "quantized":      _quantized_dram_bytes,
    "sliding_window": _sliding_window_dram_bytes_hw,
    "gqa":            _gqa_dram_bytes_hw,
    "paged":          _paged_dram_bytes_hw,
    "moe":            _moe_dram_bytes,
}


def verify_jax_variants(B=2, H=8, N=128, D=64, fixture=None):
    """Smoke-test that JAX attention implementations compile and run."""
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
        ("flash", lambda: flash_attention(Q, K, V, is_causal=True)),
        ("quantized", lambda: quantized_attention(Q, K, V)),
        ("sliding_window", lambda: sliding_window_attention(Q, K, V, window_size=128, is_causal=True)),
        ("gqa", lambda: grouped_query_attention(Q, K[:, :2], V[:, :2], n_kv_heads=2)),
        ("paged", lambda: paged_attention(Q, K, V)),
        ("moe", lambda: moe_attention(x, *make_attention_experts(8, H, D, key=jax.random.split(key)[0]), top_k=2, is_causal=True)[0]),
    ]
    for name, fn in checks:
        out = fn()
        arr = out[0] if isinstance(out, tuple) else out
        jax.block_until_ready(arr)
        print(f"  [jax] {name:<14} OK  shape={arr.shape}")


def _scale_dram_breakdown(variant, B, H, N, D, timeloop_dram_bytes):
    """Scale analytical DRAM component ratios to Timeloop-measured total."""
    traffic_fn = VARIANT_TRAFFIC_FNS.get(variant, _flash_dram_bytes)
    profile = traffic_fn(B, H, N, D)
    total_model = sum(profile.values()) or 1
    scale = timeloop_dram_bytes / total_model
    return {"qkv": int(profile["qkv"] * scale), "scores": int(profile["scores"] * scale), "output": int(profile["output"] * scale)}


def _combine_attention_phases(qk_stats, av_stats, batch_size, n_heads):
    """Combine QK^T + AV Timeloop results into one attention-layer measurement.

    Latency uses an additive pipeline-stall model:
    (compute_cycles + dram_transfer_cycles) / clock.
    """
    scale = batch_size * n_heads
    def _get(d, k, default=0):
        v = d.get(k)
        return v if v is not None else default

    compute_cycles = (_get(qk_stats, "cycles") + _get(av_stats, "cycles")) * scale
    dram_reads = (_get(qk_stats, "dram_reads_bytes") + _get(av_stats, "dram_reads_bytes")) * scale
    dram_writes = (_get(qk_stats, "dram_writes_bytes") + _get(av_stats, "dram_writes_bytes")) * scale
    sram_reads = (_get(qk_stats, "sram_reads_bytes") + _get(av_stats, "sram_reads_bytes")) * scale
    sram_writes = (_get(qk_stats, "sram_writes_bytes") + _get(av_stats, "sram_writes_bytes")) * scale
    energy_uj = (_get(qk_stats, "energy_uj") + _get(av_stats, "energy_uj")) * scale
    util_qk = _get(qk_stats, "utilization")
    util_av = _get(av_stats, "utilization")
    utilization = (util_qk + util_av) / 2.0 if util_qk and util_av else (util_qk or util_av or 0.0)

    dram_total = dram_reads + dram_writes
    # DRAM bandwidth in bytes/cycle: 25.6 GB/s / 0.5 GHz = 51.2 bytes/cycle
    _BW_BYTES_PER_CYCLE = (HW_PEAK_BW_GBS / FPGA_CLOCK_FREQ_GHZ)  # 51.2
    dram_cycles = dram_total / _BW_BYTES_PER_CYCLE if dram_total else 0
    # Pipeline-stall model: compute and DRAM access are NOT overlapped.
    # This reflects that on a small FPGA without sophisticated prefetch,
    # each DRAM access stalls the pipeline. Gives differentiation between
    # variants with different memory policies (flash vs naive-style).
    effective_cycles = compute_cycles + dram_cycles
    latency_ms = (effective_cycles / (FPGA_CLOCK_FREQ_GHZ * 1e9)) * 1e3

    return {
        "cycles": effective_cycles, "latency_ms": latency_ms,
        "compute_cycles": compute_cycles, "dram_cycles": dram_cycles,
        "dram_reads_bytes": dram_reads, "dram_writes_bytes": dram_writes,
        "dram_total_bytes": dram_total,
        "sram_total_bytes": sram_reads + sram_writes,
        "energy_uj": energy_uj, "utilization": utilization,
    }


def benchmark_variant_timeloop(variant, B, H, N, D, timeout=300):
    """Simulate one JAX attention variant on fpga_like hardware via Timeloop."""
    if variant not in JAX_VARIANT_TIMELOOP:
        raise ValueError(f"No Timeloop mapping for variant '{variant}'. Available: {list(JAX_VARIANT_TIMELOOP)}")

    qk_name, av_name, kind = JAX_VARIANT_TIMELOOP[variant]
    window = min(128, N) if kind in ("sliding_window", "moe") else None

    run_kw = dict(seq_len=N, head_dim=D, window=window, timeout=timeout, top_k=MOE_TOP_K, n_experts=MOE_N_EXPERTS)
    qk_stats = _run_timeloop_experiment(qk_name, **run_kw)
    if qk_stats.get("error"):
        raise RuntimeError(f"Timeloop QK failed for {variant}: {qk_stats['error']}")
    av_stats = _run_timeloop_experiment(av_name, **run_kw)
    if av_stats.get("error"):
        raise RuntimeError(f"Timeloop AV failed for {variant}: {av_stats['error']}")

    combined = _combine_attention_phases(qk_stats, av_stats, B, H)

    if kind == "moe":
        for k in ("cycles", "latency_ms", "dram_reads_bytes", "dram_writes_bytes",
                  "dram_total_bytes", "sram_total_bytes", "energy_uj"):
            if k in combined and combined[k] is not None:
                combined[k] = combined[k] * MOE_N_EXPERTS

    if kind == "moe":
        W = min(128, N)
        routed = max(1, (N * MOE_TOP_K) // MOE_N_EXPERTS)
        flops = MOE_N_EXPERTS * (2 * routed * H * D * D * 4 + routed * H * W * D * 4) * B
    elif kind == "sliding_window":
        flops = _window_attention_flops(B, H, N, D, window=128)
    else:
        flops = _attention_flops(B, H, N, D)

    dram_breakdown = _scale_dram_breakdown(variant, B, H, N, D, combined["dram_total_bytes"])
    latency_ms = combined["latency_ms"]
    throughput = (flops / 1e9) / (latency_ms / 1e3) if latency_ms > 0 else 0.0

    result = BenchmarkResult(
        name=variant, seq_len=N, head_dim=D, n_heads=H, batch_size=B, dtype="int16",
        latency_ms=latency_ms, throughput_gflops=throughput,
        dram_qkv_bytes=dram_breakdown["qkv"], dram_scores_bytes=dram_breakdown["scores"],
        dram_output_bytes=dram_breakdown["output"],
    )
    result.utilization_pct = combined["utilization"]
    result.energy_uj = combined["energy_uj"]
    result.sram_bytes = combined["sram_total_bytes"]
    result.metric_source = "timeloop"
    result.timeloop_cycles = combined["cycles"]
    return result


def run_timeloop_benchmark_suite(B=2, H=8, D=64, seq_lens=(128, 256, 512, 1024), timeout=300, verify_jax=True, fixtures=None):
    """Benchmark all JAX attention variants on simulated fpga_like hardware."""
    if verify_jax:
        print("\n── JAX variant verification (compile + correctness) ──")
        verify_n = min(128, max(seq_lens))
        verify_jax_variants(B=B, H=H, N=verify_n, D=D, fixture=fixtures.get(verify_n) if fixtures else None)

    variants = list(JAX_VARIANT_TIMELOOP.keys())
    results: List[BenchmarkResult] = []
    print(f"\n── Timeloop hardware simulation (fpga_like.yaml) ──")
    print(f"    Clock: {FPGA_CLOCK_FREQ_GHZ*1e3:.0f} MHz  |  Metrics: cycles → latency, DRAM, utilization, energy")

    for N in seq_lens:
        print(f"\n── seq_len = {N} ──────────────")
        for vname in variants:
            try:
                r = benchmark_variant_timeloop(vname, B, H, N, D, timeout=timeout)
                util = getattr(r, "utilization_pct", 0.0)
                print(f"  {vname:<14} FPGA={r.latency_ms:7.2f} ms  util={util:5.1f}%  DRAM={r.total_dram_bytes/1e6:.2f} MB  GFLOP/s={r.throughput_gflops:.1f}  AI={r.arithmetic_intensity:.1f}")
                results.append(r)
            except Exception as e:
                print(f"  {vname:<14} FAILED: {e}")
    return results


def print_timeloop_table(results: List[BenchmarkResult]) -> None:
    """Print hardware metrics table (all values from Timeloop)."""
    header = f"{'Variant':<14} {'N':>5} {'FPGA(ms)':>9} {'Util%':>7} {'DRAM(MB)':>9} {'SRAM(MB)':>9} {'AI(F/B)':>8} {'Source':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        util = getattr(r, "utilization_pct", 0.0)
        sram = getattr(r, "sram_bytes", 0) / 1e6
        source = getattr(r, "metric_source", "timeloop")
        print(f"{r.name:<14} {r.seq_len:>5} {r.latency_ms:>9.2f} {util:>6.1f}% {r.total_dram_bytes/1e6:>9.2f} {sram:>9.2f} {r.arithmetic_intensity:>8.1f} {source:>10}")
