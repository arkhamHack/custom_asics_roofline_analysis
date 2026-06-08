"""JAX attention benchmark suite."""
import time
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional

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


def _naive_dram_bytes(B: int, H: int, N: int, D: int, dtype_bytes: int = 2) -> dict:
    qkv = 3 * _bytes((B, H, N, D), dtype_bytes)
    scores = 2 * _bytes((B, H, N, N), dtype_bytes)
    output = _bytes((B, H, N, D), dtype_bytes)
    return {"qkv": qkv, "scores": scores, "output": output}


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
    from python.attention.naive import naive_attention
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

    variants = [
        ("naive",          naive_attention,         _naive_dram_bytes),
        ("flash",          flash_attention,         _flash_dram_bytes),
        ("quantized",      quantized_attention,     _quantized_dram_bytes),
        ("sliding_window", _sliding_window_wrapper, _sliding_window_dram_bytes),
        ("gqa",            _gqa_wrapper,            _gqa_dram_bytes),
        ("paged",          _paged_wrapper,          _naive_dram_bytes),
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
