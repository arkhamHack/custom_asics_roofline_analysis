"""JAX → StableHLO compiler inspection: extracts GemmOps and computes arithmetic intensity."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np



@dataclass
class GemmOp:
    """One dot_general operation extracted from HLO IR."""
    label: str
    lhs_shape: Tuple[int, ...]
    rhs_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    batch_dims: Tuple[int, ...] = field(default_factory=tuple)
    contracting_dims: Tuple[int, ...] = field(default_factory=tuple)
    dtype_bytes: int = 2

    @property
    def flops(self) -> int:
        """FLOPs = 2 × product(batch) × product(output) × product(contraction)."""
        batch = 1
        for d in self.batch_dims:
            batch *= self.lhs_shape[d]
        # output spatial dims
        lhs_free = [s for i, s in enumerate(self.lhs_shape)
                    if i not in self.batch_dims and i not in self.contracting_dims]
        rhs_free = [s for i, s in enumerate(self.rhs_shape)
                    if i not in self.batch_dims and i not in self.contracting_dims]
        output_elems = batch * int(np.prod(lhs_free)) * int(np.prod(rhs_free))
        contraction = int(np.prod([self.lhs_shape[i] for i in self.contracting_dims]))
        return 2 * output_elems * contraction

    @property
    def memory_bytes(self) -> int:
        """Bytes to load LHS + RHS + store output (no reuse assumed)."""
        def nbytes(shape):
            return int(np.prod(shape)) * self.dtype_bytes
        return nbytes(self.lhs_shape) + nbytes(self.rhs_shape) + nbytes(self.out_shape)

    @property
    def arithmetic_intensity(self) -> float:
        """FLOP / byte — the roofline x-axis."""
        return self.flops / self.memory_bytes

    def __repr__(self) -> str:
        return (
            f"GemmOp({self.label!r} | "
            f"lhs={self.lhs_shape} rhs={self.rhs_shape} out={self.out_shape} | "
            f"FLOPs={self.flops:,}  bytes={self.memory_bytes:,}  "
            f"AI={self.arithmetic_intensity:.2f})"
        )


# ── HLO extraction ────────────────────────────────────────────────────────────

def get_hlo_text(fn, *sample_inputs, path: Path = None) -> str:
    """
    JIT-compile *fn* and return the StableHLO text IR.
    Optionally write to *path* for inspection.
    """
    lowered = jax.jit(fn).lower(*sample_inputs)
    hlo_text = lowered.as_text()          # StableHLO MLIR text
    if path is not None:
        Path(path).write_text(hlo_text)
    return hlo_text


# ── HLO parser ────────────────────────────────────────────────────────────────

# Matches:  %0 = stablehlo.dot_general(%q, %k), ...
#           : (tensor<2x8x512x64xf16>, tensor<2x8x512x64xf16>) -> tensor<2x8x512x512xf16>
_DOT_GENERAL_RE = re.compile(
    r"stablehlo\.dot_general\s*\([^)]*\)"   # op call
    r".*?"                                   # attributes (non-greedy)
    r":\s*\(([^)]+)\)"                       # operand types
    r"\s*->\s*([^\n,]+)",                    # result type
    re.DOTALL,
)
_TENSOR_SHAPE_RE = re.compile(r"tensor<([\dx]+)x(\w+)>")


def _parse_shape(type_str: str) -> Tuple[int, ...]:
    m = _TENSOR_SHAPE_RE.search(type_str.strip())
    if not m:
        return ()
    dims_str = m.group(1)
    return tuple(int(d) for d in dims_str.split("x") if d)


def extract_gemm_ops(hlo_text: str) -> List[GemmOp]:
    """Extract all dot_general operations from StableHLO text."""
    ops = []
    for i, m in enumerate(_DOT_GENERAL_RE.finditer(hlo_text)):
        operands_str, result_str = m.group(1), m.group(2)
        parts = [p.strip() for p in operands_str.split(",")]
        if len(parts) < 2:
            continue
        lhs_shape = _parse_shape(parts[0])
        rhs_shape = _parse_shape(parts[1])
        out_shape = _parse_shape(result_str)
        if not (lhs_shape and rhs_shape and out_shape):
            continue

        dtype_bytes = 2 if "f16" in parts[0] else 4
        label = _label_gemm(lhs_shape, rhs_shape, out_shape, i)
        ops.append(GemmOp(
            label=label,
            lhs_shape=lhs_shape,
            rhs_shape=rhs_shape,
            out_shape=out_shape,
            batch_dims=tuple(range(len(lhs_shape) - 2)),
            contracting_dims=(len(lhs_shape) - 1,),
            dtype_bytes=dtype_bytes,
        ))
    return ops


def _label_gemm(lhs, rhs, out, idx: int) -> str:
    """Heuristic: if output is square it's QK^T, otherwise AV."""
    if len(out) >= 2 and out[-1] == out[-2]:
        return "QK^T"
    if len(out) >= 2 and out[-1] != out[-2]:
        return "AV"
    return f"gemm_{idx}"


# ── Top-level analysis ────────────────────────────────────────────────────────

def analyze_attention_hlo(
    B: int = 2,
    H: int = 8,
    N: int = 512,
    D: int = 64,
    sample: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = None,
    dump_dir: Path = None,
) -> dict:
    """
    Compile all three attention variants, extract GemmOps, return analysis dict.

    Returns:
        {
          "naive":     [GemmOp, ...],
          "flash":     [GemmOp, ...],
          "quantized": [GemmOp, ...],
        }
    """
    from python.attention.naive import naive_attention
    from python.attention.flash import flash_attention
    from python.attention.quantize import quantized_attention
    from python.attention.paged_attention import paged_attention
    from python.attention.moe_attention import moe_attention, make_attention_experts

    if sample is None:
        sample = (jnp.ones((B, H, N, D)), jnp.ones((B, H, N, D)), jnp.ones((B, H, N, D)))
    x_sample = (sample[0] + sample[1] + sample[2]) / 3.0
    moe_w = make_attention_experts(8, H, D, jax.random.PRNGKey(0))

    variants = {
        "naive":     naive_attention,
        "flash":     flash_attention,
        "quantized": lambda Q, K, V: quantized_attention(Q, K, V)[0],
        "paged":     lambda Q, K, V: paged_attention(Q, K, V)[0],
        "moe":       lambda Q, K, V: moe_attention(
            x_sample, moe_w[0], moe_w[1], moe_w[2], moe_w[3], moe_w[4],
            top_k=2, is_causal=True,
        )[0],
    }

    results = {}
    for name, fn in variants.items():
        dump_path = (Path(dump_dir) / f"{name}.mlir") if dump_dir else None
        try:
            hlo = get_hlo_text(fn, *sample, path=dump_path)
            ops = extract_gemm_ops(hlo)
        except Exception as e:
            print(f"[compiler] Warning: could not analyse '{name}': {e}")
            ops = []
        results[name] = ops

    return results


def print_analysis(analysis: dict) -> None:
    """Print arithmetic intensity table for all variants."""
    print(f"\n{'Variant':<12} {'Op':<8} {'FLOPs':>12} {'Bytes':>12} {'AI (F/B)':>10}")
    print("-" * 58)
    for variant, ops in analysis.items():
        for op in ops:
            print(
                f"{variant:<12} {op.label:<8} "
                f"{op.flops:>12,} {op.memory_bytes:>12,} "
                f"{op.arithmetic_intensity:>10.2f}"
            )
