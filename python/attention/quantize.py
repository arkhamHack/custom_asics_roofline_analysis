"""INT8 flash-style quantized attention — tiled O(N) memory, 4× bandwidth over FP32."""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Dict
from jax import lax


def quantize(x: jnp.ndarray, bits: int = 8) -> Tuple[jnp.ndarray, float]:
    """Symmetric per-tensor quantization to int8."""
    n_levels = (1 << (bits - 1)) - 1
    abs_max = jnp.max(jnp.abs(x))
    scale = jnp.maximum(abs_max / n_levels, 1e-8)
    x_q = jnp.clip(jnp.round(x / scale), -n_levels, n_levels).astype(jnp.int8)
    return x_q, scale


def dequantize(x_q: jnp.ndarray, scale: float) -> jnp.ndarray:
    return x_q.astype(jnp.float32) * scale


def quantized_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    bits: int = 8,
    scale: Optional[float] = None,
    block_size: int = 64,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Flash-style INT8 quantized attention. Never materialises the full N×N score matrix.
    Q/K/V are quantized to INT8 globally; each tile is dequantized to FP32 for accumulation.

    Args:
        Q, K, V:    [batch, num_heads, seq_len, head_dim]
        bits:       quantization bits (default: 8 → INT8)
        scale:      attention scaling factor (default: 1/sqrt(head_dim))
        block_size: tile size for tiled computation

    Returns:
        output:            [batch, num_heads, seq_len, head_dim]
        quantization_info: dict with per-tensor scales and bandwidth stats
    """
    d_k = Q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(float(d_k))

    Q_q, Q_s = quantize(Q, bits)
    K_q, K_s = quantize(K, bits)
    V_q, V_s = quantize(V, bits)

    B, H, N, D = Q_q.shape
    Br = min(block_size, N)
    Bc = min(block_size, N)
    num_q_blocks = N // Br
    num_k_blocks = N // Bc

    def single_head_attention(q_q, k_q, v_q):
        q_blocks = q_q.reshape(num_q_blocks, Br, D)
        k_blocks = k_q.reshape(num_k_blocks, Bc, D)
        v_blocks = v_q.reshape(num_k_blocks, Bc, D)

        def outer_loop(carry, qi):
            mi0 = jnp.full((Br,), -jnp.inf)
            li0 = jnp.zeros((Br,))
            oi0 = jnp.zeros((Br, D))
            qi_f = qi.astype(jnp.float32) * Q_s

            def inner_loop(stats, kv_data):
                mi, li, oi = stats
                kj, vj = kv_data
                kj_f = kj.astype(jnp.float32) * K_s
                vj_f = vj.astype(jnp.float32) * V_s

                sij = jnp.einsum("id,jd->ij", qi_f, kj_f) * scale
                mij_new = jnp.maximum(mi, jnp.max(sij, axis=1))
                alpha = jnp.exp(jnp.where(mi == -jnp.inf, 0.0, mi - mij_new))
                pij = jnp.exp(sij - mij_new[:, None])
                pij = jnp.where(jnp.isnan(pij), 0.0, pij)
                li_new = alpha * li + jnp.sum(pij, axis=1)
                oi_new = alpha[:, None] * oi + jnp.einsum("ij,jd->id", pij, vj_f)
                return (mij_new, li_new, oi_new), None

            (_, li_f, oi_f), _ = lax.scan(inner_loop, (mi0, li0, oi0), (k_blocks, v_blocks))
            out = jnp.where(li_f[:, None] > 0, oi_f / li_f[:, None], jnp.zeros((Br, D)))
            return carry, out

        _, o_blocks = lax.scan(outer_loop, None, q_blocks)
        return o_blocks.reshape(N, D)

    head_vmapped = jax.vmap(single_head_attention, in_axes=(0, 0, 0), out_axes=0)
    batch_vmapped = jax.vmap(head_vmapped, in_axes=(0, 0, 0))
    output = batch_vmapped(Q_q, K_q, V_q)

    return output.astype(Q.dtype), {
        "bits": bits,
        "bytes_per_element": bits / 8,
        "Q_scale": Q_s,
        "K_scale": K_s,
        "V_scale": V_s,
        "bandwidth_reduction_vs_fp32": 32 / bits,
    }
