"""Grouped Query Attention (GQA): shares KV heads across query head groups."""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple


def grouped_query_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    n_kv_heads: Optional[int] = None,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> Tuple[jnp.ndarray, dict]:
    """
    Args:
        Q:          [batch, n_heads, seq_len, head_dim]
        K:          [batch, n_kv_heads, seq_len, head_dim]  (or [batch, n_heads, ...] for MHA)
        V:          [batch, n_kv_heads, seq_len, head_dim]
        n_kv_heads: number of KV heads. If None, inferred from K.shape[1].
        scale:      attention scaling factor (default: 1/sqrt(head_dim))
        is_causal:  apply causal mask

    Returns:
        output: [batch, n_heads, seq_len, head_dim]
        stats:  dict with KV savings metrics
    """
    B, H_q, N, D = Q.shape
    H_kv = K.shape[1] if n_kv_heads is None else n_kv_heads

    assert H_q % H_kv == 0, f"n_heads ({H_q}) must be divisible by n_kv_heads ({H_kv})"
    group_size = H_q // H_kv

    if scale is None:
        scale = 1.0 / np.sqrt(D)

    # Expand KV heads to match Q heads by repeating
    # K: [B, H_kv, N, D] → [B, H_kv, 1, N, D] → [B, H_kv, group_size, N, D] → [B, H_q, N, D]
    K_expanded = jnp.repeat(K, group_size, axis=1)  # [B, H_q, N, D]
    V_expanded = jnp.repeat(V, group_size, axis=1)  # [B, H_q, N, D]

    # Standard attention
    scores = jnp.einsum("bhid,bhjd->bhij", Q, K_expanded) * scale

    if is_causal:
        causal_mask = jnp.tril(jnp.ones((N, N), dtype=jnp.bool_))
        scores = jnp.where(causal_mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)

    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum("bhij,bhjd->bhid", attn_weights, V_expanded)

    # KV cache savings
    mha_kv_bytes = 2 * B * H_q * N * D * 2   # K + V, FP16
    gqa_kv_bytes = 2 * B * H_kv * N * D * 2
    kv_memory_saved = 1.0 - (gqa_kv_bytes / mha_kv_bytes) if mha_kv_bytes > 0 else 0.0

    stats = {
        "n_heads": H_q,
        "n_kv_heads": H_kv,
        "group_size": group_size,
        "kv_cache_bytes_mha": mha_kv_bytes,
        "kv_cache_bytes_gqa": gqa_kv_bytes,
        "kv_memory_saved_ratio": kv_memory_saved,
        "kv_bandwidth_reduction": H_q / H_kv,
    }

    return output, stats
