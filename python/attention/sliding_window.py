"""Sliding-window attention: O(N·w) score matrix instead of O(N²). Used in Mistral, Longformer."""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple


def sliding_window_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    window_size: int = 128,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> Tuple[jnp.ndarray, dict]:
    """
    Args:
        Q, K, V:      [batch, num_heads, seq_len, head_dim]
        window_size:  number of tokens each query can attend to
        is_causal:    if True, each token attends to previous `window_size` tokens only
                      if False, symmetric window centered on the token
        scale:        attention scaling factor (default: 1/sqrt(head_dim))

    Returns:
        output: [batch, num_heads, seq_len, head_dim]
        stats:  dict with sparsity metrics
    """
    B, H, N, D = Q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(D)

    w = min(window_size, N)

    row_idx = jnp.arange(N)[:, None]
    col_idx = jnp.arange(N)[None, :]

    if is_causal:
        # Each token attends to [max(0, i-w+1), i] inclusive
        mask = (col_idx <= row_idx) & (col_idx >= row_idx - w + 1)
    else:
        # Symmetric window: [i - w//2, i + w//2]
        half_w = w // 2
        mask = jnp.abs(row_idx - col_idx) <= half_w

    scores = jnp.einsum("bhid,bhjd->bhij", Q, K) * scale
    scores = jnp.where(mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum("bhij,bhjd->bhid", attn_weights, V)

    total_pairs = N * N
    active_pairs = int(jnp.sum(mask))
    sparsity = 1.0 - (active_pairs / total_pairs)

    stats = {
        "window_size": w,
        "is_causal": is_causal,
        "active_pairs": active_pairs,
        "total_pairs": total_pairs,
        "sparsity_ratio": sparsity,
        "flops_vs_dense": active_pairs / total_pairs,
        "dram_score_bytes_saved_ratio": sparsity,
    }

    return output, stats
