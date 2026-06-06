"""Standard scaled dot-product attention. Materialises the full N×N score matrix → O(N²) DRAM traffic."""
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional, Tuple


@partial(jax.jit, static_argnames=["scale"])
def naive_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    scale: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Args:
        Q, K, V: [batch, heads, seq_len, head_dim]
        mask:    [batch, heads, seq_len, seq_len]
        scale:   attention scaling factor (default: 1/sqrt(head_dim))

    Returns:
        output:       [batch, heads, seq_len, head_dim]
        attn_weights: [batch, heads, seq_len, seq_len]
    """
    if scale is None:
        scale = 1.0 / jnp.sqrt(Q.shape[-1])

    scores = jnp.einsum("bhid,bhjd->bhij", Q, K) * scale
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum("bhij,bhjd->bhid", attn_weights, V)
    return output, attn_weights
