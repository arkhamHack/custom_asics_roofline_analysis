"""
Deformable Self-Attention — Xia et al., "Vision Transformer with Deformable Attention" (CVPR 2022)
https://arxiv.org/abs/2201.00520

Adapted for 1D sequences (text/tokens) with FPGA block-snapped position sampling.

Paper mechanism:
  1. Predict K sparse positions from Q via W_offset
  2. Gather both K and V at those positions
  3. Compute standard Q·K^T dot-product over n_points (not N)
  → Score matrix is [N, n_points], never [N, N]

FPGA adaptation:
  Paper: bilinear interpolation at fractional 2D positions (image patches)
  Here:  block-snap to nearest block boundary in 1D (token sequence)
         K and V block means precomputed → sequential DMA, register-file lookup

Memory traffic (N=512, D=64, n_points=4, FP16):
  Flash:  score = 0 KB DRAM            (tiles in SRAM)
  DSA:    """
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional, Tuple
from jax import lax




def make_offset_weights(D:int,n_points:int) -> jnp.ndarray:
    """Learnable linear layer to predict offsets from query means."""
    W_offset = jax.random.normal(jax.random.PRNGKey(42), (D, n_points), dtype=jnp.float16)
    return W_offset

@partial(jax.jit, static_argnames=["block_size", "n_points"])
def deformable_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    W_offset: jnp.ndarray,
    block_size: int = 64,
    n_points: int = 4,
    scale: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Args:
        Q, K, V:     [batch, num_heads, seq_len, head_dim]
        W_offset:    [head_dim, n_points]  linear layer to predict offsets
        block_size:  block size for block-snap position selection
        n_points:    number of blocks to select per query row
        scale:       attention scaling factor (default: 1/sqrt(head_dim))

    Returns:
        output:       [batch, num_heads, seq_len, head_dim]
        attn_weights: [batch, num_heads, seq_len, n_points]  ← reduced score matrix
    """
    B, H, N, D = Q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(D)

    # Precompute block means for K and V — O(B·H·num_blocks·D) — trivial FPGA logic
    num_blocks = N // block_size
    def single_head(q,k,v):
        k_means=k.reshape(num_blocks, block_size, D).mean(axis=1)  # [num_blocks, D]
        v_means = v.reshape(num_blocks, block_size, D).mean(axis=1)  # [num_blocks, D]
        ref = jnp.arange(N,dtype=jnp.float32)
        offsets = jnp.tanh(q@W_offset) * (N/2.0)
        positions = ref[:,None] + offsets  # [N, n_points]

        block_idx = jnp.clip(jnp.round(positions/block_size).astype(jnp.int32),0,num_blocks-1)  # [N, n_points]
        k_sampled = k_means[block_idx]  # [N, n_points, D]
        v_sampled = v_means[block_idx]   # [N, n_points]
        attn_logits = jnp.einsum("nd,nkd->nk",q,k_sampled)*scale
        attn_weights = jax.nn.softmax(attn_logits,axis=-1)  # [N, n_points]
        out = jnp.einsum("nk,nkd->nd",attn_weights,v_sampled)
        return out,attn_weights
    return jax.vmap(jax.vmap(single_head))(Q,K,V)