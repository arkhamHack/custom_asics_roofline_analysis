"""Flash Attention v2 — tiled, memory-efficient exact attention (online softmax).

Keeps Q/K/V tiles in SRAM; never materialises the full N×N score matrix.
Reference: Dao et al. 2022.
"""
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional
from jax import lax


def flash_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    block_size: int = 64,
    scale: Optional[float] = None,
    is_causal: bool = False,
    q_seq_lens: Optional[jnp.ndarray] = None,
    kv_seq_lens: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Args:
        Q, K, V:      [batch, num_heads, seq_len, head_dim]
        block_size:   SRAM tile size for both query and key/value blocks
        scale:        attention scaling factor (default: 1/sqrt(head_dim))
        is_causal:    if True, apply causal (autoregressive) mask so position i
                      can only attend to positions <= i.
        q_seq_lens:   [batch] — actual (unpadded) query lengths per sample.
                      If None, all positions are treated as valid.
        kv_seq_lens:  [batch] — actual (unpadded) key/value lengths per sample.
                      If None, all positions are treated as valid.

    Returns:
        output: [batch, num_heads, seq_len, head_dim]
    """
    B, H, N, D = Q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(D)

    Br = min(block_size, N)   # query block size
    Bc = min(block_size, N)   # key/value block size
    num_q_blocks = N // Br
    num_k_blocks = N // Bc

    positions = jnp.arange(N)
    if q_seq_lens is not None:
        q_valid = positions[None, :] < q_seq_lens[:, None]   # [B, N]
    else:
        q_valid = jnp.ones((B, N), dtype=jnp.bool_)

    if kv_seq_lens is not None:
        kv_valid = positions[None, :] < kv_seq_lens[:, None]  # [B, N]
    else:
        kv_valid = jnp.ones((B, N), dtype=jnp.bool_)

    def single_head_attention(q, k, v, q_mask, kv_mask):
        # q, k, v: [N, D] — one head, one batch element
        # q_mask, kv_mask: [N] boolean — True for valid positions
        q_blocks = q.reshape(num_q_blocks, Br, D)       # [Nq, Br, D]
        k_blocks = k.reshape(num_k_blocks, Bc, D)       # [Nk, Bc, D]
        v_blocks = v.reshape(num_k_blocks, Bc, D)       # [Nk, Bc, D]
        kv_mask_blocks = kv_mask.reshape(num_k_blocks, Bc)  # [Nk, Bc]

        # q_block_indices[i] = starting row index of i-th query block
        q_block_starts = jnp.arange(num_q_blocks) * Br   # [Nq]

        def outer_loop(carry, block_data):
            qi, q_block_start = block_data
            # qi: [Br, D], q_block_start: scalar

            mi0 = jnp.full((Br,), -jnp.inf)
            li0 = jnp.zeros((Br,))
            oi0 = jnp.zeros((Br, D))

            # Query-position indices for this block: [Br]
            q_positions = q_block_start + jnp.arange(Br)

            # Precompute query padding mask: [Br] — True for valid rows
            q_row_valid = lax.dynamic_slice(q_mask, (q_block_start,), (Br,))

            def inner_loop(stats, kv_data):
                mi, li, oi = stats
                kj, vj, kv_col_valid, kv_block_idx = kv_data
                # kj, vj: [Bc, D]; kv_col_valid: [Bc]; kv_block_idx: scalar

                # Key-position indices for this block: [Bc]
                k_positions = kv_block_idx * Bc + jnp.arange(Bc)

                # Score tile: [Br, Bc]
                sij = jnp.einsum("id,jd->ij", qi, kj) * scale

                # ── Causal mask ───────────────────────────────────────────
                # mask[r, c] = True if key position c <= query position r
                if is_causal:
                    causal_mask = k_positions[None, :] <= q_positions[:, None]  # [Br, Bc]
                    sij = jnp.where(causal_mask, sij, -jnp.inf)

                # ── Padding masks ─────────────────────────────────────────
                # Zero out scores for padded key positions
                sij = jnp.where(kv_col_valid[None, :], sij, -jnp.inf)
                # Zero out scores for padded query rows (their output will be 0)
                sij = jnp.where(q_row_valid[:, None], sij, -jnp.inf)

                # ── Online softmax ────────────────────────────────────────
                mij_new = jnp.maximum(mi, jnp.max(sij, axis=1))   # [Br]
                # Guard against -inf - (-inf) = nan
                alpha = jnp.exp(jnp.where(mi == -jnp.inf, 0.0, mi - mij_new))
                pij = jnp.exp(sij - mij_new[:, None])             # [Br, Bc]
                # Replace nan from -inf - (-inf) with 0
                pij = jnp.where(jnp.isnan(pij), 0.0, pij)

                li_new = alpha * li + jnp.sum(pij, axis=1)
                oi_new = alpha[:, None] * oi + jnp.einsum("ij,jd->id", pij, vj)
                return (mij_new, li_new, oi_new), None

            kv_block_indices = jnp.arange(num_k_blocks)
            (_, li_f, oi_f), _ = lax.scan(
                inner_loop,
                (mi0, li0, oi0),
                (k_blocks, v_blocks, kv_mask_blocks, kv_block_indices),
            )
            # Normalise; avoid division by zero for fully-masked rows
            out = jnp.where(
                li_f[:, None] > 0,
                oi_f / li_f[:, None],
                jnp.zeros((Br, D)),
            )
            return carry, out

        _, o_blocks = lax.scan(
            outer_loop, None, (q_blocks, q_block_starts)
        )  # [Nq, Br, D]
        return o_blocks.reshape(N, D)

    head_vmapped = jax.vmap(single_head_attention, in_axes=(0, 0, 0, None, None), out_axes=0)

    def per_batch(q, k, v, q_m, kv_m):
        return head_vmapped(q, k, v, q_m, kv_m)

    batch_vmapped = jax.vmap(per_batch, in_axes=(0, 0, 0, 0, 0))
    out = batch_vmapped(Q, K, V, q_valid, kv_valid)
    return out

