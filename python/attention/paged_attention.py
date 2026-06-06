"""Paged attention (vLLM-style): fixed-size page KV cache management."""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple


def build_page_table(
    seq_lens: jnp.ndarray,
    page_size: int,
    max_pages: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a page table mapping logical pages to physical pages.

    Args:
        seq_lens:   [batch] — actual sequence lengths
        page_size:  tokens per page
        max_pages:  total physical pages available

    Returns:
        page_table:  [batch, max_logical_pages] — maps to physical page index (-1 = unallocated)
        page_alloc:  [max_pages] — True if physical page is allocated
    """
    B = seq_lens.shape[0]
    max_seq = int(jnp.max(seq_lens))
    max_logical = (max_seq + page_size - 1) // page_size

    # Simple sequential allocation (no sharing in this version)
    page_table = np.full((B, max_logical), -1, dtype=np.int32)
    page_alloc = np.zeros(max_pages, dtype=bool)
    next_page = 0

    for b in range(B):
        n_pages_needed = (int(seq_lens[b]) + page_size - 1) // page_size
        for p in range(n_pages_needed):
            if next_page < max_pages:
                page_table[b, p] = next_page
                page_alloc[next_page] = True
                next_page += 1

    return jnp.array(page_table), jnp.array(page_alloc)


def paged_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    page_size: int = 16,
    seq_lens: Optional[jnp.ndarray] = None,
    scale: Optional[float] = None,
) -> Tuple[jnp.ndarray, dict]:
    """
    Paged attention: organizes KV cache into fixed-size pages.

    This simulates the paged memory access pattern — attention computation
    is mathematically identical to standard attention, but memory layout
    uses page-granularity allocation.

    Args:
        Q, K, V:     [batch, num_heads, seq_len, head_dim]
        page_size:   tokens per page (typically 16 or 32)
        seq_lens:    [batch] — actual lengths (None = all positions valid)
        scale:       attention scaling factor

    Returns:
        output: [batch, num_heads, seq_len, head_dim]
        stats:  dict with page utilization metrics
    """
    B, H, N, D = Q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(D)

    if seq_lens is None:
        seq_lens = jnp.full((B,), N)

    # Page table construction
    n_pages_per_seq = (N + page_size - 1) // page_size
    total_pages_needed = int(jnp.sum(
        (seq_lens + page_size - 1) // page_size
    ))
    max_pages = B * n_pages_per_seq  # worst case: fully contiguous

    page_table, page_alloc = build_page_table(seq_lens, page_size, max_pages)

    K_pages = K.reshape(B, H, n_pages_per_seq, page_size, D)
    V_pages = V.reshape(B, H, n_pages_per_seq, page_size, D)

    # For each batch element, mask out pages beyond actual length
    page_mask = jnp.arange(n_pages_per_seq)[None, :] < (
        (seq_lens[:, None] + page_size - 1) // page_size
    )  # [B, n_pages]

    # Expand page mask to token level: [B, N]
    token_mask = jnp.arange(N)[None, :] < seq_lens[:, None]

    # Standard attention with padding mask
    scores = jnp.einsum("bhid,bhjd->bhij", Q, K) * scale
    scores = jnp.where(
        token_mask[:, None, None, :],  # [B, 1, 1, N]
        scores,
        jnp.finfo(scores.dtype).min,
    )
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum("bhij,bhjd->bhid", attn_weights, V)

    # Zero out padded query positions
    output = jnp.where(token_mask[:, None, :, None], output, 0.0)

    # Utilization stats
    pages_allocated = int(jnp.sum(page_alloc))
    total_tokens_in_pages = pages_allocated * page_size
    actual_tokens = int(jnp.sum(seq_lens))
    internal_frag = 1.0 - (actual_tokens / total_tokens_in_pages) if total_tokens_in_pages > 0 else 0.0

    # Compare vs contiguous allocation (max_seq_len per sequence)
    contiguous_mem = B * N * D * 2 * 2  # K+V, 2 bytes each
    paged_mem = pages_allocated * page_size * D * 2 * 2
    mem_saved = 1.0 - (paged_mem / contiguous_mem) if contiguous_mem > 0 else 0.0

    stats = {
        "page_size": page_size,
        "pages_allocated": pages_allocated,
        "max_pages": max_pages,
        "page_utilization": pages_allocated / max_pages if max_pages > 0 else 0.0,
        "internal_fragmentation": internal_frag,
        "memory_saved_vs_contiguous": mem_saved,
        "actual_tokens": actual_tokens,
        "total_token_slots": total_tokens_in_pages,
    }

    return output, stats
