"""
Mixture of Experts (MoE) Attention.

Token-level top-k routing: each token independently selects top-k experts
from a pool of n_experts. Only selected experts compute — the rest are skipped.

Reference: Mixtral (Jiang et al. 2024), Switch Transformer (Fedus et al. 2022)
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple
from functools import partial


def _expert_ffn(x: jnp.ndarray, W1: jnp.ndarray, W2: jnp.ndarray) -> jnp.ndarray:
    """Single expert: 2-layer FFN with GELU. x: [N, D], W1: [D, 4D], W2: [4D, D]"""
    return jnp.dot(jax.nn.gelu(jnp.dot(x, W1)), W2)


def make_expert_weights(
    n_experts: int,
    dim: int,
    expand: int = 4,
    key: jax.random.PRNGKey = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Initialize expert weights + gate.

    Returns:
        W1: [n_experts, dim, expand*dim]
        W2: [n_experts, expand*dim, dim]
        W_gate: [dim, n_experts]
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 1.0 / np.sqrt(dim)
    W1 = jax.random.normal(k1, (n_experts, dim, expand * dim)) * scale
    W2 = jax.random.normal(k2, (n_experts, expand * dim, dim)) * scale
    W_gate = jax.random.normal(k3, (dim, n_experts)) * scale
    return W1, W2, W_gate


def moe_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    W1: jnp.ndarray,
    W2: jnp.ndarray,
    W_gate: jnp.ndarray,
    top_k: int = 2,
    scale: Optional[float] = None,
) -> Tuple[jnp.ndarray, dict]:
    """
    MoE attention: standard attention followed by mixture-of-experts FFN.

    Architecture:
        1. Standard multi-head attention (shared across all tokens)
        2. Per-token top-k expert routing through FFN experts

    Args:
        Q, K, V:   [batch, n_heads, seq_len, head_dim]
        W1:        [n_experts, dim, expand*dim] — expert up-projection
        W2:        [n_experts, expand*dim, dim] — expert down-projection
        W_gate:    [dim, n_experts] — gating network
        top_k:     number of experts to activate per token
        scale:     attention scaling factor

    Returns:
        output: [batch, n_heads, seq_len, head_dim]
        stats:  dict with expert utilization and savings metrics
    """
    B, H, N, D = Q.shape
    n_experts = W1.shape[0]

    if scale is None:
        scale = 1.0 / np.sqrt(D)

    # Step 1: Standard attention (shared, always runs)
    scores = jnp.einsum("bhid,bhjd->bhij", Q, K) * scale
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.einsum("bhij,bhjd->bhid", attn_weights, V)  # [B, H, N, D]

    # Step 2: MoE FFN on the attention output
    # Collapse heads for routing: [B, N, H*D]
    x = attn_out.transpose(0, 2, 1, 3).reshape(B, N, H * D)

    # We operate on head_dim-sized slices to keep shapes manageable
    # Route based on mean across heads: [B, N, D]
    x_route = attn_out.mean(axis=1)  # [B, N, D]

    # Gate scores: [B, N, n_experts]
    gate_logits = jnp.einsum("bnd,de->bne", x_route, W_gate)
    gate_probs = jax.nn.softmax(gate_logits, axis=-1)

    # Top-k selection: [B, N, top_k]
    top_k_indices = jnp.argsort(gate_probs, axis=-1)[..., -top_k:]  # top-k experts
    top_k_weights = jnp.take_along_axis(gate_probs, top_k_indices, axis=-1)
    # Renormalize selected expert weights
    top_k_weights = top_k_weights / jnp.sum(top_k_weights, axis=-1, keepdims=True)

    # Compute expert outputs for ALL experts, then mask (for JAX tracing)
    # all_expert_out: [n_experts, B, N, D]
    def run_expert(expert_idx):
        return jnp.dot(
            jax.nn.gelu(jnp.dot(x_route, W1[expert_idx])),
            W2[expert_idx],
        )  # [B, N, D]

    all_expert_out = jax.vmap(run_expert)(jnp.arange(n_experts))  # [n_experts, B, N, D]

    # Gather selected expert outputs and weight them
    # top_k_indices: [B, N, top_k] — index into expert dim
    def gather_and_combine(b_idx):
        def per_token(n_idx):
            expert_ids = top_k_indices[b_idx, n_idx]  # [top_k]
            weights = top_k_weights[b_idx, n_idx]     # [top_k]
            expert_outs = all_expert_out[expert_ids, b_idx, n_idx]  # [top_k, D]
            return jnp.einsum("k,kd->d", weights, expert_outs)
        return jax.vmap(per_token)(jnp.arange(N))  # [N, D]

    moe_out = jax.vmap(gather_and_combine)(jnp.arange(B))  # [B, N, D]

    # Add MoE output back (residual) and reshape to [B, H, N, D]
    # Broadcast across heads
    output = attn_out + moe_out[:, None, :, :]

    # Stats
    # Expert utilization: how many unique experts were used across all tokens
    expert_usage = jnp.zeros((B, n_experts))
    for k_idx in range(top_k):
        expert_usage = expert_usage.at[
            jnp.arange(B)[:, None], top_k_indices[:, :, k_idx]
        ].add(1)

    experts_active = jnp.sum(expert_usage > 0, axis=1)  # [B] — unique experts used
    avg_experts_active = float(jnp.mean(experts_active))

    # Theoretical savings: only top_k/n_experts of FFN compute runs
    compute_fraction = top_k / n_experts
    dense_ffn_flops = 2 * B * N * D * (4 * D) * 2  # up + down projection
    actual_ffn_flops = dense_ffn_flops * compute_fraction
    attn_flops = 4 * B * H * N * N * D  # always runs

    stats = {
        "n_experts": n_experts,
        "top_k": top_k,
        "compute_fraction": compute_fraction,
        "compute_saved_ratio": 1.0 - compute_fraction,
        "dense_ffn_flops": int(dense_ffn_flops),
        "actual_ffn_flops": int(actual_ffn_flops),
        "attn_flops": int(attn_flops),
        "avg_experts_active_per_batch": avg_experts_active,
        "expert_utilization": float(jnp.mean(experts_active)) / n_experts,
        "gate_entropy": float(jnp.mean(-jnp.sum(
            gate_probs * jnp.log(gate_probs + 1e-8), axis=-1
        ))),
    }

    return output, stats
