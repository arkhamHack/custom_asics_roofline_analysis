"""
Global-KV MoE Self-Attention
==============================

Routing applies to **queries** (which tokens each expert processes).
Each active expert still builds **global** K and V over the full
sequence, so a routed token can attend to every position — including
tokens handled by other experts.

Conceptual flow::

    Input tokens x [B, H, N, D]
          ↓
    Router (per-token top-k gate)
          ↓
    Token i → experts {e1, e2, …}

    For expert e (active if any token routes to it):
        Q_e = Wq[e] applied only to routed query tokens
        K_e = Wk[e] applied to the full sequence  (global KV)
        V_e = Wv[e] applied to the full sequence  (global KV)
        token i attends over all N keys/values

    Combine expert outputs with gate weights
          ↓
    Residual add → output [B, H, N, D]

Routing loss (Switch Transformer auxiliary, trainable in simulator)::

    L_aux = E · Σ_i  f_i · P_i

    f_i — hard top-k dispatch fraction (stop-gradient)
    P_i — mean softmax gate probability for expert i

    Full training loss:  L = L_task + α · L_aux
    Use ``train_gate_step()`` for a standalone gate-training demo.

Example — token 37 routes to experts {1, 4}:
    - Only experts 1 and 4 project token 37's query.
    - K,V for experts 1 and 4 still contain all N tokens.
    - Token 37 can attend to tokens 92, 15, 7, … via global K/V.

Hardware target (Timeloop fpga_like.yaml):

    DRAM → GlobalBuffer → Cluster[8] → PE[8] → RF → INT16 MAC

Shapes:
    x:              [B, H, N, D]
    Wq,Wk,Wv,Wo:    [E, H, D, D]
    W_gate:         [D, E]
"""
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple


def make_attention_experts(
    n_experts: int,
    n_heads: int,
    head_dim: int,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize per-expert, per-head attention projections and gate."""
    scale = 1.0 / np.sqrt(head_dim)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    shape = (n_experts, n_heads, head_dim, head_dim)
    Wq = jax.random.normal(k1, shape) * scale
    Wk = jax.random.normal(k2, shape) * scale
    Wv = jax.random.normal(k3, shape) * scale
    Wo = jax.random.normal(k4, shape) * scale
    W_gate = jax.random.normal(k5, (head_dim, n_experts)) * scale
    return Wq, Wk, Wv, Wo, W_gate


make_expert_weights = make_attention_experts


@dataclass
class MoeParams:
    """Trainable Global-KV MoE attention parameters."""
    Wq: jnp.ndarray
    Wk: jnp.ndarray
    Wv: jnp.ndarray
    Wo: jnp.ndarray
    W_gate: jnp.ndarray


def make_moe_params(
    n_experts: int,
    n_heads: int,
    head_dim: int,
    key: jax.random.PRNGKey,
) -> MoeParams:
    Wq, Wk, Wv, Wo, W_gate = make_attention_experts(
        n_experts, n_heads, head_dim, key
    )
    return MoeParams(Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, W_gate=W_gate)


def make_training_batch(
    key: jax.random.PRNGKey,
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    noise_std: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Synthetic denoising task: reconstruct clean hidden states from noisy input.

    L_task = MSE(moe_attention(x), target) drives the full attention path.
    """
    k1, k2 = jax.random.split(key)
    target = jax.random.normal(k1, (batch_size, n_heads, seq_len, head_dim))
    x = target + noise_std * jax.random.normal(k2, target.shape)
    return x, target


def moe_dram_traffic(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    n_experts: int,
    tokens_per_expert: List[int],
    dtype_bytes: int = 2,
) -> Dict[str, int]:
    """
    FPGA-style DRAM traffic for Global-KV MoE (scores stay in SRAM).

    Per active expert: stream full K,V (global KV); routed Q only on-chip.
    """
    def _bytes(shape):
        n = 1
        for s in shape:
            n *= s
        return n * dtype_bytes

    x_bytes = _bytes((batch_size, n_heads, seq_len, head_dim))
    active = [e for e, n in enumerate(tokens_per_expert) if n > 0]
    n_active = len(active)

    # Input x + per-active-expert global K,V streams
    qkv = x_bytes + n_active * 2 * _bytes(
        (batch_size, n_heads, seq_len, head_dim)
    )
    # Expert weights loaded from DRAM for active experts
    qkv += n_active * 4 * _bytes((n_heads, head_dim, head_dim))
    scores = 0  # flash-style: no full N×N score spill to DRAM
    output = _bytes((batch_size, n_heads, seq_len, head_dim))
    return {"qkv": qkv, "scores": scores, "output": output}


def _causal_mask(n: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((n, n), dtype=jnp.bool_))


def _global_kv_expert_attention(
    x: jnp.ndarray,
    Wq: jnp.ndarray,
    Wk: jnp.ndarray,
    Wv: jnp.ndarray,
    Wo: jnp.ndarray,
    route_mask: jnp.ndarray,
    scale: float,
    is_causal: bool,
) -> jnp.ndarray:
    """
    Global-KV attention for one expert.

    K and V are projected from the **full** sequence.  Q is projected
    for all positions but only **routed** query rows contribute to the
    output (non-routed rows are zeroed).  Each routed query still
    attends over all N key/value positions.

    Args:
        x:          [B, H, N, D]
        Wq/Wk/Wv:   [H, D, D]
        Wo:         [H, D, D]
        route_mask: [B, N] gate weight for this expert (0 = not routed)
    Returns:
        [B, H, N, D]  non-zero only at routed query positions
    """
    # Global K, V — full sequence, all N tokens
    K = jnp.einsum("bhnd,hde->bhnd", x, Wk)
    V = jnp.einsum("bhnd,hde->bhnd", x, Wv)

    # Q — projected for all positions; only routed rows are kept
    Q = jnp.einsum("bhnd,hde->bhnd", x, Wq)

    scores = jnp.einsum("bhid,bhjd->bhij", Q, K) * scale
    if is_causal:
        mask = _causal_mask(x.shape[2])
        scores = jnp.where(
            mask[None, None, :, :],
            scores,
            jnp.finfo(scores.dtype).min,
        )

    weights = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.einsum("bhij,bhjd->bhid", weights, V)
    out = jnp.einsum("bhid,hde->bhid", attn_out, Wo)

    # Routed-query mask: only tokens assigned to this expert keep output
    return out * route_mask[:, None, :, None]


def _route_tokens(
    x: jnp.ndarray,
    W_gate: jnp.ndarray,
    top_k: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-token top-k expert routing."""
    x_route = x.mean(axis=1)  # [B, N, D]
    gate_logits = jnp.einsum("bnd,de->bne", x_route, W_gate)
    gate_probs = jax.nn.softmax(gate_logits, axis=-1)
    topk_idx = jnp.argsort(gate_probs, axis=-1)[..., -top_k:]
    topk_weights = jnp.take_along_axis(gate_probs, topk_idx, axis=-1)
    topk_weights = topk_weights / jnp.sum(
        topk_weights, axis=-1, keepdims=True
    )
    return gate_probs, topk_idx, topk_weights


def _expert_weights_from_topk(
    topk_idx: jnp.ndarray,
    topk_weights: jnp.ndarray,
    n_experts: int,
) -> jnp.ndarray:
    """Scatter top-k gate weights into [B, N, E]."""
    one_hot = jax.nn.one_hot(topk_idx, n_experts, dtype=topk_weights.dtype)
    return jnp.sum(one_hot * topk_weights[..., None], axis=-2)


def load_balancing_loss(
    gate_probs: jnp.ndarray,
    topk_idx: jnp.ndarray,
    n_experts: int,
) -> jnp.ndarray:
    """
    Switch Transformer auxiliary load-balancing loss (Fedus et al. 2022).

    L_aux = E · Σ_i  f_i · P_i

    f_i — fraction of (token, slot) dispatches to expert i (hard top-k;
          stop-gradient so only router probs receive gradients)
    P_i — mean softmax router probability for expert i over all tokens
          (differentiable w.r.t. W_gate)

    Minimizing this discourages collapse onto a few experts.
    """
    one_hot = jax.nn.one_hot(topk_idx, n_experts, dtype=gate_probs.dtype)
    dispatch_count = jnp.sum(one_hot, axis=(0, 1, 2))          # [E]
    total_dispatches = gate_probs.shape[0] * gate_probs.shape[1] * topk_idx.shape[-1]
    f = dispatch_count / total_dispatches
    f = jax.lax.stop_gradient(f)

    P = jnp.mean(gate_probs, axis=(0, 1))                      # [E]
    return n_experts * jnp.sum(f * P)


def compute_routing_losses(
    gate_probs: jnp.ndarray,
    topk_idx: jnp.ndarray,
    n_experts: int,
    aux_weight: float = 0.01,
) -> dict:
    """Routing loss terms returned alongside hardware stats."""
    aux = load_balancing_loss(gate_probs, topk_idx, n_experts)
    return {
        "load_balancing_loss": aux,
        "total_routing_loss": aux_weight * aux,
        "aux_weight": aux_weight,
    }


def _compute_stats(
    x: jnp.ndarray,
    expert_weights: jnp.ndarray,
    gate_probs: jnp.ndarray,
    topk_idx: jnp.ndarray,
    top_k: int,
    aux_weight: float = 0.01,
    bytes_per_element: int = 2,
) -> dict:
    """
    Hardware stats for Global-KV MoE.

    Per active expert:
      - K, V projections over full sequence (global KV)
      - Q, O projections only for routed query tokens
      - Attention FLOPs: n_routed_queries × N keys
    """
    B, H, N, D = x.shape
    E = expert_weights.shape[-1]
    ew = np.asarray(expert_weights)

    expert_active = np.any(ew > 0, axis=(0, 1))
    n_active = int(np.sum(expert_active))
    tokens_per_expert = np.sum(ew > 0, axis=(0, 1)).astype(int)

    sparse_attn_flops = 0
    for e in range(E):
        if not expert_active[e]:
            continue
        n_routed = int(tokens_per_expert[e])
        sparse_attn_flops += 2 * B * H * N * D * D       # K, V — full seq
        sparse_attn_flops += 2 * n_routed * H * D * D    # Q — routed only
        sparse_attn_flops += n_routed * H * N * D * 4   # attn rows × N keys
        sparse_attn_flops += 2 * n_routed * H * D * D    # O — routed only

    dense_attn_flops = (
        B * H * N * N * D * 4          # one full attention
        + 4 * B * H * N * D * D        # one Q/K/V/O projection set
    )

    compute_fraction = sparse_attn_flops / max(dense_attn_flops, 1)
    sram_reads = int(np.sum(tokens_per_expert) * D * bytes_per_element)

    gate_entropy = float(
        np.mean(
            -np.sum(
                np.asarray(gate_probs) * np.log(np.asarray(gate_probs) + 1e-8),
                axis=-1,
            )
        )
    )

    routing_losses = compute_routing_losses(
        gate_probs, topk_idx, E, aux_weight=aux_weight
    )

    stats = {
        "n_experts": E,
        "top_k": top_k,
        "active_experts": n_active,
        "expert_utilization": n_active / E,
        "tokens_per_expert": tokens_per_expert.tolist(),
        "routed_queries": int(np.sum(tokens_per_expert)),
        "dense_attn_flops": dense_attn_flops,
        "sparse_attn_flops": sparse_attn_flops,
        "compute_fraction": compute_fraction,
        "compute_saved": 1.0 - compute_fraction,
        "compute_saved_ratio": 1.0 - compute_fraction,
        "estimated_sram_reads": sram_reads,
        "gate_entropy": gate_entropy,
        "load_balancing_loss": float(routing_losses["load_balancing_loss"]),
        "total_routing_loss": float(routing_losses["total_routing_loss"]),
        "aux_weight": aux_weight,
    }
    return stats


@partial(jax.jit, static_argnames=["top_k", "is_causal"])
def _moe_attention_impl(
    x: jnp.ndarray,
    Wq: jnp.ndarray,
    Wk: jnp.ndarray,
    Wv: jnp.ndarray,
    Wo: jnp.ndarray,
    W_gate: jnp.ndarray,
    top_k: int,
    is_causal: bool,
    scale: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled Global-KV MoE attention core."""
    E = Wq.shape[0]

    gate_probs, topk_idx, topk_weights = _route_tokens(x, W_gate, top_k)
    expert_weights = _expert_weights_from_topk(topk_idx, topk_weights, E)

    def per_expert(e_idx):
        return _global_kv_expert_attention(
            x,
            Wq[e_idx],
            Wk[e_idx],
            Wv[e_idx],
            Wo[e_idx],
            expert_weights[:, :, e_idx],
            scale,
            is_causal,
        )

    out_all = jax.vmap(per_expert)(jnp.arange(E))       # [E, B, H, N, D]
    out_all = jnp.transpose(out_all, (1, 0, 2, 3, 4))   # [B, E, H, N, D]

    # Gate-weighted combine across experts for each token
    output = jnp.einsum("bne,behnd->bhnd", expert_weights, out_all)
    output = output + x
    return output, expert_weights, gate_probs, topk_idx


@partial(jax.jit, static_argnames=["top_k", "is_causal", "aux_weight"])
def train_step(
    params: MoeParams,
    x: jnp.ndarray,
    target: jnp.ndarray,
    top_k: int,
    is_causal: bool,
    aux_weight: float,
    lr: float,
) -> Tuple[MoeParams, Dict[str, jnp.ndarray]]:
    """
    One end-to-end SGD step: L = L_task + α · L_aux.

    Gradients flow through the full Global-KV attention path (Wq/Wk/Wv/Wo)
    and the differentiable gate probabilities (W_gate).
    """
    scale = 1.0 / jnp.sqrt(x.shape[-1])
    n_experts = params.Wq.shape[0]

    def loss_fn(p: MoeParams):
        out, _, gate_probs, topk_idx = _moe_attention_impl(
            x, p.Wq, p.Wk, p.Wv, p.Wo, p.W_gate,
            top_k, is_causal, scale,
        )
        task_loss = jnp.mean((out - target) ** 2)
        aux_loss = load_balancing_loss(gate_probs, topk_idx, n_experts)
        total = task_loss + aux_weight * aux_loss
        return total, (task_loss, aux_loss)

    (total, (task_loss, aux_loss)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    metrics = {
        "total_loss": total,
        "task_loss": task_loss,
        "aux_loss": aux_loss,
        "load_balancing_loss": aux_loss,
    }
    return new_params, metrics


def train_moe(
    params: MoeParams,
    key: jax.random.PRNGKey,
    batch_size: int = 4,
    n_heads: int = 8,
    seq_len: int = 128,
    head_dim: int = 64,
    top_k: int = 2,
    n_steps: int = 100,
    lr: float = 1e-3,
    aux_weight: float = 0.01,
    is_causal: bool = True,
    log_every: int = 10,
) -> Tuple[MoeParams, List[Dict[str, float]]]:
    """End-to-end MoE training loop (synthetic denoising task)."""
    history: List[Dict[str, float]] = []
    for step in range(n_steps):
        key, k_batch = jax.random.split(key)
        x, target = make_training_batch(
            k_batch, batch_size, n_heads, seq_len, head_dim
        )
        params, metrics = train_step(
            params, x, target, top_k, is_causal, aux_weight, lr
        )
        record = {k: float(v) for k, v in metrics.items()}
        record["step"] = step + 1
        history.append(record)
        if log_every and (step + 1) % log_every == 0:
            print(
                f"  [train] step {step + 1:>4}/{n_steps}  "
                f"L={record['total_loss']:.5f}  "
                f"task={record['task_loss']:.5f}  "
                f"L_aux={record['aux_loss']:.4f}"
            )
    return params, history


def train_gate_step(
    x: jnp.ndarray,
    W_gate: jnp.ndarray,
    top_k: int,
    aux_weight: float = 0.01,
    lr: float = 1e-2,
) -> Tuple[jnp.ndarray, dict]:
    """One SGD step on W_gate only (aux loss). Kept for quick routing demos."""
    def loss_fn(Wg):
        gate_probs, topk_idx, _ = _route_tokens(x, Wg, top_k)
        return aux_weight * load_balancing_loss(
            gate_probs, topk_idx, Wg.shape[1]
        )

    loss, grad = jax.value_and_grad(loss_fn)(W_gate)
    W_gate = W_gate - lr * grad
    return W_gate, {
        "load_balancing_loss": float(loss / aux_weight),
        "total_routing_loss": float(loss),
        "aux_weight": aux_weight,
        "gate_grad_norm": float(jnp.linalg.norm(grad)),
    }


def moe_attention(
    x: jnp.ndarray,
    Wq: jnp.ndarray,
    Wk: jnp.ndarray,
    Wv: jnp.ndarray,
    Wo: jnp.ndarray,
    W_gate: jnp.ndarray,
    top_k: int = 2,
    is_causal: bool = True,
    scale: Optional[float] = None,
    aux_weight: float = 0.01,
) -> Tuple[jnp.ndarray, dict]:
    """
    Global-KV MoE multi-head self-attention.

    Routing selects which experts process each token's **query**.
    Each expert's **K** and **V** cover the full sequence, preserving
    global self-attention semantics while sparsifying query-side compute.

    Args:
        x:          [B, H, N, D]
        Wq/Wk/Wv/Wo:[E, H, D, D]
        W_gate:     [D, E]
        top_k:      experts per token
        is_causal:  decoder causal mask
        scale:      attention scale (default 1/sqrt(D))
        aux_weight: coefficient on load-balancing loss in stats / training

    Returns:
        output: [B, H, N, D]
        stats:  routing / hardware / loss metrics
    """
    if scale is None:
        scale = 1.0 / np.sqrt(x.shape[-1])

    output, expert_weights, gate_probs, topk_idx = _moe_attention_impl(
        x, Wq, Wk, Wv, Wo, W_gate, top_k, is_causal, scale
    )
    stats = _compute_stats(
        x, expert_weights, gate_probs, topk_idx, top_k, aux_weight=aux_weight
    )
    return output, stats


def moe_attention_from_params(
    x: jnp.ndarray,
    params: MoeParams,
    top_k: int = 2,
    is_causal: bool = True,
    scale: Optional[float] = None,
    aux_weight: float = 0.01,
) -> Tuple[jnp.ndarray, dict]:
    """Forward pass using an ``MoeParams`` bundle (trained or init)."""
    return moe_attention(
        x, params.Wq, params.Wk, params.Wv, params.Wo, params.W_gate,
        top_k=top_k, is_causal=is_causal, scale=scale, aux_weight=aux_weight,
    )


moe_self_attention = moe_attention


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B, H, N, D, E = 4, 8, 128, 64, 8

    x = jax.random.normal(key, (B, H, N, D))
    params = make_moe_params(n_experts=E, n_heads=H, head_dim=D, key=key)

    print("End-to-end MoE training (L_task + L_aux) …")
    params, history = train_moe(
        params, key, batch_size=B, n_heads=H, seq_len=N, head_dim=D,
        top_k=2, n_steps=50, lr=1e-3, log_every=10,
    )

    y, stats = moe_attention_from_params(x, params, top_k=2, is_causal=True)
    print("\nTrained model hardware stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
