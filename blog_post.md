# When Algorithmic Sparsity Becomes Hardware Speedup

Most posts about attention optimization stop at FLOPs. This one doesn't. We implement FlashAttention, Grouped-Query Attention, and windowed local-KV MoE routing in JAX, simulate each on a 200 GFLOP/s, 25.6 GB/s FPGA-like memory hierarchy using Timeloop, and ask a harder question: when an algorithm skips work, does the hardware actually run faster?

The answer depends on *where* the skipped work was happening — compute, SRAM, or DRAM — and whether your memory hierarchy can exploit the savings. The roofline model is the tool that makes this concrete.

**Run it yourself:**
```bash
python3 run_experiment.py              # Timeloop hardware sim
python3 run_experiment.py --moe-train  # train MoE routing, then FPGA eval
```

---

## 1. FlashAttention: The Win Is in the Buffer, Not the FLOP Count

Standard attention computes $QK^T$ and writes the full $N \times N$ score matrix to DRAM before the softmax pass reads it back. At $N = 512$, that's 512 × 512 × 2 bytes = 512 KB of round-trip DRAM traffic just for scores — before you touch values at all.

FlashAttention doesn't reduce that multiply-accumulate count. It reorders the computation into 64×64 tiles that fit in on-chip SRAM, fusing QK and AV into a single pass with a running-max softmax correction. The score tensor never materializes in DRAM.

**JAX component:** `python/attention/flash.py` expresses the kernel as nested `lax.scan` loops over query blocks and key/value blocks. Inside each tile it computes a local score matrix, updates the online softmax statistics, and accumulates the output tile. This is still readable Python-like tensor code, but JAX traces the whole thing into compiler IR so we can inspect the `dot_general`, masks, reductions, and loop structure.

**Timeloop component:** the Timeloop mapping mimics the FlashAttention memory policy rather than trying to auto-derive it from HLO. In `workspace/configs/mappings/flash_qk.yaml`, the score tensor `Z` is kept below DRAM:

```yaml
- target: GlobalBuffer
  type: datatype
  keep:   [A, B, Z]   # scores (Z) stay in GlobalBuffer
  bypass: []
```

Compared to the naive mapping where `Z` is bypassed *out* of GlobalBuffer to DRAM and re-read, this single policy change is what drives the latency difference. At $N = 512$, naive QK traffic is dominated by the score round-trip; flash reduces DRAM reads by roughly 3–4× on this workload.

**What the plots tell us:** `dram_traffic.html` should show Flash with little or no score spill traffic, not flat total DRAM. Its total bar still grows with sequence length because Q, K, V, and output still touch DRAM. In the stacked plot, the Flash segments should be QKV reads plus output write; the missing component is the $N^2$ score round-trip. `latency_scaling.html` shows the compute cost still rising with $N^2$ for full attention, which is why Flash is memory-efficient but not compute-sparse.

---

## 2. Grouped-Query Attention: Inference Is a Different Problem

Training attention is compute-bound on large batches. Inference is different: you're decoding one token at a time, and the KV cache — which grows linearly with sequence length — becomes your bandwidth bottleneck.

Standard multi-head attention (MHA) stores one K and V tensor per head. With 8 attention heads at $d = 64$, a single sequence of length $N$ requires $2 \times 8 \times N \times 64 \times 2$ bytes of KV cache. At $N = 4096$, that's 8 MB per sequence.

Grouped-Query Attention (GQA) assigns multiple Q heads to each KV head. With 8 Q heads and 2 KV heads (our `8Q/2KV` configuration), you keep only 2 K tensors and 2 V tensors, reducing KV bandwidth by 4×. The Q projections are unchanged — all 8 heads still compute full attention, just over shared keys.

The FLOPs are nearly identical. The win is purely in DRAM reads on the decode path:

| Config | KV heads | KV cache bandwidth | Relative cost |
|--------|----------|-------------------|---------------|
| 8Q/8KV (MHA) | 8 | 100% | 1× |
| 8Q/4KV | 4 | 50% | 0.5× |
| 8Q/2KV | 2 | 25% | 0.25× |
| 8Q/1KV (MQA) | 1 | 12.5% | 0.125× |

**JAX component:** `python/attention/gqa.py` takes Q with 8 query heads and K/V with fewer KV heads. In the reference implementation, JAX repeats each KV head across a group of query heads so the output shape stays `[B, 8, N, D]`. The stats dictionary reports the KV-cache savings directly: 8Q/2KV means a 4× KV bandwidth reduction in decode.

**Timeloop component:** the hardware model has to be careful here. The QK matmul shape per query head is still full attention, so GQA does not automatically reduce QK MACs. What changes is the memory accounting: Q and output scale with 8 query heads, but K and V scale with only 2 KV heads. The GQA mappings now also bypass score traffic like FlashAttention, so the Timeloop model represents “shared K/V, scores on-chip,” not “full MHA with a GQA label.”

**What to look for in `kv_cache.html`:** KV memory cost drops linearly with group size while compute cost is flat. At sequence lengths above 1024, the memory savings become the dominant factor in latency.

**What the plots tell us:** `dram_traffic.html` should show GQA below Flash for DRAM once K/V sharing is accounted for, while `latency_scaling.html` stays close to Flash because both still do full QK and AV work for every query head. That is the important lesson: GQA is a serving-memory optimization first, not a per-head compute reduction.

---

## 3. MoE Routing: Sparse Queries, Local Keys

Most MoE coverage focuses on sparsifying the FFN layer — each token routes to 1 of $E$ expert feed-forward networks, saving $(1 - k/E)$ of FFN compute. We do something different: we route attention queries to experts and pair that with a fixed local K/V window.

The setup: 8 experts, each with its own $W_Q$, $W_K$, $W_V$, and $W_O$. A learned gating network routes each token to `top_k=2` experts. For each active expert, only the routed query tokens are processed, and each routed query attends to a local window of size `W=128` rather than the full sequence. That changes the expert GEMM from sparse-M × full-N to sparse-M × local-W.

```
Token 37 → experts {1, 4}
  Expert 1 QK: Q[37] × K[37-64 ... 37+63]
  Expert 4 QK: Q[37] × K[37-64 ... 37+63]
  combine expert outputs with gate weights
```

This is intentionally local-KV attention: the router chooses which expert subspaces process each token, while the window bounds how many keys and values each routed query can touch. The payoff is that the attention part scales like $O(N \cdot top_k \cdot W \cdot D)$ instead of $O(N^2 \cdot D)$ when `W` is held constant.

**Training:** the gating network is trained end-to-end with a Switch Transformer load-balancing auxiliary loss:

$$\mathcal{L} = \mathcal{L}_\text{task} + \alpha \cdot \sum_e f_e \cdot p_e$$

where $f_e$ is the fraction of tokens dispatched to expert $e$ and $p_e$ is the router's mean probability for that expert. This penalizes routing collapse (all tokens to one expert) without hard constraints.

**JAX component:** `python/attention/moe_attention.py` keeps the routing trainable. JAX is useful here because the gate, expert projections, auxiliary load-balancing loss, and denoising loss are all differentiable. The implementation uses masks and vectorized expert calls to simulate which tokens route to which experts, while the stats track expert utilization and gate entropy.

**Timeloop component:** the hardware model does not issue a sparse matrix. It emits a dense GEMM for the work that actually fires: `[M_e, W, D]`, where `M_e` is the routed-query count for an expert and `W=min(128, N)` is the local KV window. The mapping generator in `python/hardware.py` creates MoE problem shapes with `N = W`, not `N = seq_len`, so Timeloop sees the linear-scaling local-KV workload.

**What the plots tell us:** `compute_utilization.html` shows what fraction of dense attention/FFN-style work actually fires. `expert_routing.html` shows whether the gate is balanced or collapsed. `latency_scaling.html` is the hardware payoff: once the window stops growing, MoE grows much more slowly with sequence length than full attention. If the expert routing is imbalanced, the theoretical sparsity does not fully become speedup.

---

## 4. The Hardware Target: Every Buffer Decision Is Explicit

Our target (`workspace/configs/arch/fpga_like.yaml`) is an FPGA-like accelerator model with 45nm energy numbers:

- **DRAM:** LPDDR4, 12.8 GB/s read + 12.8 GB/s write = 25.6 GB/s aggregate
- **GlobalBuffer:** 64 KB on-chip SRAM (holds ~64×64×3 tiles of 16-bit elements)
- **Spatial hierarchy:** 8 Clusters × 8 PEs = 64 MAC lanes
- **Clock:** 500 MHz

The spatial mapping follows a two-level fanout:

```
DRAM
 └─ GlobalBuffer (1×)      ← cluster spatial fanout here (M-dim, 8 clusters)
     └─ ClusterBuffer (8×) ← PE spatial fanout here (N-dim, 8 PEs per cluster)
         └─ RegisterFile (64×, one per PE)
             └─ MAC (64×)
```

This matters for Timeloop mappings. Spatial tiling must align with the instance count at the *parent* level — a `spatial` directive at `GlobalBuffer` fans out into 8 Cluster instances, and a `spatial` directive at `ClusterBuffer` fans out into 8 PE instances. RegisterFile has 1 instance per PE, so no spatial fanout is possible there.

The design is "FPGA-like" because of the small SRAM hierarchy, LPDDR-style bandwidth, explicit spatial clusters, and modest 64-lane PE array. The `45nm` label is not meant to claim a specific commercial FPGA process node; it is the technology point used by Timeloop/Accelergy-style component tables so SRAM, register-file, MAC, and DRAM accesses have consistent energy estimates.

**Energy:** Timeloop uses a pre-built Energy Reference Table with 45nm values: MAC at 0.298 pJ, RegisterFile reads at 0.146 pJ, ClusterBuffer at 2.464 pJ/access, GlobalBuffer at 17.872 pJ/access, DRAM at 512 pJ/access. The 3-order-of-magnitude gap between MAC and DRAM energy is why memory policy dominates total energy — reducing DRAM accesses matters far more than reducing FLOPs.

**How we mimic attention kernels in Timeloop:** JAX verifies the algorithm and exposes compiler IR, but Timeloop needs explicit GEMM problems and mappings. QK is modeled as `A[M,K] × B[N,K] → Z[M,N]`; AV is modeled as `A[M,K] × B[K,N] → Z[M,N]`. Each variant changes those dimensions and datatype bypass rules: Flash changes where scores live, GQA changes how K/V traffic is counted, sliding window changes `N` into a local window, and MoE changes `M` into routed queries while keeping `N=W` for local KV.

---

## 5. Roofline: When Does Sparsity Cross the Ridge?

The roofline model places each workload on a two-axis plot: arithmetic intensity (FLOPs/byte) on the x-axis and achieved throughput (GFLOP/s) on the y-axis. The ridge point is where the memory bandwidth ceiling meets the compute ceiling:

$$\text{Ridge} = \frac{\text{Peak compute}}{\text{Peak bandwidth}} = \frac{200 \text{ GFLOP/s}}{25.6 \text{ GB/s}} \approx 7.8 \text{ FLOP/byte}$$

Workloads left of the ridge are memory-bound; their achieved GFLOP/s scales with bandwidth, not compute. Right of the ridge, they're compute-bound.

**In the current Timeloop run, most points sit to the right of the ridge.** That means the 64-lane array is often compute-limited for these GEMM shapes after score traffic is kept on-chip. This does not make DRAM irrelevant: DRAM traffic still changes latency through the pipeline-stall term and dominates energy per access. It means that once the mapping avoids pathological score round-trips, reducing bytes alone is not enough — the variant also has to reduce issued compute or improve PE utilization.

The roofline explains why each variant wins differently:

| Variant | How it shifts the roofline point |
|---------|----------------------------------|
| **Flash** | Same FLOPs, fewer DRAM bytes → moves *right* along x-axis |
| **GQA** | Fewer DRAM KV reads → moves *right* |
| **Sliding window** | Fewer FLOPs *and* fewer bytes → smaller point, same regime |
| **MoE** | Fewer routed Q rows and fixed local window → smaller GEMM |
| **Quantized** | Halves bytes per element → 2× intensity, moves *right* |

Flash and GQA win mainly by *raising arithmetic intensity*, not by reducing QK FLOPs. Sliding window and local-KV MoE reduce the actual GEMM shape, so they move differently: less total work is issued to the array.

**The uncomfortable truth about MoE:** routing fewer tokens is not automatically a perfect speedup. The gap between theoretical sparsity and hardware time comes from three sources:

1. Local K/V windows still load real K and V tiles for every routed query
2. Smaller-M GEMMs can underfill a 64-lane PE array when routing gets too sparse
3. Load imbalance concentrates work in a few experts unless the auxiliary loss keeps routing balanced

The `--moe-train` experiment makes this visible — FPGA latency improves as the router learns balanced dispatch, but the final speedup depends on both the routing distribution and the fixed local-window shape.

**What to look for in `roofline.html`:** Flash and GQA sit near each other in throughput because they issue similar full-attention compute, but GQA moves right because it moves fewer K/V bytes. Sliding window and MoE can sit farther right because they issue less attention work per token. Quantized should move right because INT8 halves bytes per element while now using the same 64-lane topology as the FP16 model.

This is why algorithmic sparsity and hardware speedup aren't the same thing. The right question is not just "how many FLOPs did we skip?" It is "which GEMM shape did we actually issue, how many DRAM bytes did we avoid, and does the mapping keep the PE array busy?" The roofline is the bridge between those questions.

---

## 6. What Each Plot Is Supposed to Prove

**`dram_traffic.html`** is the memory-policy plot. Flash should have little or no score spill traffic because scores stay on chip, but its total DRAM still increases with `N` because Q, K, V, and output tensors get larger. GQA should reduce K/V-related traffic because 2 KV heads are shared across 8 query heads. Sliding-window and MoE reduce traffic by shrinking the effective key dimension.

**`latency_scaling.html`** is the "did it actually get faster?" plot. Flash and GQA can remain close because they still issue dense QK/AV compute. Sliding-window and local-KV MoE should separate more clearly at larger `N` because their Timeloop GEMMs are smaller.

**`roofline.html`** tells whether a point is limited more by bytes or by MAC throughput. Moving right means higher arithmetic intensity. Moving up means better achieved throughput. If a point ever has high intensity but very low throughput, that usually points to a hardware/mapping issue rather than an algorithmic win.

**`kv_cache.html`** is where GQA and paged attention make the most sense. GQA reduces KV cache size by reducing KV heads; paging reduces wasted allocation and fragmentation.

**`expert_routing.html`** and **`compute_utilization.html`** explain whether MoE routing is balanced enough to turn sparse routing into hardware speedup.

---

## Limitations

- Timeloop mappings are hand-crafted; auto-derivation from JAX/XLA traces remains future work
- Single architecture config — sensitivity to GlobalBuffer size and bandwidth ratio not explored
- Attention layer only; no FFN or residual connections in the hardware model
- MoE Timeloop model uses an expected routed-query count per expert and a fixed local KV window; `--moe-train` adds learned routing on the analytical/JAX path, but Timeloop does not replay every token-level route
- 16-bit data throughout for the main FPGA config; the quantized variant uses a separate INT8 architecture with the same 8-cluster × 8-PE topology for comparability

---

## Reproduce

```bash
python3 -m pip install -r python/requirements.txt
python3 run_experiment.py              # requires Docker
python3 run_experiment.py --moe-train
```

Outputs: `workspace/outputs/plots/*.html`  
Docker image: `timeloopaccelergy/timeloop-accelergy-pytorch:latest-arm64`
