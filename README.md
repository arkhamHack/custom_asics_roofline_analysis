## Overview

This project explores a simple question:

> When an attention optimization reduces theoretical work, does the hardware actually run faster?

We implement and compare several modern attention mechanisms in JAX:

* FlashAttention
* Grouped-Query Attention (GQA)
* Quantized Attention (INT8)
* Windowed Local-KV Mixture-of-Experts (MoE)
* Paged Attention

Each kernel is analyzed from two perspectives:

1. **Compiler-level analysis** using JAX → StableHLO inspection to extract GEMMs, FLOPs, memory traffic, and arithmetic intensity.
2. **Hardware-level analysis** using Timeloop/Accelergy simulation on a custom FPGA-like accelerator (200 GFLOP/s, 25.6 GB/s LPDDR4).

The goal is to bridge the gap between:

```text
Algorithmic savings
        ↓
Compiler-generated kernels
        ↓
Memory hierarchy behavior
        ↓
Actual hardware speedup
```

## Key Findings

### FlashAttention

FlashAttention reduces DRAM traffic by keeping attention scores on-chip instead of materializing the full `N × N` score matrix. FLOPs remain largely unchanged, but arithmetic intensity increases significantly.

### Grouped-Query Attention (GQA)

GQA shares KV heads across multiple query heads, reducing KV-cache bandwidth during inference. Compute remains similar to standard multi-head attention, while memory traffic scales down with the number of KV heads.

### Windowed Local-KV MoE Attention

Queries are routed to experts and attend only to a fixed local KV window. This changes attention complexity from:

```math
O(N²D)
```

to approximately:

```math
O(N · k · W · D)
```

where:

* `k` = active experts per token
* `W` = local attention window

The implementation includes Switch Transformer–style load-balancing loss to prevent expert collapse and improve hardware utilization.

### Quantized Attention

INT8 quantization reduces memory traffic by up to 4× compared to FP32 and increases arithmetic intensity, moving workloads closer to the compute-bound region of the roofline model.

## Hardware Model

The accelerator consists of:

* LPDDR4 DRAM (25.6 GB/s)
* 64 KB Global SRAM Buffer
* 8 Clusters × 8 Processing Elements
* 64 MAC lanes at 500 MHz

Energy and performance are evaluated using Timeloop and Accelergy.

## Roofline Analysis

Using GEMMs extracted from StableHLO, the project computes:

* FLOPs
* Memory Traffic
* Arithmetic Intensity
* Roofline Placement

This enables direct comparison between theoretical algorithmic savings and actual hardware utilization.

## Run

```bash
python3 run_experiment.py
python3 run_experiment.py --moe-train
```

## Outputs

Generated visualizations include:

* Roofline Analysis
* DRAM Traffic
* Latency Scaling
* KV Cache Usage
* Expert Routing Balance
* Compute Utilization

## Takeaway

Reducing FLOPs alone does not guarantee hardware speedup.

Whether an optimization translates into real performance gains depends on:

* The GEMMs actually emitted by the compiler
* Memory hierarchy behavior
* Arithmetic intensity
* Processing-element utilization
* Load balancing across experts

This project explores the boundary between algorithm design, compiler-generated kernels, and accelerator performance.
