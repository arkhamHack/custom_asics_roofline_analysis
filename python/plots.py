"""Plotly visualisation suite for FlashAccel experiments."""
from typing import List, Optional, Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_COLORS = {
    "naive":        "#E74C3C",
    "flash":        "#2ECC71",
    "quantized":    "#3498DB",
    "dsa":         "#9B59B6",
    "sliding_window": "#F39C12",
    "paged":       "#1ABC9C",
    "gqa":         "#E67E22",
    "moe":         "#8E44AD",
}
_VARIANT_ORDER = ["naive", "flash", "quantized", "dsa", "sliding_window", "paged", "gqa", "moe"]



def roofline_plot(
    benchmark_results,
    hw_peak_flops_gflops: float = 200.0,    # FPGA-like peak (adjust to your target)
    hw_peak_bw_gb_s: float = 25.6,          # matches fpga_like.yaml DRAM bandwidth
    seq_lens_to_label: tuple = (128, 512, 1024),
) -> go.Figure:
    """
    Classic roofline chart.
    X-axis: arithmetic intensity (FLOP / DRAM byte)
    Y-axis: performance (GFLOP/s)
    """
    ai_range = np.logspace(-2, 3, 500)
    memory_roof = hw_peak_bw_gb_s * ai_range
    compute_roof = np.full_like(ai_range, hw_peak_flops_gflops)
    roofline = np.minimum(memory_roof, compute_roof)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ai_range, y=roofline,
        mode="lines", name="Roofline",
        line=dict(color="black", width=2, dash="dash"),
    ))
    fig.add_hline(
        y=hw_peak_flops_gflops, line_dash="dot", line_color="grey",
        annotation_text=f"Compute peak {hw_peak_flops_gflops} GFLOP/s",
        annotation_position="top left",
    )

    grouped: Dict[str, list] = {v: [] for v in _VARIANT_ORDER}
    for r in benchmark_results:
        if r.name in grouped and r.seq_len in seq_lens_to_label:
            grouped[r.name].append(r)

    for vname in _VARIANT_ORDER:
        pts = grouped[vname]
        if not pts:
            continue
        xs = [r.arithmetic_intensity for r in pts]
        ys = [r.throughput_gflops for r in pts]
        labels = [f"N={r.seq_len}" for r in pts]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            name=vname,
            text=labels,
            textposition="top center",
            marker=dict(color=_COLORS[vname], size=12, symbol="circle"),
        ))

    fig.update_layout(
        title="Roofline: Attention Variants on FPGA-like Hardware",
        xaxis=dict(title="Arithmetic Intensity (FLOP / DRAM byte)", type="log"),
        yaxis=dict(title="Performance (GFLOP/s)", type="log"),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        width=800, height=500,
    )

    ridge_point = hw_peak_flops_gflops / hw_peak_bw_gb_s
    fig.add_vline(
        x=ridge_point, line_dash="dash", line_color="orange",
        annotation_text=f"Ridge ({ridge_point:.1f} F/B)",
        annotation_position="top right",
    )
    fig.add_annotation(
        x=np.log10(ridge_point / 4), y=np.log10(hw_peak_flops_gflops * 0.7),
        text="← Memory Bound", showarrow=False,
        font=dict(size=12, color="blue"),
        xref="x", yref="y",
    )
    fig.add_annotation(
        x=np.log10(ridge_point * 4), y=np.log10(hw_peak_flops_gflops * 0.7),
        text="Compute Bound →", showarrow=False,
        font=dict(size=12, color="red"),
        xref="x", yref="y",
    )

    return fig



def dram_traffic_plot(benchmark_results) -> go.Figure:
    """
    Grouped + stacked bar chart showing DRAM traffic breakdown
    (QKV reads, scores r/w, output write) per variant at each sequence length.
    """
    seq_lens = sorted({r.seq_len for r in benchmark_results})
    fig = make_subplots(
        rows=1, cols=len(seq_lens),
        subplot_titles=[f"N = {n}" for n in seq_lens],
        shared_yaxes=True,
    )

    component_labels = ["QKV reads", "Scores traffic", "Output write"]

    for col_idx, N in enumerate(seq_lens, start=1):
        for vname in _VARIANT_ORDER:
            match = [r for r in benchmark_results if r.name == vname and r.seq_len == N]
            if not match:
                continue
            r = match[0]
            traffic_mb = [
                r.dram_qkv_bytes / 1e6,
                r.dram_scores_bytes / 1e6,
                r.dram_output_bytes / 1e6,
            ]
            for i, (val, comp) in enumerate(zip(traffic_mb, component_labels)):
                fig.add_trace(
                    go.Bar(
                        name=f"{vname} — {comp}",
                        x=[vname],
                        y=[val],
                        marker_color=_COLORS[vname],
                        opacity=0.4 + 0.3 * i,
                        legendgroup=comp,
                        showlegend=(col_idx == 1),
                    ),
                    row=1, col=col_idx,
                )

    fig.update_layout(
        barmode="stack",
        title="DRAM Traffic Breakdown by Attention Variant",
        yaxis_title="DRAM traffic (MB)",
        template="plotly_white",
        width=900, height=450,
    )
    return fig



def latency_scaling_plot(benchmark_results) -> go.Figure:
    fig = go.Figure()

    grouped: Dict[str, list] = {v: [] for v in _VARIANT_ORDER}
    for r in benchmark_results:
        if r.name in grouped:
            grouped[r.name].append(r)

    for vname in _VARIANT_ORDER:
        pts = sorted(grouped[vname], key=lambda r: r.seq_len)
        if not pts:
            continue
        color = _COLORS.get(vname, "#888")
        fig.add_trace(go.Scatter(
            x=[r.seq_len for r in pts],
            y=[r.fpga_latency_ms() for r in pts],
            mode="lines+markers",
            name=vname,
            line=dict(color=color, width=2),
            marker=dict(size=8, symbol="diamond"),
        ))

    fig.update_layout(
        title="Simulated FPGA Latency vs Sequence Length (200 GFLOP/s, 25.6 GB/s)",
        xaxis_title="Sequence Length (N)",
        yaxis_title="Latency (ms)",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        width=700, height=450,
    )
    return fig



def timeloop_energy_plot(hw_stats: Dict[str, Dict]) -> go.Figure:
    names  = list(hw_stats.keys())
    energy = [hw_stats[n].get("energy_uj") or 0 for n in names]
    dram   = [(hw_stats[n].get("dram_total_bytes") or 0) / 1e6 for n in names]
    latency = [hw_stats[n].get("latency_ms") for n in names]
    has_latency = any(v is not None for v in latency)

    if has_latency:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Energy (uJ) + DRAM Traffic",
                "FPGA Latency from Timeloop Cycles",
            ],
        )
        fig.add_trace(
            go.Bar(x=names, y=energy, name="Energy (uJ)",
                   marker_color=[_COLORS.get(n.split("_")[0], "#888") for n in names]),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=names, y=dram, name="DRAM (MB)",
                       mode="lines+markers",
                       line=dict(color="black", width=2),
                       marker=dict(size=10)),
            row=1, col=1,
        )
        lat_ms = [v if v is not None else 0 for v in latency]
        fig.add_trace(
            go.Bar(x=names, y=lat_ms, name="Latency (ms)",
                   marker_color=[_COLORS.get(n.split("_")[0], "#888") for n in names],
                   opacity=0.8),
            row=1, col=2,
        )
        fig.update_yaxes(title_text="Energy (uJ) / DRAM (MB)", row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms) @ 500 MHz", row=1, col=2)
        fig.update_layout(
            title="Timeloop/Accelergy Results: Energy, DRAM Traffic & Latency",
            template="plotly_white",
            width=1100, height=450,
        )
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=names, y=energy, name="Energy (uJ)",
                   marker_color=[_COLORS.get(n.split("_")[0], "#888") for n in names]),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=names, y=dram, name="DRAM traffic (MB)",
                       mode="lines+markers",
                       line=dict(color="black", width=2),
                       marker=dict(size=10)),
            secondary_y=True,
        )
        fig.update_layout(
            title="Timeloop/Accelergy Results: Energy and DRAM Traffic",
            template="plotly_white",
            width=750, height=450,
        )
        fig.update_yaxes(title_text="Energy (uJ)", secondary_y=False)
        fig.update_yaxes(title_text="DRAM traffic (MB)", secondary_y=True)
    return fig


# ── Convenience: generate all plots ──────────────────────────────────────────

def compute_utilization_plot(cc_results: Dict) -> go.Figure:
    from python.sparsity import ComputeStats

    savings = cc_results["savings"]
    names = [s.name for s in savings]
    active_pct = [(s.active_flops / s.total_flops * 100) if s.total_flops > 0 else 100 for s in savings]
    skipped_pct = [100 - a for a in active_pct]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=active_pct,
        name="Active Compute",
        marker_color="#2ECC71",
    ))
    fig.add_trace(go.Bar(
        x=names, y=skipped_pct,
        name="Skipped Compute",
        marker_color="#95A5A6",
    ))
    fig.update_layout(
        barmode="stack",
        title="Compute Utilization: What Actually Fires",
        yaxis_title="% of Dense Compute",
        template="plotly_white",
        width=800, height=450,
    )
    return fig


def expert_routing_plot(cc_results: Dict) -> go.Figure:
    moe_sweep = cc_results.get("moe_sweep", [])
    if not moe_sweep:
        fig = go.Figure()
        fig.add_annotation(text="No MoE data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Create a matrix: rows = expert configs, cols = metrics
    configs = [f"{m['n_experts']}e top-{m['top_k']}" for m in moe_sweep]
    labels = ["Compute Saved %", "Expert Util. %", "Gate Entropy"]

    z = []
    for m in moe_sweep:
        row = [
            m["compute_saved_ratio"] * 100,
            m["expert_utilization"] * 100,
            m["gate_entropy"] * 25,
        ]
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=configs,
        colorscale="Viridis",
    ))
    fig.update_layout(
        title="MoE Expert Routing: Compute Saved / Expert Utilization / Gate Entropy",
        template="plotly_white",
        width=700, height=400,
    )
    return fig


def kv_cache_plot(cc_results: Dict) -> go.Figure:
    fig = go.Figure()

    gqa_sweep = cc_results.get("gqa_sweep", [])
    if gqa_sweep:
        configs = [f"{g['n_heads']}Q/{g['n_kv_heads']}KV" for g in gqa_sweep]
        savings = [g["kv_memory_saved_ratio"] * 100 for g in gqa_sweep]
        fig.add_trace(go.Bar(
            x=configs, y=savings,
            name="GQA KV Memory Saved %",
            marker_color="#E67E22",
        ))

    paged = cc_results.get("paged", {})
    if paged:
        fig.add_trace(go.Bar(
            x=[f"Paged (pg={paged['page_size']})"],
            y=[paged["memory_saved_vs_contiguous"] * 100],
            name="Paged Memory Saved %",
            marker_color="#1ABC9C",
        ))

    fig.update_layout(
        title="KV Cache Memory Savings",
        yaxis_title="Memory Saved (%)",
        template="plotly_white",
        width=700, height=400,
    )
    return fig


def savings_dashboard(cc_results: Dict) -> go.Figure:
    """
    Combined dashboard: compute saved, memory saved, and bandwidth reduction.
    """
    savings = cc_results["savings"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Compute Saved (%)", "Memory Saved (%)"],
    )

    names = [s.name for s in savings]
    compute_saved = [s.compute_saved_ratio * 100 for s in savings]
    memory_saved = [s.memory_saved_ratio * 100 for s in savings]

    colors = ["#2ECC71" if c > 0 else "#E74C3C" for c in compute_saved]

    fig.add_trace(go.Bar(
        x=names, y=compute_saved,
        marker_color=colors,
        name="Compute",
    ), row=1, col=1)

    colors_m = ["#3498DB" if m > 0 else "#E74C3C" for m in memory_saved]
    fig.add_trace(go.Bar(
        x=names, y=memory_saved,
        marker_color=colors_m,
        name="Memory",
    ), row=1, col=2)

    fig.update_layout(
        title="Conditional Compute Savings Dashboard",
        template="plotly_white",
        width=1000, height=450,
        showlegend=False,
    )
    return fig


def make_all_plots(
    benchmark_results,
    hw_stats: Optional[Dict] = None,
    cc_results: Optional[Dict] = None,
    save_dir: Optional[str] = None,
) -> List[go.Figure]:
    """
    Generate all plots.  If save_dir is given, write HTML files there.
    Returns list of Figure objects.
    """
    figs = [
        ("roofline",         roofline_plot(benchmark_results)),
        ("dram_traffic",     dram_traffic_plot(benchmark_results)),
        ("latency_scaling",  latency_scaling_plot(benchmark_results)),
    ]
    if hw_stats:
        figs.append(("timeloop_energy", timeloop_energy_plot(hw_stats)))

    if cc_results:
        figs.append(("compute_utilization", compute_utilization_plot(cc_results)))
        figs.append(("expert_routing", expert_routing_plot(cc_results)))
        figs.append(("kv_cache", kv_cache_plot(cc_results)))
        figs.append(("savings_dashboard", savings_dashboard(cc_results)))

    if save_dir:
        from pathlib import Path
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs:
            path = out / f"{name}.html"
            fig.write_html(str(path))
            print(f"[plots] Saved {path}")

    return [fig for _, fig in figs]
