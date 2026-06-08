"""Timeloop + Accelergy orchestration via Docker."""
import subprocess
import re
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


_REPO_ROOT = Path(__file__).parent.parent
_WORKSPACE = _REPO_ROOT / "workspace"
_CONFIGS   = _WORKSPACE / "configs"
_OUTPUTS   = _WORKSPACE / "outputs"

DOCKER_IMAGE = "timeloopaccelergy/timeloop-accelergy-pytorch:latest-arm64"

FPGA_CLOCK_FREQ_GHZ = 0.5

_PROBLEMS_GEN = _CONFIGS / "problems" / "generated"
_MAPPINGS_GEN = _CONFIGS / "mappings" / "generated"

MOE_TOP_K = 2
MOE_N_EXPERTS = 8
PAGED_PAGE_SIZE = 16

# JAX attention variant → (QK^T experiment, AV experiment, problem kind)
JAX_VARIANT_TIMELOOP: Dict[str, tuple] = {
    "naive":          ("naive_qk",          "naive_av",          "standard"),
    "flash":          ("flash_qk",          "flash_av",          "standard"),
    "quantized":      ("quantized_qk",      "quantized_av",      "standard"),
    "gqa":            ("gqa_qk",            "gqa_av",            "standard"),
    "paged":          ("paged_qk",          "paged_av",          "paged"),
    "sliding_window": ("sliding_window_qk", "sliding_window_av", "sliding_window"),
    "moe":            ("moe_qk",            "moe_av",            "moe"),
}

EXPERIMENTS: Dict[str, tuple] = {
    "naive_qk": (
        "arch/fpga_like.yaml",
        "problems/qk_matmul.yaml",
        "mappings/naive_qk.yaml",
    ),
    "flash_qk": (
        "arch/fpga_like.yaml",
        "problems/qk_matmul.yaml",
        "mappings/flash_qk.yaml",
    ),
    "naive_av": (
        "arch/fpga_like.yaml",
        "problems/av_matmul.yaml",
        "mappings/naive_av.yaml",
    ),
    "flash_av": (
        "arch/fpga_like.yaml",
        "problems/av_matmul.yaml",
        "mappings/flash_av.yaml",
    ),
    "quantized_qk": ("arch/fpga_like_int8.yaml", "problems/qk_matmul.yaml", "mappings/quantized_qk.yaml"),
    "quantized_av": ("arch/fpga_like_int8.yaml", "problems/av_matmul.yaml", "mappings/quantized_av.yaml"),
    "sliding_window_qk": ("arch/fpga_like.yaml", "problems/sliding_window_qk_matmul.yaml", "mappings/sliding_window_qk.yaml"),
    "sliding_window_av": ("arch/fpga_like.yaml", "problems/sliding_window_av_matmul.yaml", "mappings/sliding_window_av.yaml"),
    "gqa_qk": ("arch/fpga_like.yaml", "problems/qk_matmul.yaml", "mappings/gqa_qk.yaml"),
    "gqa_av": ("arch/fpga_like.yaml", "problems/av_matmul.yaml", "mappings/gqa_av.yaml"),
    "dsa_score":  ("arch/fpga_like.yaml", "problems/dsa_score.yaml",  "mappings/dsa_score.yaml"),
    "dsa_output": ("arch/fpga_like.yaml", "problems/dsa_output.yaml", "mappings/dsa_output.yaml"),
    "paged_qk": ("arch/fpga_like.yaml", "problems/qk_matmul.yaml", "mappings/paged_qk.yaml"),
    "paged_av": ("arch/fpga_like.yaml", "problems/av_matmul.yaml", "mappings/paged_av.yaml"),
    "moe_qk": ("arch/fpga_like.yaml", "problems/moe_qk_matmul.yaml", "mappings/moe_qk.yaml"),
    "moe_av": ("arch/fpga_like.yaml", "problems/moe_av_matmul.yaml", "mappings/moe_av.yaml"),
}


def _problem_kind(experiment_name: str) -> str:
    if experiment_name.endswith("_qk") or experiment_name.endswith("_score"):
        return "qk"
    return "av"


def _write_gemm_problem(
    path: Path,
    M: int,
    N: int,
    K: int,
    kind: str,
) -> None:
    """Write a Timeloop GEMM problem YAML for QK or AV attention phase."""
    if kind == "qk":
        path.write_text(
            f"# Auto-generated QK GEMM  M={M} N={N} K={K}\n"
            f"problem:\n"
            f"  shape:\n"
            f"    name: GEMM\n"
            f"    dimensions: [M, N, K]\n"
            f"    data_spaces:\n"
            f"      - name: A\n"
            f"        projection:\n"
            f"          - [[M]]\n"
            f"          - [[K]]\n"
            f"      - name: B\n"
            f"        projection:\n"
            f"          - [[N]]\n"
            f"          - [[K]]\n"
            f"      - name: Z\n"
            f"        projection:\n"
            f"          - [[M]]\n"
            f"          - [[N]]\n"
            f"        read_write: True\n"
            f"  M: {M}\n"
            f"  N: {N}\n"
            f"  K: {K}\n"
        )
    else:
        path.write_text(
            f"# Auto-generated AV GEMM  M={M} N={N} K={K}\n"
            f"problem:\n"
            f"  shape:\n"
            f"    name: GEMM\n"
            f"    dimensions: [M, N, K]\n"
            f"    data_spaces:\n"
            f"      - name: A\n"
            f"        projection:\n"
            f"          - [[M]]\n"
            f"          - [[N]]\n"
            f"      - name: B\n"
            f"        projection:\n"
            f"          - [[N]]\n"
            f"          - [[K]]\n"
            f"      - name: Z\n"
            f"        projection:\n"
            f"          - [[M]]\n"
            f"          - [[K]]\n"
            f"        read_write: True\n"
            f"  M: {M}\n"
            f"  N: {N}\n"
            f"  K: {K}\n"
        )


def _split_dim(dim: int, outer: int) -> Tuple[int, int]:
    """Split dim into (outer, inner) temporal factors for Timeloop mappings."""
    if dim % outer != 0:
        for o in (4, 8, 16, 2):
            if dim % o == 0:
                outer = o
                break
        else:
            return 1, dim
    return outer, dim // outer


def _write_paged_mapping(path: Path, M: int, N: int, K: int, phase: str, page_size: int) -> None:
    """Page-granular DRAM tiling; flash-style SRAM for scores."""
    m_dram, m_gb = _split_dim(M, 8)
    n_dram, n_gb = _split_dim(N, page_size)
    if phase == "qk":
        dram_keep, dram_bypass = "[A, B]", "[Z]"
        gb_keep = "[A, B, Z]"
    else:
        dram_keep, dram_bypass = "[B, Z]", "[A]"
        gb_keep = "[A, B, Z]"
    path.write_text(
        f"# Auto-generated paged {phase.upper()}  M={M} N={N} page={page_size}\n"
        f"mapping:\n"
        f"  - target: DRAM\n    type: temporal\n    factors: M{m_dram} N{n_dram} K1\n    permutation: MNK\n"
        f"  - target: DRAM\n    type: datatype\n    keep:   {dram_keep}\n    bypass: {dram_bypass}\n"
        f"  - target: GlobalBuffer\n    type: temporal\n    factors: M{m_gb} N{n_gb} K1\n    permutation: MNK\n"
        f"  - target: GlobalBuffer\n    type: datatype\n    keep:   {gb_keep}\n    bypass: []\n"
        f"  - target: RegisterFile\n    type: temporal\n    factors: M1 N1 K{K}\n    permutation: KMN\n"
        f"  - target: RegisterFile\n    type: datatype\n    keep:   [Z]\n    bypass: [A, B]\n"
    )


def _write_moe_mapping(path: Path, M: int, N: int, K: int, phase: str) -> None:
    """Global-KV MoE: sparse M (routed queries), full N (global K/V)."""
    m_dram, m_gb = _split_dim(M, 4)
    n_dram, n_gb = _split_dim(N, 8)
    if phase == "qk":
        dram_keep, dram_bypass = "[B]", "[A, Z]"
    else:
        dram_keep, dram_bypass = "[B, Z]", "[A]"
    path.write_text(
        f"# Auto-generated MoE {phase.upper()}  M={M} N={N} (global KV)\n"
        f"mapping:\n"
        f"  - target: DRAM\n    type: temporal\n    factors: M{m_dram} N{n_dram} K1\n    permutation: MNK\n"
        f"  - target: DRAM\n    type: datatype\n    keep:   {dram_keep}\n    bypass: {dram_bypass}\n"
        f"  - target: GlobalBuffer\n    type: temporal\n    factors: M{m_gb} N{n_gb} K1\n    permutation: MNK\n"
        f"  - target: GlobalBuffer\n    type: datatype\n    keep:   [A, B, Z]\n    bypass: []\n"
        f"  - target: RegisterFile\n    type: temporal\n    factors: M1 N1 K{K}\n    permutation: KMN\n"
        f"  - target: RegisterFile\n    type: datatype\n    keep:   [Z]\n    bypass: [A, B]\n"
    )


def _gemm_dims(
    experiment_name: str,
    seq_len: int,
    head_dim: int,
    window: Optional[int],
    top_k: int,
    n_experts: int,
) -> Tuple[int, int, int]:
    """Return (M, N, K) problem dimensions for a Timeloop experiment."""
    if experiment_name.startswith("moe"):
        routed_m = max(1, (seq_len * top_k) // n_experts)
        return routed_m, seq_len, head_dim
    if experiment_name.startswith("sliding_window"):
        win = window if window is not None else min(128, seq_len)
        return seq_len, win, head_dim
    return seq_len, seq_len, head_dim


def _resolve_problem_path(
    experiment_name: str,
    default_problem: str,
    seq_len: int,
    head_dim: int,
    window: Optional[int] = None,
    top_k: int = MOE_TOP_K,
    n_experts: int = MOE_N_EXPERTS,
) -> str:
    """
    Return config-relative problem path, generating YAML when N or D differ
    from the bundled 512×64 defaults.
    """
    M, N, K = _gemm_dims(experiment_name, seq_len, head_dim, window, top_k, n_experts)
    kind = _problem_kind(experiment_name)

    if experiment_name.startswith("moe"):
        if M == 128 and N == 512 and K == 64:
            return default_problem
    elif experiment_name.startswith("sliding_window"):
        win = window if window is not None else min(128, seq_len)
        if seq_len == 512 and head_dim == 64 and win == 128:
            return default_problem
    elif (
        seq_len == 512
        and head_dim == 64
        and window is None
        and not experiment_name.startswith("paged")
    ):
        return default_problem

    _PROBLEMS_GEN.mkdir(parents=True, exist_ok=True)
    tag = f"{kind}_M{M}_N{N}_K{K}"
    out = _PROBLEMS_GEN / f"{tag}.yaml"
    _write_gemm_problem(out, M, N, K, kind)
    return f"problems/generated/{tag}.yaml"


def _resolve_mapping_path(
    experiment_name: str,
    default_mapping: str,
    M: int,
    N: int,
    K: int,
    page_size: int = PAGED_PAGE_SIZE,
) -> str:
    """Return mapping path, generating YAML when dimensions differ from defaults."""
    if experiment_name.startswith("paged"):
        if M == 512 and N == 512 and K == 64:
            return default_mapping
        _MAPPINGS_GEN.mkdir(parents=True, exist_ok=True)
        phase = "qk" if experiment_name.endswith("_qk") else "av"
        tag = f"paged_{phase}_M{M}_N{N}_pg{page_size}"
        out = _MAPPINGS_GEN / f"{tag}.yaml"
        _write_paged_mapping(out, M, N, K, phase, page_size)
        return f"mappings/generated/{tag}.yaml"

    if experiment_name.startswith("moe"):
        if M == 128 and N == 512 and K == 64:
            return default_mapping
        _MAPPINGS_GEN.mkdir(parents=True, exist_ok=True)
        phase = "qk" if experiment_name.endswith("_qk") else "av"
        tag = f"moe_{phase}_M{M}_N{N}_K{K}"
        out = _MAPPINGS_GEN / f"{tag}.yaml"
        _write_moe_mapping(out, M, N, K, phase)
        return f"mappings/generated/{tag}.yaml"

    return default_mapping


def run_experiment(
    name: str,
    seq_len: int = 512,
    head_dim: int = 64,
    window: Optional[int] = None,
    top_k: int = MOE_TOP_K,
    n_experts: int = MOE_N_EXPERTS,
    page_size: int = PAGED_PAGE_SIZE,
    timeout: int = 300,
) -> Dict:
    """
    Run one Timeloop experiment by name (must be a key in EXPERIMENTS).

    Returns a dict with keys:
        cycles, energy_uj, utilization,
        dram_reads_bytes, dram_writes_bytes,
        sram_reads_bytes, sram_writes_bytes,
        name
    """
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{name}'. Valid: {list(EXPERIMENTS)}")

    arch, problem, mapping = EXPERIMENTS[name]
    M, N, K = _gemm_dims(name, seq_len, head_dim, window, top_k, n_experts)
    problem = _resolve_problem_path(
        name, problem, seq_len, head_dim, window, top_k, n_experts
    )
    mapping = _resolve_mapping_path(name, mapping, M, N, K, page_size)
    run_id = f"{name}_N{seq_len}_D{head_dim}"
    if window is not None:
        run_id += f"_W{window}"
    if name.startswith("moe"):
        run_id += f"_M{M}"
    output_dir = _OUTPUTS / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_out = output_dir / "timeloop-model.stats.txt"
    if stats_out.exists():
        stats_out.unlink()

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{_CONFIGS.resolve()}:/configs:ro",
        "-v", f"{output_dir.resolve()}:/output",
        "-w", "/output",
        DOCKER_IMAGE,
        "timeloop-model",
        f"/configs/{arch}",
        f"/configs/{problem}",
        f"/configs/{mapping}",
    ]

    ert_file = output_dir / "timeloop-model.ERT.yaml"
    if ert_file.exists():
        cmd.append("/output/timeloop-model.ERT.yaml")

    print(f"[timeloop] Running '{name}' …")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        raise RuntimeError(
            f"timeloop-model failed for '{name}'.\n"
            f"STDOUT:\n{result.stdout[-3000:]}\n"
            f"STDERR:\n{result.stderr[-3000:]}"
        )

    stats_file = output_dir / "timeloop-model.stats.txt"
    if not stats_file.exists():
        candidates = list(output_dir.glob("*.stats.txt"))
        if not candidates:
            raise FileNotFoundError(
                f"No stats file produced by Timeloop in {output_dir}.\n"
                f"stdout: {result.stdout[-2000:]}"
            )
        stats_file = candidates[0]

    stats = _parse_stats(stats_file)
    stats["name"] = name
    return stats


def run_all(timeout: int = 300) -> Dict[str, Dict]:
    """Run all experiments and return a dict keyed by experiment name."""
    results = {}
    for name in EXPERIMENTS:
        try:
            results[name] = run_experiment(name, timeout=timeout)
        except Exception as e:
            print(f"[timeloop] SKIPPED '{name}': {e}")
            results[name] = {"name": name, "error": str(e)}
    return results


def _parse_stats(stats_file: Path) -> Dict:
    """
    Parse timeloop-model.stats.txt into a structured dict.
    Handles both v0.3 and v0.4 output formats.
    """
    text = stats_file.read_text()
    stats: Dict = {}


    _extract_scalar(text, stats, "cycles",
                    r"Cycles\s*:\s*([\d,]+)")
    _extract_scalar(text, stats, "utilization",
                    r"Utilization\s*:\s*([\d.]+)")
    _extract_scalar(text, stats, "energy_uj",
                    r"Energy\s*:\s*([\d.eE+\-]+)\s*uJ")
    _extract_scalar(text, stats, "gflops",
                    r"GFLOPs?\s*:\s*([\d.eE+\-]+)")


    dram_reads  = _sum_level_stat(text, "DRAM", "Reads")
    dram_writes = _sum_level_stat(text, "DRAM", "Writes")
    sram_reads  = _sum_level_stat(text, "GlobalBuffer", "Reads")
    sram_writes = _sum_level_stat(text, "GlobalBuffer", "Writes")

    stats["dram_reads_bytes"]  = dram_reads
    stats["dram_writes_bytes"] = dram_writes
    stats["sram_reads_bytes"]  = sram_reads
    stats["sram_writes_bytes"] = sram_writes
    stats["dram_total_bytes"]  = (dram_reads or 0) + (dram_writes or 0)
    stats["sram_total_bytes"]  = (sram_reads or 0) + (sram_writes or 0)

    cycles = stats.get("cycles")
    stats["latency_ms"] = (cycles / (FPGA_CLOCK_FREQ_GHZ * 1e9) * 1e3) if cycles else None

    return stats


def _extract_scalar(text: str, out: Dict, key: str, pattern: str) -> None:
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        val_str = m.group(1).replace(",", "")
        out[key] = float(val_str)
    else:
        out[key] = None


def _sum_level_stat(text: str, level_name: str, stat_name: str) -> Optional[float]:
    """
    Sum scalar access counts for a named memory level in timeloop stats.

    stat_name: "reads"  → sum Scalar reads (per-instance)
               "writes" → sum Scalar updates + Scalar fills (per-instance)

    Returns total bytes (scalars × word_bits/8), or None if level not found.
    """
    header = f"=== {level_name} ==="
    start = text.find(header)
    if start == -1:
        return None

    # Grab text up to the next === header (skip the opening header itself)
    end = text.find("\n===", start + len(header))
    section = text[start:end] if end != -1 else text[start:]

    # Only proceed if this is the detailed section (has Word bits in SPECS)
    word_bits_match = re.search(r"Word bits\s*:\s*(\d+)", section)
    if not word_bits_match:
        return None

    bytes_per_word = int(word_bits_match.group(1)) // 8

    total = 0.0
    found = False

    if stat_name.lower() == "reads":
        for m in re.finditer(r"Scalar reads \(per-instance\)\s*:\s*([\d,]+)", section):
            total += float(m.group(1).replace(",", ""))
            found = True
    else:  # writes
        for m in re.finditer(r"Scalar (?:updates|fills) \(per-instance\)\s*:\s*([\d,]+)", section):
            total += float(m.group(1).replace(",", ""))
            found = True

    return (total * bytes_per_word) if found else None



def print_summary(results: Dict[str, Dict]) -> None:
    """Print a comparison table for all experiment results."""
    header = (
        f"{'Experiment':<20} {'Cycles':>12} {'Latency(ms)':>12} "
        f"{'Energy(uJ)':>12} {'DRAM(MB)':>10} {'SRAM(MB)':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        def _mb(v):
            return f"{v/1e6:.2f}" if v is not None else "N/A"
        def _fmt(v):
            return f"{v:.0f}" if v is not None else "N/A"
        def _ms(v):
            return f"{v:.3f}" if v is not None else "N/A"
        print(
            f"{name:<20} {_fmt(r.get('cycles')):>12} "
            f"{_ms(r.get('latency_ms')):>12} "
            f"{_fmt(r.get('energy_uj')):>12} "
            f"{_mb(r.get('dram_total_bytes')):>10} "
            f"{_mb(r.get('sram_total_bytes')):>10}"
        )
    print(f"\n  Clock: {FPGA_CLOCK_FREQ_GHZ*1e3:.0f} MHz  "
          f"— latency_ms = cycles / {FPGA_CLOCK_FREQ_GHZ*1e9:.0e} Hz")
