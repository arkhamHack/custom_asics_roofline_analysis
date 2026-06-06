"""Timeloop + Accelergy orchestration via Docker."""
import subprocess
import re
import os
from pathlib import Path
from typing import Dict, Optional


_REPO_ROOT = Path(__file__).parent.parent
_WORKSPACE = _REPO_ROOT / "workspace"
_CONFIGS   = _WORKSPACE / "configs"
_OUTPUTS   = _WORKSPACE / "outputs"

DOCKER_IMAGE = "timeloopaccelergy/timeloop-accelergy-pytorch:latest-arm64"

FPGA_CLOCK_FREQ_GHZ = 0.5

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
    "paged_qk": ("arch/fpga_like.yaml", "problems/qk_matmul.yaml", "mappings/naive_qk.yaml"),
    "paged_av": ("arch/fpga_like.yaml", "problems/av_matmul.yaml", "mappings/naive_av.yaml"),
}


def run_experiment(name: str, timeout: int = 300) -> Dict:
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
    output_dir = _OUTPUTS / name
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
