"""Aggregate phase-2 c_t seed runs and write a compact report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "pixel": ROOT / "runs" / "p2" / "pixel",
    "pixel_fm": ROOT / "runs" / "p2" / "pixel_fm",
    "jepa": ROOT / "runs" / "p2" / "jepa",
}
OLD_RUNS = {"pixel": ROOT / "runs" / "p2_pixel", "jepa": ROOT / "runs" / "p2_jepa"}
OUT = ROOT / "runs" / "analysis"
COLORS = {"pixel": "tab:orange", "pixel_fm": "tab:green", "jepa": "tab:blue"}


def _read(path: Path) -> dict[str, float]:
    with path.open() as file:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(file)}


def _seed_runs(name: str) -> list[Path]:
    paths = sorted(RUNS[name].glob("seed_*/probes.csv"))
    if paths:
        return paths
    fallback = OLD_RUNS.get(name, Path()) / "probes.csv"
    return [fallback] if fallback.exists() else []


def _aggregate(paths: list[Path]) -> dict[str, tuple[float, float, int]]:
    values: dict[str, list[float]] = {}
    for path in paths:
        for metric, value in _read(path).items():
            if metric.startswith("c_t_"):
                values.setdefault(metric, []).append(value)
    return {
        metric: (float(np.mean(vals)), float(np.std(vals)), len(vals))
        for metric, vals in values.items()
    }


def _write_summary(runs: dict[str, dict[str, tuple[float, float, int]]]) -> None:
    with (OUT / "p2_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("objective", "metric", "mean", "std", "n"))
        for objective, metrics in runs.items():
            for metric, (mean, std, n) in metrics.items():
                writer.writerow((objective, metric, mean, std, n))


def _report(runs: dict[str, dict[str, tuple[float, float, int]]]) -> None:
    metrics = (
        "c_t_state_k3_position_mae",
        "c_t_state_k24_position_mae",
        "c_t_state_k48_position_mae",
        "c_t_state_final_position_mae",
        "c_t_digit_accuracy",
        "c_t_bounce_accuracy",
        "c_t_bounce_balanced_accuracy",
        "c_t_variance",
        "c_t_cosine",
        "c_t_rank",
    )
    lines = ["# Phase 2 seeds", "", "| metric | pixel | pixel-FM | JEPA |", "|---|---:|---:|---:|"]
    for metric in metrics:
        cells = []
        for objective in ("pixel", "pixel_fm", "jepa"):
            value = runs.get(objective, {}).get(metric)
            cells.append("" if value is None else f"{value[0]:.4f} ± {value[1]:.4f} (n={value[2]})")
        lines.append(f"| {metric} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["", "Final position MAE by context length:", "", "| m | pixel | pixel-FM | JEPA |", "|---:|---:|---:|---:|"]
    for m in (3, 6, 12):
        cells = []
        for objective in ("pixel", "pixel_fm", "jepa"):
            value = runs.get(objective, {}).get(f"c_t_state_final_m{m}_position_mae")
            cells.append("" if value is None else f"{value[0]:.3f} ± {value[1]:.3f}")
        lines.append(f"| {m} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["", "![comparison](p2_comparison.png)", ""]
    (OUT / "p2_report.md").write_text("\n".join(lines))


def _plot(runs: dict[str, dict[str, tuple[float, float, int]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ks = (3, 6, 12, 24, 48)
    for objective, color in COLORS.items():
        means = [runs.get(objective, {}).get(f"c_t_state_k{k}_position_mae", (np.nan, 0, 0))[0] for k in ks]
        stds = [runs.get(objective, {}).get(f"c_t_state_k{k}_position_mae", (np.nan, 0, 0))[1] for k in ks]
        axes[0].errorbar(ks, means, yerr=stds, fmt="o-", color=color, label=objective)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("position MAE")
    axes[0].set_title("held-out horizon")
    axes[0].legend(fontsize=7)

    labels = ("final",)
    x = np.arange(len(labels))
    for offset, objective, color in ((-0.2, "pixel", COLORS["pixel"]), (0.0, "pixel_fm", COLORS["pixel_fm"]), (0.2, "jepa", COLORS["jepa"])):
        value = runs.get(objective, {}).get("c_t_state_final_position_mae", (np.nan, 0, 0))
        axes[1].bar(x + offset, value[0], yerr=value[1], width=0.18, color=color, label=objective, capsize=3)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("final position MAE")
    axes[1].set_title("terminal prediction")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "p2_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    runs = {name: _aggregate(paths) for name in RUNS if (paths := _seed_runs(name))}
    if not runs:
        raise SystemExit("no phase-2 probe files found")
    OUT.mkdir(parents=True, exist_ok=True)
    _write_summary(runs)
    _report(runs)
    _plot(runs)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
