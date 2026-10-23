"""Aggregate the five-seed runs into mean/std tables and plots."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs" / "seeds"
OUT = Path(__file__).resolve().parents[1] / "runs" / "analysis"
ARMS = ("m1_pixel", "m2_jepa", "m3_stopgrad", "m4_ema", "m5_sigreg")
METRICS = (
    "c_t_state_mse",
    "c_t_position_mae",
    "c_t_velocity_mae",
    "c_t_r2",
    "z_hat_state_mse",
    "z_hat_r2",
    "c_t_variance",
    "c_t_cosine",
    "c_t_rank",
)


def read_probes(path: Path) -> dict[str, float]:
    with path.open() as file:
        return {
            row["metric"]: float(row["value"])
            for row in csv.DictReader(file)
            if row.get("value", "") != ""
        }


def load_arm(arm: str) -> dict[str, tuple[float, float, int]]:
    runs = sorted((RUNS / arm).glob("seed_*/probes.csv"))
    values = {metric: [] for metric in METRICS}
    for path in runs:
        probes = read_probes(path)
        for metric in METRICS:
            values[metric].append(probes[metric])
    return {
        metric: (
            float(np.mean(vals)),
            float(np.std(vals)),
            len(vals),
        )
        for metric, vals in values.items()
    }


def write_summary(rows: dict[str, dict[str, tuple[float, float, int]]]) -> None:
    with (OUT / "seed_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("arm", "metric", "mean", "std", "n"))
        for arm, metrics in rows.items():
            for metric, (mean, std, n) in metrics.items():
                writer.writerow((arm, metric, mean, std, n))


def markdown(rows: dict[str, dict[str, tuple[float, float, int]]]) -> str:
    lines = [
        "# Seed sweep",
        "",
        "Five seeds, 5,000 steps, 10,000 samples.",
        "",
        "| arm | c_t MSE | c_t R² | pos MAE | vel MAE | z MSE | c_t var | c_t cos | c_t rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, metrics in rows.items():

        def cell(name: str) -> str:
            mean, std, _ = metrics[name]
            return f"{mean:.4f} ± {std:.4f}"

        lines.append(
            f"| {arm} | {cell('c_t_state_mse')} | {cell('c_t_r2')} | "
            f"{cell('c_t_position_mae')} | {cell('c_t_velocity_mae')} | "
            f"{cell('z_hat_state_mse')} | {cell('c_t_variance')} | "
            f"{cell('c_t_cosine')} | {cell('c_t_rank')} |"
        )
    lines += ["", "![seed comparison](seed_comparison.png)", ""]
    return "\n".join(lines)


def plot(rows: dict[str, dict[str, tuple[float, float, int]]]) -> None:
    arms = list(rows)
    x = np.arange(len(arms))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    def errorbar(axis, metric: str, label: str, color: str) -> None:
        means = [rows[arm][metric][0] for arm in arms]
        stds = [rows[arm][metric][1] for arm in arms]
        axis.errorbar(x, means, yerr=stds, fmt="o-", label=label, color=color)

    errorbar(axes[0, 0], "c_t_state_mse", "c_t", "tab:blue")
    errorbar(axes[0, 0], "z_hat_state_mse", "z_hat", "tab:orange")
    axes[0, 0].set_title("state readout MSE")
    axes[0, 0].set_xticks(x, arms, rotation=20)
    axes[0, 0].legend()

    errorbar(axes[0, 1], "c_t_position_mae", "position", "tab:green")
    errorbar(axes[0, 1], "c_t_velocity_mae", "velocity", "tab:red")
    axes[0, 1].set_title("c_t MAE (px, px/frame)")
    axes[0, 1].set_xticks(x, arms, rotation=20)
    axes[0, 1].legend()

    errorbar(axes[1, 0], "c_t_r2", "c_t", "tab:blue")
    errorbar(axes[1, 0], "z_hat_r2", "z_hat", "tab:orange")
    axes[1, 0].set_title("state readout R²")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.5)
    axes[1, 0].set_xticks(x, arms, rotation=20)
    axes[1, 0].legend()

    errorbar(axes[1, 1], "c_t_variance", "variance", "tab:purple")
    errorbar(axes[1, 1], "c_t_rank", "rank", "tab:brown")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("c_t geometry")
    axes[1, 1].set_xticks(x, arms, rotation=20)
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "seed_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rows = {arm: metrics for arm in ARMS if (metrics := load_arm(arm))}
    if not rows:
        raise SystemExit(f"no seed runs found under {RUNS}")
    OUT.mkdir(parents=True, exist_ok=True)
    write_summary(rows)
    (OUT / "seed_report.md").write_text(markdown(rows))
    plot(rows)
    print(f"wrote {OUT / 'seed_report.md'}")


if __name__ == "__main__":
    main()
