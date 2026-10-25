"""Compare phase-2 runs and write a small report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"pixel": ROOT / "runs" / "p2_pixel", "jepa": ROOT / "runs" / "p2_jepa"}
OUT = ROOT / "runs" / "analysis"


def _read(path: Path) -> dict[str, float]:
    with path.open() as file:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(file)}


def _probes(runs: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    rows = []
    for objective, probes in runs.items():
        for metric, value in probes.items():
            rows.append({"objective": objective, "metric": metric, "value": value})
    return rows


def main() -> None:
    runs = {name: _read(path / "probes.csv") for name, path in RUNS.items() if (path / "probes.csv").exists()}
    if not runs:
        raise SystemExit("no phase-2 probe files found")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _probes(runs)

    with (OUT / "p2_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("objective", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)

    metrics = sorted(runs["pixel"]) if "pixel" in runs else sorted(next(iter(runs.values())))
    lines = ["# Phase 2", "", "| metric | pixel | JEPA |", "|---|---:|---:|"]
    for metric in metrics:
        values = [runs.get(name, {}).get(metric) for name in ("pixel", "jepa")]
        cells = ["" if value is None else f"{value:.5f}" for value in values]
        lines.append(f"| {metric} | {cells[0]} | {cells[1]} |")
    (OUT / "p2_report.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ("pixel", "jepa"):
        for feature in ("c_t", "z_hat"):
            values = []
            for k in (3, 6, 12, 24, 48):
                values.append(runs.get(name, {}).get(f"{feature}_state_k{k}_position_mae", np.nan))
            axes[0].plot((3, 6, 12, 24, 48), values, marker="o", label=f"{name}/{feature}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("position MAE")
    axes[0].set_title("held-out horizon")
    axes[0].legend(fontsize=7)
    for name in ("pixel", "jepa"):
        for feature in ("c_t", "z_hat"):
            axes[1].scatter(
                runs.get(name, {}).get(f"{feature}_digit_accuracy", np.nan),
                runs.get(name, {}).get(f"{feature}_state_final_position_mae", np.nan),
                label=f"{name}/{feature}",
            )
    axes[1].set_xlabel("digit accuracy")
    axes[1].set_ylabel("final position MAE")
    axes[1].set_title("multi-purpose probes")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "p2_comparison.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
