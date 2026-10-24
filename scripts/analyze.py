"""Compare completed arms and sweeps in ``runs/``."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = RUNS / "analysis"
ARMS = (
    ("M1 pixel", "m1_pixel"),
    ("M2 JEPA", "m2_jepa"),
    ("M3 stop-grad", "m3_stopgrad"),
    ("M4 EMA", "m4_ema"),
    ("M5 SIGReg", "m5_sigreg"),
)
SWEEPS = (
    ("EMA", "sweep_ema", "decay_", "decay"),
    ("SIGReg", "sweep_sigreg", "weight_", "weight"),
)
ARM_FIELDS = (
    "arm",
    "loss",
    "persistence",
    "c_t_mse",
    "c_t_pos_mae",
    "c_t_vel_mae",
    "c_t_r2",
    "z_mse",
    "z_r2",
    "c_t_var",
    "c_t_cos",
    "c_t_rank",
    "num_samples",
    "num_steps",
)
SWEEP_FIELDS = (
    "sweep",
    "setting",
    "c_t_mse",
    "z_mse",
    "c_t_r2",
    "c_t_var",
    "c_t_cos",
    "c_t_rank",
)


def read_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open() as file:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(file) if row.get("value", "") != ""}


def load_arm(name: str, dirname: str) -> dict[str, float | str] | None:
    run = RUNS / dirname
    if not (run / "probes.csv").exists():
        print(f"skip missing {run}")
        return None
    m = read_csv(run / "probes.csv")
    config = json.loads((run / "config.json").read_text())
    return {
        "arm": name,
        "loss": m.get("test_pixel_mse", m.get("test_loss", np.nan)),
        "persistence": m.get(
            "test_persistence_mse",
            m.get("test_latent_persistence_mse", np.nan),
        ),
        "c_t_mse": m["c_t_state_mse"],
        "c_t_pos_mae": m["c_t_position_mae"],
        "c_t_vel_mae": m["c_t_velocity_mae"],
        "c_t_r2": m["c_t_r2"],
        "z_mse": m["z_hat_state_mse"],
        "z_r2": m["z_hat_r2"],
        "c_t_var": m["c_t_variance"],
        "c_t_cos": m["c_t_cosine"],
        "c_t_rank": m["c_t_rank"],
        "num_samples": int(config.get("num_samples", 0)),
        "num_steps": int(config.get("num_steps", 0)),
    }


def write_table(rows: list[dict], fields: tuple[str, ...], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], fields: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        cells = []
        for field in fields:
            value = row[field]
            cells.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_arms(rows: list[dict], path: Path) -> None:
    names = [str(row["arm"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].bar(x - 0.18, [r["c_t_mse"] for r in rows], 0.36, label="c_t")
    axes[0, 0].bar(x + 0.18, [r["z_mse"] for r in rows], 0.36, label="z_hat")
    axes[0, 0].set_title("state readout MSE")
    axes[0, 0].set_xticks(x, names, rotation=20)
    axes[0, 0].legend()

    axes[0, 1].bar(x - 0.18, [r["c_t_pos_mae"] for r in rows], 0.36, label="position")
    axes[0, 1].bar(x + 0.18, [r["c_t_vel_mae"] for r in rows], 0.36, label="velocity")
    axes[0, 1].set_title("c_t MAE (px, px/frame)")
    axes[0, 1].set_xticks(x, names, rotation=20)
    axes[0, 1].legend()

    axes[1, 0].bar(x - 0.18, [r["c_t_r2"] for r in rows], 0.36, label="c_t")
    axes[1, 0].bar(x + 0.18, [r["z_r2"] for r in rows], 0.36, label="z_hat")
    axes[1, 0].set_title("state readout R²")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.5)
    axes[1, 0].set_xticks(x, names, rotation=20)
    axes[1, 0].legend()

    axes[1, 1].bar(x - 0.18, [r["c_t_var"] for r in rows], 0.36, label="c_t variance")
    axes[1, 1].bar(x + 0.18, [r["c_t_rank"] for r in rows], 0.36, label="c_t rank")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("c_t geometry")
    axes[1, 1].set_xticks(x, names, rotation=20)
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_sweeps() -> list[dict]:
    rows = []
    for label, dirname, prefix, key in SWEEPS:
        for run in sorted((RUNS / dirname).glob(f"{prefix}*")):
            probes = read_csv(run / "probes.csv")
            if not probes:
                continue
            rows.append(
                {
                    "sweep": label,
                    "setting": float(run.name.removeprefix(prefix)),
                    "c_t_mse": probes["c_t_state_mse"],
                    "z_mse": probes["z_hat_state_mse"],
                    "c_t_r2": probes["c_t_r2"],
                    "c_t_var": probes["c_t_variance"],
                    "c_t_cos": probes["c_t_cosine"],
                    "c_t_rank": probes["c_t_rank"],
                }
            )
    return rows


def plot_sweeps(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(10, 3.5))
    for axis, (label, *_) in zip(axes, SWEEPS):
        subset = [row for row in rows if row["sweep"] == label]
        if not subset:
            axis.set_title(f"{label}: no runs")
            continue
        settings = [row["setting"] for row in subset]
        axis.plot(settings, [row["c_t_mse"] for row in subset], "o-", label="c_t")
        axis.plot(settings, [row["z_mse"] for row in subset], "o-", label="z_hat")
        axis.set_xlabel("setting")
        axis.set_ylabel("state MSE")
        axis.set_title(label)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(arms: list[dict], sweeps: list[dict], path: Path) -> None:
    lines = [
        "# Run comparison",
        "",
        "## Main arms",
        "",
        markdown_table(arms, ARM_FIELDS),
        "",
    ]
    if sweeps:
        lines += ["## Sweeps", "", markdown_table(sweeps, SWEEP_FIELDS), ""]
        for label, *_ in SWEEPS:
            subset = [row for row in sweeps if row["sweep"] == label]
            if subset:
                best = min(subset, key=lambda row: row["c_t_mse"])
                lines.append(f"- Best {label} by c_t MSE: {best['setting']} ({best['c_t_mse']:.4f})")
    lines += ["", "![arms](comparison.png)", ""]
    if sweeps:
        lines += ["![sweeps](sweeps.png)", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arms = [row for name, dirname in ARMS if (row := load_arm(name, dirname))]
    sweeps = load_sweeps()
    if arms:
        write_table(arms, ARM_FIELDS, OUT / "summary.csv")
        plot_arms(arms, OUT / "comparison.png")
    if sweeps:
        write_table(sweeps, SWEEP_FIELDS, OUT / "sweeps.csv")
        plot_sweeps(sweeps, OUT / "sweeps.png")
        best = {
            label: min(
                (row for row in sweeps if row["sweep"] == label),
                key=lambda row: row["c_t_mse"],
            )["setting"]
            for label, *_ in SWEEPS
            if any(row["sweep"] == label for row in sweeps)
        }
        (OUT / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    write_report(arms, sweeps, OUT / "report.md")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
