"""CSV, JSON, and dashboard artifacts shared by the run scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Protocol

from toy_jepa import constants, experiment
from toy_jepa.viz import plot_dashboard


class RunConfig(Protocol):
    """The fields every arm's Config provides to the artifact writer."""

    @property
    def run_dir(self) -> Path: ...

    @property
    def embed_dim(self) -> int: ...

    @property
    def hidden_dim(self) -> int: ...

    def _asdict(self) -> dict[str, object]: ...


def write_run_artifacts(
    config: RunConfig,
    title: str,
    history: list[experiment.EvalRow],
    metrics: dict[str, float | int],
    readouts: dict[str, dict[str, float]],
) -> None:
    """Write config, history, probes, and the dashboard for one run."""
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **config._asdict(),
        "position_scale": constants.POSITION_SCALE,
        "velocity_scale": constants.VELOCITY_SCALE,
        "seed": constants.SEED,
        "readout": {
            "architecture": [config.embed_dim, config.hidden_dim, 8],
            **constants.READOUT_CONFIG,
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    for name, header, rows in (
        ("metrics.csv", experiment.EvalRow._fields, history),
        ("probes.csv", ("metric", "value"), metrics.items()),
    ):
        with (run_dir / name).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
    plot_dashboard(history, readouts, run_dir / "dashboard.png", title=title)
