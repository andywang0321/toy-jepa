# Phase 2

## Question

Does a stop-gradient JEPA objective produce a better multi-purpose world model
than pixel prediction when context length `m` and prediction horizon `k` vary?

Primary endpoint: final position error at the stop time.

## Data

Friction Moving MNIST from `tiny-worlds`:

- `T=200`, multiple bounces per sequence;
- per-digit friction and restitution;
- sequences stop at rest;
- contexts of `m` frames, targets at `t+k` or at the final stop.

Held-out axes:

```text
m train: 2, 4, 8       m test: 3, 6, 12
k train: 1, 2, 4, 8, 16, 32
k test:  3, 6, 12, 24, 48
```

Training also includes a `FINAL` query for the terminal frame/state.

## Models

Shared context path:

```text
per-frame CNN -> temporal transformer -> masked mean -> c_t
```

The query is `(k / 64, final_flag)`, embedded and concatenated with `c_t`.

- **Pixel:** MLP decoder predicts the target frame; loss is pixel MSE.
- **JEPA:** MLP predictor predicts the target-frame embedding; the target
  encoder is stop-gradient; loss is latent MSE.

The same frame encoder and temporal transformer are used by both arms.

## Probes

Frozen features from `c_t` and `z_hat`:

- state at test horizons `k`;
- final state at stop;
- digit identity;
- bounce before the target;
- feature variance, cosine, rank.

The state readout is trained on mixed training `m`/`k`/`FINAL` queries and
evaluated on held-out `m` and `k`.

## Run

```bash
bash scripts/run_p2.sh
```

This runs the pixel arm, the JEPA arm, and the analyzer:

```text
runs/p2_pixel/
runs/p2_jepa/
runs/analysis/p2_report.md
runs/analysis/p2_comparison.png
```

The default screen uses 2,000 sequences. For a full run, raise `num_samples`
in `scripts/p2_pixel.py` and `scripts/p2_jepa.py`.

## Optimizer

Phase 2 uses Optax AdamW. Muon is deferred until AdamW shows underfitting or
instability at a fixed architecture and budget.

## Not in v1

- auxiliary losses;
- future-context targets;
- EMA/SIGReg;
- Muon;
- seed sweep and final-report tables.
