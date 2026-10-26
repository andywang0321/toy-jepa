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

- **Pixel-MSE:** MLP decoder predicts the target frame; loss is pixel MSE.
- **Pixel-FM:** same context path, MLP velocity field; rectified flow matching
  on future frames.
- **JEPA:** MLP predictor predicts the target-frame embedding; the target
  encoder is stop-gradient; loss is latent MSE.

The same frame encoder and temporal transformer are used by all three arms.

## Probes

Frozen features from `c_t` only. `z_hat` is a JEPA-specific predictor output,
so it is not used as a cross-arm representation.

- state at test horizons `k`;
- final state at stop;
- digit identity;
- bounce before the target;
- feature variance, cosine, rank.

The state readout is trained on mixed training `m`/`k`/`FINAL` queries and
evaluated on held-out `m` and `k`. Results are reported both aggregate and
per context length `m`.

The final state metric is **position MAE**. Final velocity is zero by
construction, so it is excluded from R²-style comparisons.

## Run

```bash
bash scripts/run_all.sh
```

This runs seeds `0..4` by default for pixel, pixel-FM, and JEPA. Pixel and
pixel-FM run sequentially on GPU `0`; JEPA runs on GPU `3`. Override with `PIXEL_GPU`,
`JEPA_GPU`, or `SEEDS`.

```text
runs/p2/pixel/seed_0..4/
runs/p2/pixel_fm/seed_0..4/
runs/p2/jepa/seed_0..4/
checkpoints/p2/pixel/seed_0..4/params.pkl      # gitignored
checkpoints/p2/pixel_fm/seed_0..4/params.pkl
checkpoints/p2/jepa/seed_0..4/params.pkl
runs/analysis/p2_report.md
runs/analysis/p2_comparison.png
```

Per-seed logs are under each run's `train.log`. Checkpoints are local and
gitignored.

## Optimizer

Phase 2 uses Optax AdamW. Muon is deferred until AdamW shows underfitting or
instability at a fixed architecture and budget.

## Not in v1

- auxiliary losses;
- future-context targets;
- EMA/SIGReg;
- Muon;
- shared or post-hoc flow decoding of JEPA features;
- generative sample metrics beyond state probes.
