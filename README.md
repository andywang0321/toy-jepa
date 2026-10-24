# toy-jepa

A minimal JAX study of what a JEPA representation learns on Moving MNIST.

## Question

Given `m=2` context frames and a fixed `k=4` horizon, which training objective
produces context features `c_t` that best support future-state prediction under
a frozen, fixed-capacity readout?

```text
c_t = fuse(f(x_{t-1}), f(x_t))
g(c_t) ~= f(x_{t+k})
```

The readout predicts the future state `(x, y, vx, vy)` of both digits. It is
trained after the model on frozen features, so no state loss shapes the
representation.

## Phase 1 answer

At this scale:

- barebones JEPA collapses;
- stop-gradient is sufficient to restore a usable representation;
- EMA does not beat stop-gradient;
- SIGReg does not improve state prediction at the tested weights;
- the pixel baseline is statistically tied with the best JEPA arms on `c_t`
  state prediction, and its `z_hat` readout is better.

State prediction saturates around `R² ~= 0.36`; changing the objective moves
the geometry, not the state information. The full report is
[docs/phase1.md](docs/phase1.md).

The phase 1 results use the original constant-velocity, `T=20` world. The
current data path now generates the friction world from `tiny-worlds` (`T=200`,
damping, inelastic bounces, stop times) for phase 2; phase 1 artifacts remain
historical.

## Protocol

- 10,000 Moving MNIST sequences, `T=20`, `64x64`.
- Split by sequence: 8,000 train / 1,000 val / 1,000 test.
- Context `(x_{t-1}, x_t)`, target `x_{t+k}`, `k=4`.
- Shared CNN frame encoder, context fusion, MLP predictor.
- Common readout: `128 -> 256 -> 8`, GELU hidden, trained on frozen features.
- Readout data: 6,000 / 1,000 / 1,000 train/val/test.
- Five seeds, 5,000 optimizer steps, batch 64.

## Results

Five seeds, mean ± std, test set. `c_t` is the context feature; `z_hat` is the
predicted future latent. `MSE` is normalized state MSE; MAE is in physical
units.

| Arm | c_t MSE | c_t R² | z_hat MSE | pos MAE | vel MAE |
|---|---:|---:|---:|---:|---:|
| M1 pixel | 0.1087 ± 0.0021 | 0.358 ± 0.014 | **0.1094 ± 0.0018** | 6.370 ± 0.110 | 0.998 ± 0.011 |
| M2 JEPA | 0.1628 ± 0.0012 | -0.007 ± 0.003 | 0.1633 ± 0.0010 | 9.045 ± 0.058 | 1.275 ± 0.003 |
| M3 stop-grad | **0.1082 ± 0.0016** | **0.361 ± 0.013** | 0.1266 ± 0.0026 | 6.389 ± 0.111 | 0.992 ± 0.008 |
| M4 EMA | 0.1085 ± 0.0017 | 0.359 ± 0.013 | 0.1289 ± 0.0023 | 6.394 ± 0.133 | 0.992 ± 0.007 |
| M5 SIGReg | 0.1084 ± 0.0026 | 0.359 ± 0.016 | 0.1278 ± 0.0023 | 6.392 ± 0.096 | 0.994 ± 0.013 |

The non-collapsed arms are within one standard deviation of each other on
`c_t`. M2 is the collapsed control. The state target variance is about `0.161`;
M2 is at the mean predictor, while the others reduce the MSE by roughly one
third.

## Layout

```text
src/toy_jepa/
  constants.py, viz.py
  core/        types, optim, train
  data/        dataset, mnist
  evaluation/  protocol, metrics, readout
  models/      mlp, norm, cnn, baseline, jepa_*
scripts/
  m1_pixel.py ... m5_sigreg.py
  sweep_ema.py, sweep_sigreg.py
  sweep_seeds.py
  analyze.py, analyze_seeds.py
  run_sweeps.sh, run_arms.sh, run_seeds.sh
tests/
```

## Run

Set `TINY_WORLDS_MNIST_DIR` (or edit `MNIST_DIR` in `constants.py`) to the MNIST
IDX directory. `tiny-worlds` takes the path explicitly and never vendors MNIST.
Keep `tiny-worlds` as a sibling of `toy-jepa`; the dependency is an editable path.

```bash
uv sync
uv run pytest -q

# 1. Tune EMA/SIGReg first.
bash scripts/run_sweeps.sh

# Inspect runs/analysis/report.md and best.json, then set
# EMA_DECAY in scripts/m4_ema.py and SIGREG_WEIGHT in scripts/m5_sigreg.py.

# 2. Main comparison.
bash scripts/run_arms.sh

# 3. Five-seed replication at 5,000 steps.
bash scripts/run_seeds.sh
```

Typing and linting:

```bash
ty check src/toy_jepa tests scripts
ruff check src/toy_jepa tests scripts
```

## Docs

- [Phase 1 report](docs/phase1.md)
- [Metric guide](docs/metrics.md)

Phase 2 uses the friction world from `tiny-worlds`: variable context `m`,
variable horizon `k`, final-stop targets, and new model/experiment designs.
That is the next implementation step.
