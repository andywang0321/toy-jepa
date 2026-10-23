# toy-jepa

A minimal, from-scratch JAX implementation of JEPA on Moving MNIST.

The goal is understanding, not benchmark performance. By the end you should be able to explain what JEPA predicts, why it collapses, what stop-gradient and EMA actually change, and what SIGReg buys you — and you should have the JAX/PyTree/jit habits to build the policy version next.

## The question

Given m=2 context frames and a fixed k=4 horizon, which training objective produces context features `c_t` that best support future-state prediction under a fixed-capacity readout? The readout is trained after the model on frozen features, so no auxiliary state loss shapes the representation.

With two context frames `(x_{t-1}, x_t)`, learn a frame encoder, a fusion module, and a predictor such that

```
c_t = fuse(f_frame(x_{t-1}), f_frame(x_t))
g(c_t) ≈ f_frame(x_{t+k})
```

without reconstructing pixels. Then ask: what does the representation capture, when does it collapse, and which fixes actually help?

## Scope

**In scope (v1)**

- Moving MNIST, temporal latent prediction.
- One CNN encoder, one MLP predictor.
- Baseline: same encoder/trunk, pixel-only head; state quality is measured post hoc with a common MLP readout.
- JEPA variants: barebones, stop-grad, EMA, EMA + SIGReg.
- Probes: digit identity, position, velocity; collapse metrics.
- 5-seed comparison at 5,000 steps and a short write-up.

**Out of scope (v1)**

- Push-T, JEPA-Policy, ABC, planning, control.
- ViT/patch tokens, I-JEPA masking, VICReg, diffusion.
- SOTA results, large-scale training, hyperparameter search.

## Rules of the game

1. **Vanilla JAX.** Only `jax`, `jax.numpy`, `numpy`, `matplotlib`, `pytest` (plus `tqdm` if you want). No Flax, Equinox, Haiku, or other model frameworks. No classes with methods; a frozen dataclass or `NamedTuple` for static config is fine.
2. **Params are PyTrees.** A model is `init_params(rng, config) -> params` and `apply(params, x) -> y`. Params are named records (`NamedTuple`) of arrays.
3. **Pure core, impure shell.** Model/loss/optimizer/train-step functions do not read files, plot, print, or touch global RNG. Scripts handle seeding, logging, checkpointing, and plotting. Data I/O is `data/mnist.py` and `data/dataset.py`; plotting is `viz.py`.
4. **One level of abstraction per function.** No function both loads data and trains. No generic framework, registries, or `Any`-heavy config objects.
5. **JIT at the boundary.** `train_step` is pure, wrapped in `jax.jit` inside each arm script.
6. **Explicit randomness.** Pass `PRNGKey`s; use `jax.random.split`; no `np.random` inside pure functions.
7. **Tests.** Shape tests, gradient checks, EMA updates, readout early stopping, and batch/split tests. Keep them cheap.

## Data

- Generated Moving MNIST: `(N, T, 64, 64)`, `N=10_000`, `T=20`.
- Split by sequence: 8000 train / 1000 val / 1000 test. Never split by frame.
- Context `(x_{t-1}, x_t)`, target `x_{t+k}`; `k=4` default. Two frames are needed to infer velocity.
- Normalize to `[0, 1]`; no augmentation in v1.
- `generate_dataset(seed, num_samples)` loads MNIST from `MNIST_DIR`, generates a fixed in-memory pool, and returns a `MovingMNIST`. Call it once per run. Use the same seed for every arm.

The standard `mnist_test_seq.npy` contains frames only; there are no digit labels, positions, or velocities. Download the IDX files into `MNIST_DIR` (default `~/data/mnist`; at least the `train-*` pair, the `t10k-*` pair is optional). Expected names are `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, and optionally the `t10k-*` equivalents; hyphenated `.gz` variants work too. `data.mnist.load_mnist` parses them and does not download anything.

Data defaults live in `src/toy_jepa/constants.py`:

```python
MNIST_DIR = Path.home() / "data" / "mnist"
NUM_SAMPLES = 10_000
NUM_FRAMES = 20
CANVAS_SIZE = 64
DIGIT_SIZE = 28
SPEED_RANGE = (1.0, 3.0)
SEED = 0
```

The returned `MovingMNIST` has:

| array | shape | dtype | meaning |
|---|---|---|---|
| `frames` | `(N, T, 64, 64)` | `uint8` | rendered frames |
| `digit_label` | `(N, 2)` | `int8` | MNIST class labels (0–9) |
| `digit_index` | `(N, 2)` | `int32` | source MNIST example indices |
| `position` | `(N, T, 2, 2)` | `float32` | top-left `(x, y)` per digit, pixels |
| `velocity` | `(N, T, 2, 2)` | `float32` | `(vx, vy)` per digit, pixels/frame |
| `split` | `(N,)` | `int8` | 0=train, 1=val, 2=test |

Record positions even when the digits overlap; the renderer composites with max, and the labels stay clean. These labels are for probes and the post-hoc readout; they are **not** extra JEPA losses.

## Models

### Frame encoder

Per frame: 3 conv layers (32, 64, 128 channels; 3×3; stride 2; GroupNorm; GELU), then **flatten** the `8×8×128` feature map, a linear projection to `d=128`, and a final LayerNorm. Global average pooling is deliberately avoided: it makes the representation translation-invariant, so the model cannot know where the digits are and the probes fail.

### Context fusion and predictor

Encode `x_{t-1}` and `x_t` with the shared frame encoder, concatenate the two 128-d features, and fuse with `256 -> 256 -> 128` (GELU, linear). The result is `c_t`. The predictor is a 2-layer MLP `128 -> 256 -> 128` with LayerNorm + GELU. No dropout in v1.

### Baseline — pixel only + common readout

Same frame encoder, fusion, and predictor trunk, with a pixel head `128 -> 512 -> 4096` reshaped to `64×64`. Train with pixel MSE only. `apply_baseline -> (pixels, c_t, z_hat)`.

After training, freeze the model and fit the **common MLP readout** (`128 -> 256 -> 8`, GELU hidden, linear output) to predict the normalized state at `t+k` from `c_t` (primary) and `z_hat` (secondary). The readout only sees frozen arrays, so it never backpropagates into the encoder. Normalize position `/ 36` and velocity `/ 3`; report MSE, position MAE in pixels, velocity MAE in pixels/frame, and R². Compare pixels against persistence.

### JEPA variants

Each variant lives in its own small module with its own params, init, apply, and loss. Duplication is intentional; the differences stay visible.

- **Barebones (`jepa_base.py`):** shared encoder, no stop-grad.
  `L = || g(c_t) - f_frame(x_{t+k}) ||²`.
- **Stop-grad (`jepa_stopgrad.py`):** `L = || g(c_t) - sg(f_frame(x_{t+k})) ||²`.
- **EMA (`jepa_ema.py`):** target encoder `f_ema` updated as
  `f_ema ← decay * f_ema + (1 - decay) * f`; loss
  `L = || g(c_t) - sg(f_ema(x_{t+k})) ||²`, decay 0.99.
- **EMA + SIGReg (`jepa_sigreg.py`):** `L = L_JEPA + λ * L_SIGReg(c_t)`, with `λ=0.1` by default and a fixed SIGReg key (`SEED + 5`).

`models/ema.py` contains the pure EMA update; `core/train.py` contains generic `train_step` and `ema_train_step` with no model-specific losses.

### SIGReg

Follow LeJEPA §3:

```
φ̂_m(t) = (1/B) Σ_b exp(i t (z_b · u_m))
L_SIGReg = (1/M) Σ_m Σ_j w_j |φ̂_m(t_j) - exp(-t_j²/2)|²
```

`u_m` are random unit directions, `t_j` is a trapezoid quadrature grid, and `w_j` are weights. Apply it to `c_t`, not the EMA target. Defaults: `M=256`, 17 grid points in `[0, 3]`.

SIGReg is not guaranteed to help. Report what you see, including negative results.

## Metrics

At every evaluation:

- `loss`: pixel MSE for the baseline, or latent MSE for JEPA.
- `rank`: effective rank / participation ratio of centered features:
  `(Σ σ_i)² / Σ σ_i²`.
- `var`: mean per-dimension variance.
- `cos`: mean pairwise cosine similarity.
- `pixel_mse`: pixel-head error for the baseline.
- `state_mse`, `position_mae`, `velocity_mae`, `r2`: frozen-feature MLP readout of the future state, for both `c_t` and `z_hat`.
- `target_variance`: variance of the target features.

The **collapse dashboard** is one figure with loss/persistence, rank, variance, and cosine over training, plus final readout state MSE bars. A lower loss with a lower rank/probe is the signature of collapse. Loss alone is not evidence of a good representation. Compute probes and metrics in NumPy float64 after transferring features; training stays float32.

See [docs/metrics.md](docs/metrics.md) for a first-principles guide.

## Protocol and scale

Every arm uses the same defaults (override the `Config` in a script for a smoke test):

- `NUM_SAMPLES = 10_000`, `NUM_STEPS = 3_000`, `BATCH_SIZE = 64`, `K = 4`.
- Fixed evaluation and probe batches of 1,000 contexts.
- Readout train/val/test contexts: 6,000 / 1,000 / 1,000, drawn once from `np.random.default_rng(SEED + 2)` in train/val/test order.
- Readout architecture `128 -> 256 -> 8`, AdamW with validation early stopping. Readout hyperparameters live in `constants.py`.
- Model init and both readout keys come from `jax.random.split(jax.random.key(SEED), 3)`, so every arm sees the same contexts and keys.
- Freeze the **online** encoder after training. `c_t` is primary; `z_hat` is secondary.

## Curriculum

| Module | Question | Build | Deliverable |
|---|---|---|---|
| M0 | Can I load, split, and overfit? | Data + MLP warm-up | Overfit curve, tests |
| M1 | What does a pixel objective learn? | Pixel-only baseline + common readout | Pixel/readout MSE + collapse metrics |
| M2 | What happens with no asymmetry? | Barebones JEPA | Collapse dashboard |
| M3 | Does stop-grad fix it? | JEPA + stop-grad | Rank/probe comparison |
| M4 | Does a moving target stabilize it? | JEPA + EMA | EMA decay sweep |
| M5 | Does an explicit regularizer help? | JEPA + SIGReg | λ sweep, rank/probe comparison |
| M6 | Is the effect real? | 5-seed ablations + report | Table, dashboard, write-up |

**Minimum path:** M0–M5 with one seed each, then the core arms in M6 with five seeds.

### M0 — Data + JAX warm-up

Generate, split, and visualize Moving MNIST in memory; inspect frames, labels, positions, and velocities. Write a 2-layer MLP with manual AdamW, `jit`, `vmap`, and `grad`; overfit 8 samples; add gradient and shape tests. M0 predicts a single frame's state, so velocity is only memorized for 8 samples. M1 fixes this with two context frames.

### M1 — Pixel-only baseline + common readout

Train the pixel baseline (`scripts/m1_pixel.py`), compare against persistence (`x_t`), then freeze it and train the common readout on `c_t` and `z_hat`. Write `runs/m1_pixel/` (`config.json`, `metrics.csv`, `probes.csv`, `dashboard.png`). Done when pixel MSE beats persistence, both readouts are reported, and the dashboard exists.

### M2 — Barebones JEPA

`scripts/m2_jepa.py` trains the shared-encoder model with no target asymmetry. `scripts/m2_jepa.py` owns its full loop: data, training, evaluation, readout, and logging. Write `runs/m2_jepa/`. Done when loss/rank curves and the final readout are logged and you can say whether collapse occurred.

### M3 — Stop-grad

`scripts/m3_stopgrad.py` uses `jepa_stopgrad.py`, which applies `jax.lax.stop_gradient` to the shared-encoder target. Everything else matches M2. Write `runs/m3_stopgrad/`. Compare rank, collapse metrics, and readout MSE to M2.

### M4 — EMA

`scripts/m4_ema.py` pairs the online `TrainState` with an EMA target encoder in `core.types.EMAState`. `ema_train_step` updates the online model with AdamW, then EMA-updates the target outside the optimizer. Default decay is 0.99. Write `runs/m4_ema/`. Sweep `decay ∈ {0.99, 0.996, 0.999}` with `scripts/sweep_ema.py`.

### M5 — SIGReg

`scripts/m5_sigreg.py` adds `weight * SIGReg(c_t)` to the EMA loss using a fixed key (`SEED + 5`). Default weight is 0.1. Write `runs/m5_sigreg/`. Sweep `λ ∈ {0.01, 0.03, 0.1}` with `scripts/sweep_sigreg.py`.

### M6 — Ablations + report

5 seeds per arm at 5,000 steps; `k=1` vs `k=4`; predictor depth; batch size. Final table and dashboard; 1–2 page write-up.

## Sweeps

Both sweeps are standalone scripts with no CLI arguments. Each run uses the same scaled protocol, readout, contexts, and keys as its arm, and writes its own `config.json`, `metrics.csv`, `probes.csv`, and `dashboard.png`.

- `scripts/sweep_ema.py`: decay `[0.99, 0.996, 0.999]` → `runs/sweep_ema/decay_<value>/`.
- `scripts/sweep_sigreg.py`: weight `[0.01, 0.03, 0.1]`, EMA decay 0.99 → `runs/sweep_sigreg/weight_<value>/`.

## Collapse knobs

If barebones JEPA does not collapse, that is a result, not a failure. To probe the boundary: increase predictor depth/width; remove LayerNorm; increase learning rate and train longer; reduce batch size; use `k=1`; try a residual predictor. If it collapses immediately and never recovers, try the opposite. Record what happened.

## Questions the project should answer

1. Does the baseline learn better probes than JEPA? For which factors?
2. What does barebones JEPA collapse to, and how does it show up in the metrics?
3. Does stop-grad prevent collapse? Does EMA? Does SIGReg?
4. Is a lower JEPA loss always a better representation? (No — show it.)
5. Which factors (digit, position, velocity) does each variant capture?
6. What is the cost of SIGReg in compute and probe quality?

## Repo layout

```
src/toy_jepa/
  constants.py            # data and readout constants
  viz.py                  # plotting, including the dashboard
  core/
    types.py              # array aliases, dataset/param records, states
    optim.py              # manual AdamW
    train.py              # generic train_step and ema_train_step
  data/
    dataset.py            # generate_dataset, pairs, state lookup
    mnist.py              # MNIST IDX loading
  evaluation/
    protocol.py           # batches, state normalization, readout splits, EvalRow
    metrics.py            # rank, variance, cosine, state metrics
    readout.py            # frozen-feature MLP readout
  models/
    mlp.py, norm.py, cnn.py
    baseline.py           # pixel-only params/init/apply/loss
    jepa_base.py          # barebones params/init/apply/loss
    jepa_stopgrad.py      # stop-grad params/init/apply/loss
    jepa_ema.py           # EMA params/init/apply/loss
    jepa_sigreg.py        # EMA + SIGReg params/init/apply/loss
    ema.py, sigreg.py     # pure helpers
scripts/
  m0_overfit.py
  m1_pixel.py ... m5_sigreg.py
  sweep_ema.py, sweep_sigreg.py
  sweep_seeds.py           # M1-M5 at five seeds, 5,000 steps
  analyze.py               # compare completed runs and sweeps
  analyze_seeds.py         # mean/std over seed runs
  run_sweeps.sh            # phase 1
  run_arms.sh              # phase 2
  run_seeds.sh             # seed replication
  _artifacts.py           # CSV/JSON/dashboard writer
  visualize_data.py
tests/
```

## How to run

Download the MNIST IDX files into `~/data/mnist`, then:

```bash
uv sync
uv run pytest -q
uv run python scripts/visualize_data.py

# Phase 1: tune EMA and SIGReg first.
bash scripts/run_sweeps.sh

# Inspect runs/analysis/report.md and runs/analysis/best.json.
# Set EMA_DECAY in scripts/m4_ema.py and
# SIGREG_WEIGHT in scripts/m5_sigreg.py.

# Phase 2: the main comparison with the selected values.
bash scripts/run_arms.sh

# Seed replication: five seeds, 5,000 steps, all five arms.
bash scripts/run_seeds.sh
```

`run_seeds.sh` writes `runs/seeds/<arm>/seed_<n>/` and summarizes to
`runs/analysis/seed_summary.csv`, `seed_report.md`, and `seed_comparison.png`.

`analyze.py` writes `runs/analysis/summary.csv`, `report.md`, and the comparison plots. It reads existing run directories; it does not train anything.

For typing/linting:

```bash
ty check src/toy_jepa tests scripts
ruff check src/toy_jepa tests scripts
```

Do not run full training on CPU; use a tiny overridden `Config` for smoke tests.

## Results

| Arm | seed | loss | rank | var | cos | readout state MSE | pos MAE | vel MAE | R² |
|-----|------|------|------|-----|-----|-------------------|---------|---------|----|
| ... | ...  | ...  | ...  | ... | ... | ...               | ...     | ...     | ... |
