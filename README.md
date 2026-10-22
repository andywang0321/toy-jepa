# toy-jepa

A minimal, from-scratch JAX implementation of JEPA on Moving MNIST.

The goal is understanding, not benchmark performance. By the end you should be able to explain what JEPA predicts, why it collapses, what stop-gradient and EMA actually change, and what SIGReg buys you — and you should have the JAX/PyTree/jit habits to build the policy version next.

## The question

Given m=2 context frames and a fixed k=4 horizon, which training objective produces context features `c_t` that best support future-state prediction under a fixed-capacity readout? The readout is trained after the model on frozen features, so no auxiliary state loss shapes the representation.

Formally, with two context frames `(x_{t-1}, x_t)`, learn a frame encoder, a fusion module, and a predictor such that

```
c_t = fuse(f_frame(x_{t-1}), f_frame(x_t))
g(c_t) ≈ f_frame(x_{t+k})
```

without reconstructing pixels. Then ask: what does the representation capture, when does it collapse, and which fixes actually help?

The global-latent predictor here is the same object as JEPA-Policy's future token, minus actions and the shared transformer. The policy version adds an action chunk and a control task; the representation question is the same.

## Scope

**In scope (v1)**

- Moving MNIST, temporal latent prediction.
- One CNN encoder, one MLP predictor.
- Baseline (required): same encoder/trunk, pixel-only head; state quality is measured post hoc with a common MLP readout rather than a supervised head.
- JEPA variants: barebones, stop-grad, EMA, EMA + SIGReg.
- Probes: digit identity, position, velocity; collapse metrics.
- 3-seed comparison and a short write-up.

**Out of scope (v1)**

- Push-T, JEPA-Policy, ABC, planning, control.
- ViT/patch tokens, I-JEPA masking, VICReg, diffusion.
- SOTA results, large-scale training, hyperparameter search.

These are stretch goals, not requirements. The point is to finish a small, clean, falsifiable study.

## Rules of the game

1. **Vanilla JAX.** Only `jax`, `jax.numpy`, `numpy`, `matplotlib`, `pytest` (plus `tqdm` if you want). No Flax, Equinox, Haiku, or other model frameworks. No classes with methods (a frozen dataclass or NamedTuple for static config is fine).
2. **Params are PyTrees.** A model is `init_params(rng, config) -> params` and `apply(params, x) -> y`. Params are named records (`NamedTuple`) of arrays, defined in `types.py`.
3. **Pure core, impure shell.** Model/loss/optimizer/train-step functions do not read files, plot, print, or touch global RNG. The scripts under `scripts/` handle seeding, logging, checkpointing, and plotting. The data boundary is `mnist_utils.py` (IDX I/O) and `data.py` (`generate_dataset`); plotting is in `viz.py`.
4. **One level of abstraction per function.** `linear`, `layer_norm`, `conv`, `encoder`, `predictor`, `jepa_loss`, `train_step`, `train_loop`. Each does one thing; no function both loads data and trains.
5. **JIT at the boundary.** `train_step` is a pure function wrapped in `jax.jit` at the script level. Use `jax.eval_shape` to check shapes and `jax.make_jaxpr` to inspect.
6. **Explicit randomness.** Pass `PRNGKey`s; use `jax.random.split`; no `np.random` inside pure functions.
7. **Tests.** Shape tests, a gradient check, an EMA update test, and an overfit-8-samples test for each model.

Shared aliases and dataset records live in `types.py`; constants live in `constants.py`. Import `types` relatively (`from .types import ...`) to avoid shadowing the standard-library module.

## Data

- Generated Moving MNIST: `(N, T, 64, 64)`, `N=10_000`, `T=20`.
- Split by sequence: 8000 train / 1000 val / 1000 test. Never split by frame.
- Context `(x_{t-1}, x_t)`, target `x_{t+k}`; `k=4` default, `k=1` ablation. Two frames are needed to infer velocity.
- Normalize to `[0, 1]`; no augmentation in v1.
- Use a subset for fast iteration by calling `generate_dataset(SEED, 2000)`; scale up for final runs.

`generate_dataset(seed, num_samples)` is the only entry point. It loads MNIST from `MNIST_DIR`, generates a fixed in-memory pool, and returns a `MovingMNIST`; nothing is written to disk. Call it once at startup and reuse the result; it is not meant to be called per batch. Use the same seed for every arm so they all see the same pool.

The standard `mnist_test_seq.npy` contains frames only; there are no digit labels, positions, or velocities. The factor probes need those labels. Download the IDX files into `MNIST_DIR` (default `~/data/mnist`; at least the `train-*` pair, the `t10k-*` pair is optional). Expected names are `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, and optionally the `t10k-*` equivalents; the hyphenated `.gz` variants work too. `mnist_utils.load_mnist` parses them, and does not download anything.

Defaults live in `src/toy_jepa/constants.py`:

```python
MNIST_DIR = Path.home() / "data" / "mnist"
NUM_SAMPLES = 10_000
NUM_FRAMES = 20
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

Record positions even when the digits overlap; the renderer composites with max, and the labels stay clean. These labels are for the probes and the post-hoc state readout; they are **not** extra JEPA losses. The JEPA target is still the future frame's latent.

## Models

### Frame encoder

Per frame: 3 conv layers (32, 64, 128 channels; 3×3; stride 2; GroupNorm; GELU), then **flatten** the `8×8×128` feature map to preserve position, a linear projection to `d=128`, and a final LayerNorm. Global average pooling is deliberately avoided: it makes the representation translation-invariant, so the model cannot know where the digits are and the probes fail.

### Context fusion

Encode `x_{t-1}` and `x_t` with the shared frame encoder, concatenate the two 128-d features, and fuse with `256 -> 256 -> 128` (GELU, linear). The result is `c_t`, the context representation used by the predictor and probes.

### Predictor

2-layer MLP: `128 -> 256 -> 128`, LayerNorm + GELU. No dropout in v1.

### Baseline — pixel only + common state readout

Same frame encoder, fusion, and predictor trunk, with one output head:

- a pixel head `128 -> 512 -> 4096` reshaped to `64×64`, MSE to `x_{t+k}`.

Train with pixel MSE only. There is no supervised state head and no state loss. State quality is measured after training: freeze the model, extract `c_t` and `z_hat`, and fit the **common MLP readout** (`128 -> 256 -> 8`, GELU hidden, linear output) to predict the state at `t+k`. The readout sees frozen feature arrays, so it never backpropagates into the encoder. Normalize state targets with position `/ 36` and velocity `/ 3`; report MSE, position MAE in pixels, velocity MAE in pixels/frame, and R². Compare pixels against persistence.

### JEPA variants

- **Barebones:** shared frame encoder for context and target, no stop-grad:
  `L = || g(c_t) - f_frame(x_{t+k}) ||²`, where
  `c_t = fuse(f_frame(x_{t-1}), f_frame(x_t))`.
- **Stop-grad:** `L = || g(c_t) - sg(f_frame(x_{t+k})) ||²`.
- **EMA:** `L = || g(c_t) - sg(f_ema(x_{t+k})) ||²`, with
  `f_ema ← decay * f_ema + (1 - decay) * f`.
- **EMA + SIGReg:** `L = L_JEPA + λ * L_SIGReg(c_t)`.

Use separate loss functions for the variants rather than a pile of boolean flags; the differences should be visible in the code. All four arms share one runner (`src/toy_jepa/jepa_experiment.py`) and the exact M1 evaluation protocol: the same readout contexts/keys, the same frozen-feature MLP readout, and the same state normalization. `c_t` is the primary readout; `z_hat` is secondary.

### SIGReg

Follow LeJEPA §3 for the exact statistic. The shape is:

```
φ̂_m(t) = (1/B) Σ_b exp(i t (z_b · u_m))
L_SIGReg = (1/M) Σ_m Σ_j w_j |φ̂_m(t_j) - exp(-t_j²/2)|²
```

where `u_m` are random unit directions, `t_j` is a quadrature grid, and `w_j` are quadrature weights. Apply it to the context embeddings `c_t`; do not apply it to the EMA target. Start with `M=256`, 17 grid points in `[0, 3]`, and sweep `λ ∈ {0.01, 0.1, 1.0}`.

SIGReg is not guaranteed to help. In the supervised JEPA-Policy setting it hurt at weight 0.09. Here, in pure JEPA, it may help. Report what you see, including negative results.

## Metrics

At every evaluation:

- `loss`: pixel MSE for the baseline, or latent MSE for JEPA.
- `rank`: effective rank / participation ratio of centered features:
  `(Σ σ_i)² / Σ σ_i²`, with `σ_i` the singular values.
- `var`: mean per-dimension variance.
- `cos`: mean pairwise cosine similarity.
- `digit_acc`: kNN accuracy for digit identity (k=5, cosine).
- `pos_r2`, `vel_r2`: ridge regression R² for position and velocity.
- `pixel_mse`: pixel-head error for the baseline.
- `state_mse`, `position_mae`, `velocity_mae`, `r2`: frozen-feature MLP readout of the future state.

The **collapse dashboard** is one figure with loss/persistence, rank, variance, and cosine over training, plus final readout state MSE bars. A lower loss with a lower rank/probe is the signature of collapse. Loss alone is not evidence of a good representation.

Compute probes and metrics in NumPy float64 after transferring features; training stays float32.

See [docs/metrics.md](docs/metrics.md) for a first-principles guide to variance, rank, cosine, state readouts, and how to read the dashboard.

## Probes and state readout

The main external metric is the **common post-hoc MLP readout** (`src/toy_jepa/readout.py`): freeze the model, extract `c_t` and `z_hat` for the same large train/val/test context set in every arm (e.g. 6,000 / 2,000 / 2,000 contexts), train on train features with validation early stopping, and report test normalized state MSE, position MAE, velocity MAE, and R². Features stay in RAM; no `.npz` files. The ridge and kNN probes below are secondary diagnostics of linear readability.

Freeze the **online** encoder, extract context features `c_t` from `(x_{t-1}, x_t)` for the train/test splits, then:

- **Position/velocity:** closed-form ridge regression; report test R².
- **Digit identity:** kNN (k=5, cosine) or nearest centroid; report accuracy.

The targets are per digit: flatten `position` and `velocity` from `(2, 2)` to 4-d, and treat `digit_label` as two 10-way labels. Report per-digit means. Use the same fixed probe set of context frames and the same probe protocol/feature budget for every arm. Include two references: **raw pixels** (the linear floor) and a **random encoder** (the untrained floor). If a trained arm cannot beat the random encoder, something is wrong. This is a JAX/NumPy exercise: `jnp.linalg.solve`, `jnp.argsort`, `jnp.linalg.svd`.

## Curriculum

| Module | Question | Build | Deliverable |
|---|---|---|---|
| M0 | Can I load, split, and overfit? | Data + MLP warm-up | Data grid, splits, overfit curve, tests |
| M1 | What does a pixel objective learn? | Pixel-only baseline + common readout | Pixel/readout MSE + collapse metrics |
| M2 | What happens with no asymmetry? | Barebones JEPA | Collapse dashboard |
| M3 | Does stop-grad fix it? | JEPA + stop-grad | Rank/probe comparison |
| M4 | Does a moving target stabilize it? | JEPA + EMA | EMA decay sweep |
| M5 | Does an explicit regularizer help? | JEPA + SIGReg | λ sweep, rank/probe comparison |
| M6 | Is the effect real? | 3-seed ablations + report | Table, dashboard, write-up |

**Minimum path:** M0–M5 with one seed each, then the five core arms in M6 with three seeds. The ablations are optional; do them only after the core comparison is clean.

### M0 — Data + JAX warm-up

- Generate, split, and visualize Moving MNIST in memory; inspect a few frames, labels, positions, and velocities.
- Note: M0 predicts a single frame's state, so velocity is only memorized for 8 samples. M1 fixes this with two context frames.
- Write a 2-layer MLP with manual AdamW, `jit`, `vmap`, and `grad`.
- Overfit 8 samples; gradient check; shape tests.
- **Done when:** the MLP overfits, the data splits are disjoint, and the labels/positions look right.

### M1 — Pixel-only baseline + common readout

- Two-frame context `(x_{t-1}, x_t)`, target `x_{t+k}`, `k=4`.
- Frame encoder + context fusion + predictor + pixel head. No state head and no state loss.
- Train with pixel MSE and compare against persistence (`x_t`).
- Freeze the model; extract `c_t` and `z_hat` for a large train/val/test context set in RAM.
- Train the shared fixed-capacity readout (`128 -> 256 -> 8`, AdamW, validation early stopping) on each frozen feature set. Report test state MSE, position MAE, velocity MAE, and R² for both.
- Report variance, cosine, effective rank, and target variance; write the collapse dashboard.
- **Done when:** pixel MSE beats persistence, both readouts are reported, and the dashboard is written.

New M1 runs write to `runs/m1_pixel/`. The old `runs/m1_baseline/` directory is the historical two-headed run and must not be overwritten.

### M2 — Barebones JEPA

- Shared frame encoder for context and target, no stop-grad; `jepa_base_loss` intentionally lets gradients flow through the target.
- `src/toy_jepa/jepa_experiment.py` is the shared runner for M2–M5: `JEPAConfig` holds the variant, run directory, and shared constants; `run_jepa` trains, evaluates, and fits the common readout.
- `scripts/m2_jepa.py` sets `variant="base"` and writes `runs/m2_jepa/` (`config.json`, `metrics.csv`, `probes.csv`, `dashboard.png`).
- Dashboard: loss + latent persistence, effective rank, variance, cosine, and the final readout state MSE for `c_t` and `z_hat`.
- **Done when:** the loss/rank curves and final readout are logged and you can say whether collapse occurred.

### M3 — Stop-grad

- `jepa_stopgrad_loss` adds `jax.lax.stop_gradient` to the shared-encoder target; the model and data are otherwise identical to M2.
- `scripts/m3_stopgrad.py` writes `runs/m3_stopgrad/`.
- Compare rank, collapse metrics, and readout MSE to M2.
- **Done when:** you can explain what stop-grad does to the gradient path, and whether it fixed collapse.

### M4 — EMA

- `ema.py::ema_update` moves a `CNNParams` target toward the online encoder. `JEPAEMAState` pairs the online `TrainState` with the target encoder; `ema_train_step` updates the online model with AdamW, then EMA-updates the target outside the optimizer.
- `scripts/m4_ema.py` writes `runs/m4_ema/`; default decay is 0.99.
- Compare to M3; sweep `decay ∈ {0.99, 0.996, 0.999}`.
- **Done when:** the EMA target is stable and the comparison is in the dashboard.

### M5 — SIGReg

- `sigreg.py` implements the LeJEPA statistic on `c_t`; `jepa_ema_sigreg_loss` adds `weight * SIGReg(c_t)` to the EMA JEPA loss using a fixed key (`SEED + 5`).
- `scripts/m5_sigreg.py` writes `runs/m5_sigreg/`; default weight is 0.1.
- Sweep `λ`; compare "EMA + SIGReg" to "EMA".
- Optional: "SIGReg without EMA" to isolate the mechanism.
- **Done when:** you can say whether SIGReg improves rank/probes at some weight, and what it costs.

### M6 — Ablations + report

- 3 seeds per arm; `k=1` vs `k=4`; predictor depth; batch size.
- Final table and dashboard; 1–2 page write-up.
- **Done when:** the table answers the questions below with mean ± std.

## Collapse knobs

If barebones JEPA does not collapse, that is a result, not a failure. To probe the boundary:

- Increase predictor depth/width.
- Remove LayerNorm from the predictor or encoder.
- Increase learning rate and train longer.
- Reduce batch size.
- Use `k=1` (easier target).
- Try a residual predictor.

If it collapses immediately and never recovers, try the opposite. Record what happened; the real story is often messier than "barebones collapses, EMA fixes it."

## Questions the project should answer

1. Does the baseline learn better probes than JEPA? For which factors?
2. What does barebones JEPA collapse to, and how does it show up in the metrics?
3. Does stop-grad prevent collapse? Does EMA? Does SIGReg?
4. Is a lower JEPA loss always a better representation? (No — show it.)
5. Which factors (digit, position, velocity) does each variant capture?
6. What is the cost of SIGReg in compute and probe quality?

## Experiments

Core comparison (3 seeds):

- Baseline (pixel only) + common post-hoc readout
- Barebones JEPA
- Stop-grad
- EMA
- EMA + SIGReg

Ablations (optional, 1–3 seeds):

- `k = 1, 4`
- EMA decay
- SIGReg `λ`
- Predictor depth (1 vs 2 hidden layers)
- Train set size (2000 vs 8000 sequences)

Keep the encoder, predictor, batch size, and step count identical across arms. The only difference should be the target/loss.

## Compute and logging

Start with 2,000 train sequences, batch 128, and 5,000 steps; each run should be minutes on a GPU and tens of minutes on CPU. Final runs: 8,000 sequences, 20,000 steps, 3 seeds. Evaluate every 500 steps on a fixed 1,000-sample probe set; do the full test-set evaluation at the end.

Log to `runs/<arm>/`: `config.json`, `metrics.csv`, `probes.csv`, and the PNG plots. Keep the logging in the script, not in the pure training functions.

## JAX checklist

By the end, you should have used each of these at least once:

- `jax.tree_util.tree_map`, `tree_flatten`, `tree_leaves`; count parameters.
- `jax.jit` on `train_step`; compare eager vs jit; measure compile time.
- `jax.eval_shape` to check output shapes without computing.
- `jax.make_jaxpr` to inspect the train step.
- `jax.value_and_grad`; a gradient check against finite differences.
- `jax.vmap` for a per-sample loss; compare the mean to the batch loss.
- `jax.lax.stop_gradient`; a test that the target encoder gets zero gradient.
- `jax.random.split`, `fold_in`; explicit keys for SIGReg projections.
- `jax.lax.scan` for a short training loop or EMA rollout (stretch).
- `donate_argnums` on `train_step` and a memory measurement (stretch).
- `jax.debug.print` inside jit and `jax.disable_jit()` for debugging.

## Repo layout

```
src/toy_jepa/
  constants.py     # paths/config constants (outermost layer)
  types.py         # shared type aliases and dataset records
  mnist_utils.py   # MNIST IDX loading, I/O boundary
  data.py          # generate_dataset, in-memory pool
  models/
    mlp.py            # affine layers (pure)
    cnn.py            # conv blocks + CNN built from them (pure)
    norm.py           # group/layer norm (pure)
    baseline.py       # pixel-only baseline model (pure)
    jepa_base.py      # barebones JEPA model (pure)
  experiment.py       # shared batches, state normalization, collapse metrics
  readout.py          # frozen-feature MLP state readout
  ema.py              # EMA update for the target encoder (pure)
  sigreg.py           # SIGReg statistic (pure)
  jepa_experiment.py  # shared M2-M5 runner: train, evaluate, readout, log
  metrics.py          # rank, variance, cosine, probes, state readout metrics
  optim.py            # manual AdamW (pure)
  train.py            # generic pure train step + M1/JEPA losses + EMA step
  viz.py              # plotting (impure), including the collapse dashboard
scripts/
  m0_overfit.py       # M0 overfit check
  m1_pixel.py         # M1 pixel-only baseline + common readout
  m2_jepa.py          # M2 barebones JEPA
  m3_stopgrad.py      # M3 stop-grad JEPA
  m4_ema.py           # M4 EMA JEPA
  m5_sigreg.py        # M5 EMA + SIGReg JEPA
  visualize_data.py   # play a few samples (impure)
tests/
```

## Definition of done

- Five arms implemented with the same encoder/predictor.
- 3-seed results with mean ± std.
- Collapse dashboard for each arm.
- Short report with the questions answered.
- Core functions pure; tests pass; `train_step` jitted.

## Stretch goals

- Swap the CNN for a small ViT and predict patch embeddings (I-JEPA-style).
- Add spatial masking instead of temporal prediction.
- Add VICReg as a second regularizer.
- Move to Push-T and the JEPA-Policy extension.
- `lax.scan` training loop, `donate_argnums`, multi-device `pmap`/`shard_map`.

## Reading

- I-JEPA (Assran et al., 2023), §3.
- V-JEPA (Bardes et al., 2024), §3.
- LeJEPA (Balestriero & LeCun, 2025), §3 (SIGReg).
- BYOL (Grill et al., 2020) and SimSiam (Chen & He, 2021) for stop-grad/EMA.
- JEPA-Policy (Xu et al., 2026), §III, for the later extension.

## How to run

Download the MNIST IDX files into `~/data/mnist`, then:

```bash
uv sync
uv run python scripts/visualize_data.py  # generates a few samples in memory
uv run pytest -q

# Five core arms (each writes its own runs/<arm>/ directory)
uv run python scripts/m1_pixel.py
uv run python scripts/m2_jepa.py
uv run python scripts/m3_stopgrad.py
uv run python scripts/m4_ema.py
uv run python scripts/m5_sigreg.py
```

`visualize_data.py` overlays a velocity arrow (scaled 5×) on each digit and prints its label, top-left position, and velocity in the footer. Training entry points will call `generate_dataset(SEED, NUM_SAMPLES)` directly; there is no dataset file to prepare first.

## Results

| Arm | seed | loss | rank | var | cos | digit acc | pos R² | vel R² |
|-----|------|------|------|-----|-----|-----------|--------|--------|
| ... | ...  | ...  | ...  | ... | ... | ...       | ...    | ...    |
