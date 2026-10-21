# toy-jepa

A minimal, from-scratch JAX implementation of JEPA on Moving MNIST.

The goal is understanding, not benchmark performance. By the end you should be able to explain what JEPA predicts, why it collapses, what stop-gradient and EMA actually change, and what SIGReg buys you — and you should have the JAX/PyTree/jit habits to build the policy version next.

## The question

Given a frame `x_t`, learn an encoder `f` and a predictor `g` such that

```
g(f(x_t)) ≈ f(x_{t+k})
```

without reconstructing pixels. Then ask: what does the representation capture, when does it collapse, and which fixes actually help?

The global-latent predictor here is the same object as JEPA-Policy's future token, minus actions and the shared transformer. The policy version adds an action chunk and a control task; the representation question is the same.

## Scope

**In scope (v1)**

- Moving MNIST, temporal latent prediction.
- One CNN encoder, one MLP predictor.
- Baseline A (required): pixel future predictor — same encoder/trunk, pixel output.
- Baseline B (optional): supervised state predictor — same encoder, predict `(x, y, vx, vy)`.
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
2. **Params are PyTrees.** A model is `init_params(rng, config) -> params` and `apply(params, x) -> y`. Params are nested dicts/tuples of arrays.
3. **Pure core, impure shell.** Model/loss/optimizer/train-step functions do not read files, plot, print, or touch global RNG. The scripts under `scripts/` handle seeding, logging, checkpointing, and plotting. The data boundary is `mnist_utils.py` (IDX I/O) and `data.py` (`generate_dataset`); plotting is in `viz.py`.
4. **One level of abstraction per function.** `linear`, `layer_norm`, `conv`, `encoder`, `predictor`, `jepa_loss`, `train_step`, `train_loop`. Each does one thing; no function both loads data and trains.
5. **JIT at the boundary.** `train_step` is a pure function wrapped in `jax.jit` at the script level. Use `jax.eval_shape` to check shapes and `jax.make_jaxpr` to inspect.
6. **Explicit randomness.** Pass `PRNGKey`s; use `jax.random.split`; no `np.random` inside pure functions.
7. **Tests.** Shape tests, a gradient check, an EMA update test, and an overfit-8-samples test for each model.

Shared aliases and dataset records live in `types.py`; constants live in `constants.py`. Import `types` relatively (`from .types import ...`) to avoid shadowing the standard-library module.

## Data

- Generated Moving MNIST: `(N, T, 64, 64)`, `N=10_000`, `T=20`.
- Split by sequence: 8000 train / 1000 val / 1000 test. Never split by frame.
- Pairs `(x_t, x_{t+k})`; `k=1` default, `k=4` ablation.
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

Record positions even when the digits overlap; the renderer composites with max, and the labels stay clean. These labels are for the probes and Baseline B; they are **not** extra JEPA losses. The JEPA target is still the future frame's latent.

## Models

### Encoder

3 conv layers: 32, 64, 128 channels; 3×3 kernels; stride 2; GroupNorm; GELU; global average pool; linear to `d=128`; final LayerNorm on `z`. Use the same encoder and predictor in every arm; the final LayerNorm is an architecture choice, not an anti-collapse fix. Measure collapse with rank/probes, not the loss.

### Predictor

2-layer MLP: `128 -> 256 -> 128`, LayerNorm + GELU. No dropout in v1.

### Baseline A — pixel future predictor

Same encoder and predictor trunk; a pixel head `128 -> 512 -> 4096` reshaped to `64×64`; MSE to `x_{t+k}`. It must model texture/background, not just the state. This is the generative contrast. The pixel head is larger than the latent head; that is inherent to the objective, so compare representation quality (probes/rank), not raw loss values.

### Baseline B (optional) — supervised state predictor

Same encoder; an 8-d head for `(x, y, vx, vy)` of both digits at `x_{t+k}` (or a 4-d head for one digit). Normalize state targets (z-score) for training and report probes in consistent units. This is an upper bound on how much state information the encoder can carry, not a JEPA method.

### JEPA variants

- **Barebones:** shared encoder for context and target, no stop-grad:
  `L = || g(f(x_t)) - f(x_{t+k}) ||²`.
- **Stop-grad:** `L = || g(f(x_t)) - sg(f(x_{t+k})) ||²`.
- **EMA:** `L = || g(f(x_t)) - sg(f_ema(x_{t+k})) ||²`, with
  `f_ema ← decay * f_ema + (1 - decay) * f`.
- **EMA + SIGReg:** `L = L_JEPA + λ * L_SIGReg(f(x_t))`.

Use separate loss functions for the variants rather than a pile of boolean flags; the differences should be visible in the code.

### SIGReg

Follow LeJEPA §3 for the exact statistic. The shape is:

```
φ̂_m(t) = (1/B) Σ_b exp(i t (z_b · u_m))
L_SIGReg = (1/M) Σ_m Σ_j w_j |φ̂_m(t_j) - exp(-t_j²/2)|²
```

where `u_m` are random unit directions, `t_j` is a quadrature grid, and `w_j` are quadrature weights. Apply it to the context embeddings `z_t = f(x_t)`; do not apply it to the EMA target. Start with `M=256`, 17 grid points in `[0, 3]`, and sweep `λ ∈ {0.01, 0.1, 1.0}`.

SIGReg is not guaranteed to help. In the supervised JEPA-Policy setting it hurt at weight 0.09. Here, in pure JEPA, it may help. Report what you see, including negative results.

## Metrics

At every evaluation:

- `loss`: latent MSE (or pixel MSE for Baseline A).
- `rank`: effective rank / participation ratio of centered features:
  `(Σ σ_i)² / Σ σ_i²`, with `σ_i` the singular values.
- `var`: mean per-dimension variance.
- `cos`: mean pairwise cosine similarity.
- `digit_acc`: kNN accuracy for digit identity (k=5, cosine).
- `pos_r2`, `vel_r2`: ridge regression R² for position and velocity.
- `pixel_mse`: for Baseline A.

The **collapse dashboard** is one figure with loss, rank, variance, and probe R²/accuracy on a shared x-axis. A lower loss with a lower rank/probe is the signature of collapse. Loss alone is not evidence of a good representation.

Compute probes and metrics in NumPy float64 after transferring features; training stays float32.

## Probes

Freeze the **online** encoder, extract features from **context frames** `x_t` for the train/test splits, then:

- **Position/velocity:** closed-form ridge regression; report test R².
- **Digit identity:** kNN (k=5, cosine) or nearest centroid; report accuracy.

The targets are per digit: flatten `position` and `velocity` from `(2, 2)` to 4-d, and treat `digit_label` as two 10-way labels. Report per-digit means. Use the same fixed probe set of context frames and the same probe protocol/feature budget for every arm. Include two references: **raw pixels** (the linear floor) and a **random encoder** (the untrained floor). If a trained arm cannot beat the random encoder, something is wrong. This is a JAX/NumPy exercise: `jnp.linalg.solve`, `jnp.argsort`, `jnp.linalg.svd`.

## Curriculum

| Module | Question | Build | Deliverable |
|---|---|---|---|
| M0 | Can I load, split, and overfit? | Data + MLP warm-up | Data grid, splits, overfit curve, tests |
| M1 | What does a pixel objective learn? | Baseline A (pixel future predictor) | Pixel MSE + probe metrics |
| M2 | What happens with no asymmetry? | Barebones JEPA | Collapse dashboard |
| M3 | Does stop-grad fix it? | JEPA + stop-grad | Rank/probe comparison |
| M4 | Does a moving target stabilize it? | JEPA + EMA | EMA decay sweep |
| M5 | Does an explicit regularizer help? | JEPA + SIGReg | λ sweep, rank/probe comparison |
| M6 | Is the effect real? | 3-seed ablations + report | Table, dashboard, write-up |

**Minimum path:** M0–M5 with one seed each, then the five core arms in M6 with three seeds. The ablations are optional; do them only after the core comparison is clean.

### M0 — Data + JAX warm-up

- Generate, split, and visualize Moving MNIST in memory; inspect a few frames, labels, positions, and velocities.
- Write a 2-layer MLP with manual AdamW, `jit`, `vmap`, and `grad`.
- Overfit 8 samples; gradient check; shape tests.
- **Done when:** the MLP overfits, the data splits are disjoint, and the labels/positions look right.

### M1 — Baseline A (pixel future predictor)

- CNN encoder + MLP predictor + pixel head.
- Train with pixel MSE; evaluate reconstruction and probes.
- **Done when:** the baseline trains, the probes run, and the collapse dashboard has its first column.

### M2 — Barebones JEPA

- Replace the pixel target with the encoder's latent of `x_{t+k}`.
- Shared encoder, no stop-grad.
- Train and measure collapse.
- **Done when:** you can show the loss/rank/probe curves and say whether collapse occurred.

### M3 — Stop-grad

- Add `jax.lax.stop_gradient` to the target.
- Compare rank/probes to M2.
- **Done when:** you can explain what stop-grad does to the gradient path, and whether it fixed collapse.

### M4 — EMA

- Add an EMA target encoder; update it outside the optimizer.
- Compare to M3; sweep `decay ∈ {0.99, 0.996, 0.999}`.
- **Done when:** the EMA target is stable and the comparison is in the dashboard.

### M5 — SIGReg

- Implement SIGReg; add it to the context embeddings.
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

1. Does the pixel baseline learn better probes than JEPA? For which factors?
2. What does barebones JEPA collapse to, and how does it show up in the metrics?
3. Does stop-grad prevent collapse? Does EMA? Does SIGReg?
4. Is a lower JEPA loss always a better representation? (No — show it.)
5. Which factors (digit, position, velocity) does each variant capture?
6. What is the cost of SIGReg in compute and probe quality?

## Experiments

Core comparison (3 seeds):

- Pixel baseline
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

Log to `runs/<arm>_<seed>/`: `config.json`, `metrics.csv`, `checkpoint.npz`, and `plots/`. Keep the logging in the script, not in the pure training functions.

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
  models.py        # init/apply for encoder, predictor, heads (pure)
  losses.py        # pixel, JEPA, SIGReg losses (pure)
  optim.py         # manual AdamW (pure)
  ema.py           # EMA update (pure)
  metrics.py       # rank, variance, cosine, probes (NumPy/JAX)
  train.py         # pure train_step
  viz.py           # plotting (impure)
scripts/
  visualize_data.py   # play a few samples (impure)
  train.py            # impure entry point
  evaluate.py         # impure entry point
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
```

`visualize_data.py` overlays a velocity arrow (scaled 5×) on each digit and prints its label, top-left position, and velocity in the footer. Training entry points will call `generate_dataset(SEED, NUM_SAMPLES)` directly; there is no dataset file to prepare first.

## Results

| Arm | seed | loss | rank | var | cos | digit acc | pos R² | vel R² |
|-----|------|------|------|-----|-----|-----------|--------|--------|
| ... | ...  | ...  | ...  | ... | ... | ...       | ...    | ...    |
