# toy-jepa

A small JAX study of world-model objectives on a friction-moving-MNIST world.

## Current question

Does a stop-gradient JEPA representation support future-state prediction better
than deterministic pixel prediction or a generative pixel model?

The three phase-2 arms share:

- a per-frame CNN and temporal transformer;
- variable context length `m`;
- a query-conditioned predictor for horizon `k` or the `FINAL` stop state;
- frozen-feature probes from `c_t` only.

| Arm | Prediction | Loss |
|---|---|---|
| Pixel-MSE | future frame | pixel MSE |
| Pixel-FM | future frame | rectified flow matching |
| JEPA | future frame embedding | latent MSE, stop-gradient target |

## Protocol

- Friction Moving MNIST from
  [`tiny-worlds`](../tiny-worlds), `T=200`.
- 10,000 sequences, split 8k / 1k / 1k.
- Contexts `m ∈ {2,4,8}`; held out `{3,6,12}`.
- Horizons `k ∈ {1,2,4,8,16,32}`; held out `{3,6,12,24,48}`.
- A `FINAL` query for the terminal frame/state.
- Five seeds by default.

Probes on frozen `c_t`:

- state at held-out horizons;
- final position at the stop;
- unordered digit pair;
- bounce before the target;
- feature variance, cosine, rank.

`z_hat` is not used for cross-arm analysis; it is JEPA-specific.

## Layout

```text
src/toy_jepa/
  config.py, data.py, transformer.py
  models.py, train.py, experiment.py
  features.py, probes.py, metrics.py, checkpoint.py
  cnn.py, mlp.py, norm.py, types.py
scripts/
  train_pixel.py, train_pixel_flow.py, train_jepa.py
  analyze.py, run_all.sh, visualize.py, sample_models.py
tests/
  test_data.py, test_models.py, test_train.py
  test_probes.py, test_metrics.py, test_checkpoint.py
```

## Run

The world package is external and must be a sibling:

```text
jepa-workspace/
  toy-jepa/
  tiny-worlds/
```

Set the MNIST directory if needed:

```bash
export TINY_WORLDS_MNIST_DIR="$HOME/data/mnist"
uv sync
```

Three GPUs, full parallel:

```bash
PIXEL_GPU=0 JEPA_GPU=1 FLOW_GPU=2 bash scripts/run_all.sh
```

One GPU, sequential:

```bash
PIXEL_GPU=0 JEPA_GPU=0 FLOW_GPU=0 SEEDS="0 1 2" bash scripts/run_all.sh
```

Set all three GPU variables equal for serial execution, or all three distinct
for parallel execution. Subset seeds with `SEEDS="0 1 2"`.

Outputs:

```text
runs/p2/<arm>/seed_<n>/
checkpoints/p2/<arm>/seed_<n>/params.pkl   # gitignored
runs/analysis/p2_report.md
runs/analysis/p2_comparison.png
```

## Docs

## Samples

After training, plot target / Pixel-MSE / Pixel-FM / JEPA-decoded futures:

```bash
uv run python scripts/sample_models.py
```

JEPA uses frozen `c_t` features and trains two post-hoc generative heads:
rectified-flow and VAE. The grid has **models as rows** and horizons as columns,
so left-to-right is forward in time:

```text
target
pixel-MSE
pixel-FM
JEPA + flow
JEPA + VAE
```

## Docs

- [Phase 1 report](docs/phase1.md) — historical constant-velocity study
- [Phase 2 protocol](docs/phase2.md)
