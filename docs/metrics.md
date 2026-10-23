# Metrics

## Representation health

For test features `Z` with shape `(N, d)`:

- **Variance:** mean per-dimension variance. Near zero means all samples share
  one feature vector.
- **Cosine:** mean pairwise cosine similarity. Near 1 means all samples point
  in the same direction.
- **Effective rank:** `(Σσ)² / Σσ²` after centering. It measures the spread of
  the centered residual, not task usefulness.

The rank trap: a near-constant feature cloud can have a high centered rank
because the tiny residual noise spans many directions. Variance and cosine are
better collapse signals. Always pair geometry with a readout.

## State readout

The primary metric is a frozen-feature MLP:

```text
features -> 128 -> 256 -> 8
```

Train it on the train split, early-stop on validation, and report test:

- normalized state MSE;
- position MAE in pixels;
- velocity MAE in pixels/frame;
- per-dimension R².

The state targets are future state at `t+k`, normalized as position `/36` and
velocity `/3`. Use a fixed readout architecture, data split, and training
budget for every arm.

Read out from:

- `c_t`: context representation quality;
- `z_hat`: end-to-end prediction quality.

They answer different questions; report both.

## Losses

- **Pixel MSE:** only comparable within the pixel baseline.
- **Latent MSE:** only comparable within JEPA arms; normalize by target
  variance.

A low latent loss can mean the target collapsed. Log target variance and
prediction variance. M2 is the control: its loss is tiny and its readout is
the mean predictor.

## Reading the dashboard

1. State readout — usefulness.
2. Variance/cosine — collapse.
3. Loss — optimization.
4. Rank — secondary geometry.

Low variance with a good readout is compression. Low variance with a bad
readout is collapse.

## Protocol checklist

- Same splits, `k`, normalization, and readout budget across arms.
- Train readouts on train features; select on validation; never tune on test.
- Five seeds before ranking small differences.
- Do not compare pixel MSE with latent MSE.
