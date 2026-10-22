# Understanding the metrics

This note explains what each metric measures, why it is useful, and how it can
mislead. The goal is to make the collapse dashboard readable from first
principles.

## The object being measured

For each sample, the encoder and fusion produce a feature vector

```text
c_t in R^d        (d = 128 in the current models)
```

Stack the test features into a matrix `Z` with shape `(N, d)`. A good
representation should:

1. vary across samples — not map everything to the same vector;
2. preserve information that is useful for the task — here, future state;
3. do so without relying on a readout that is so powerful it can compensate
   for a bad representation.

No single metric measures all three. The dashboard combines them.

## 1. Feature variance

For each feature dimension `j`, compute its variance across samples, then
average over dimensions:

```text
var = (1/d) * sum_j Var_i(z_ij)
```

Interpretation:

- Near zero means every sample has nearly the same feature vector: collapse.
- Larger means features move around across samples.
- It is **scale-dependent**. Compare arms only when features have the same
  normalization.

The encoder ends in LayerNorm, so individual feature vectors keep a roughly
fixed norm. Collapse therefore appears as **all vectors pointing in the same
direction**, not as vectors shrinking to zero. Variance and cosine catch that;
looking only at vector norms would not.

## 2. Effective rank

Center the features, take the singular values `sigma_i` of the centered
matrix, and compute the participation ratio:

```text
rank_eff = (sum_i sigma_i)^2 / sum_i sigma_i^2
```

Interpretation:

- `1` means all variance lies along one direction.
- `d` means variance is spread evenly across all `d` directions.
- It measures how many directions the **centered** feature cloud uses.

### The trap

Effective rank is scale-invariant. If the features are

```text
z_i = mu + epsilon_i
```

with a dominant common direction `mu` and tiny residual noise `epsilon_i`,
centering removes `mu`. The rank then describes `epsilon`, not `mu`.

A collapsed representation can therefore have:

- tiny variance,
- cosine close to 1,
- bad probes,

while still reporting a rank of 30 or 50 because the residual noise spans
many directions. This happened in M2: rank stayed near 49 while variance fell
by four orders of magnitude and cosine reached 0.99997.

**Use rank as a secondary diagnostic, never as the sole collapse metric.**
A useful companion is an **uncentered** rank or a **top-1 energy ratio**,
which detects the dominant common direction that centering removes.

## 3. Mean pairwise cosine

Normalize each feature vector to unit length, compute all pairwise cosine
similarities, and average the off-diagonal entries:

```text
cos = mean_{i != j} (z_i . z_j) / (||z_i|| ||z_j||)
```

Interpretation:

- `1` means all features point in the same direction.
- `0` means they are orthogonal.
- `-1` means opposite directions.

For LayerNormed features, this is one of the clearest collapse signals. A
value like `0.99997` means all samples are within roughly 0.4 degrees of one
another, even if the rank looks healthy.

## 4. State readout — the common performance metric

Collapse metrics describe the representation internally. They do not say
whether it is useful. The state readout supplies the external task metric.

Protocol:

1. Freeze the model.
2. Extract features from the **train** split.
3. Train a fixed-capacity readout to predict the future state at `t+k`.
4. Evaluate on the **test** split.

Use the same readout capacity, optimizer, training budget, and state
normalization for every arm.

The readout can be:

- **linear/ridge** — deterministic, cheap, and a lower bound on linearly
  decodable information;
- **MLP** — stronger, nonlinear, and closer to "what could a downstream head
  learn?"

Either way, it is trained on frozen features, so it does **not** shape the
representation. This is the key difference from the baseline's old state head,
which was trained end-to-end and therefore influenced the encoder.

### Input: `c_t` or `z_hat`?

- `c_t` is the context representation. A readout from `c_t` measures what the
  encoder/fusion learned.
- `z_hat` is the predicted future latent. A readout from `z_hat` measures the
  full model, including the predictor.

Report both when possible. If one must be primary, use `c_t` for
representation quality and `z_hat` for end-to-end prediction performance, and
label the table accordingly.

### Metrics from the readout

For normalized state targets:

```text
MSE   = mean((state_pred - state_true)^2)
MAE   = mean(|state_pred - state_true|)
R^2   = 1 - SSE / SST
```

Report position MAE in pixels and velocity MAE in pixels/frame, because those
units are interpretable. `R^2 = 1` is perfect, `0` is the mean predictor, and
negative is worse than predicting the mean.

Readout MSE depends on the state normalization. Keep the same normalization
(position `/ 36`, velocity `/ 3`) across every arm.

## 5. Prediction losses

### Pixel MSE

For the pixel baseline:

```text
pixel_mse = mean((pixel_pred - target_frame)^2)
```

Lower is better, but pixel MSE rewards blur: the squared-error optimum for an
uncertain future is the conditional mean, which looks fuzzy. Compare against
persistence, but do not treat low pixel MSE as proof of a useful
representation.

### Latent MSE and persistence

For JEPA variants:

```text
jepa_mse = mean((z_hat - z_target)^2)
```

This is not comparable to pixel MSE. It is also easy to make small by making
`z_target` small or nearly constant. Always log:

- `target_variance` — how much the target varies;
- `prediction_variance` — how much the prediction varies;
- `normalized_loss = jepa_mse / target_variance`.

A falling loss with falling target variance is not progress.

Latent persistence uses the current encoder:

```text
persistence = mean((f(x_t) - f(x_{t+k}))^2)
```

Because the encoder changes during training, this baseline moves too. A fixed
initial encoder or a random encoder is a more stable reference.

## 6. Reading the dashboard

Use this hierarchy:

1. **State readout** — does the representation predict the future state?
2. **Variance and cosine** — is the representation collapsing?
3. **Probes** — what information is linearly readable?
4. **Loss** — is optimization proceeding?
5. **Rank** — a secondary geometry diagnostic.

Decision rules:

- High variance, low cosine, low readout error: healthy and useful.
- Low variance, cosine near 1, high readout error: collapsed, even if rank is
  high.
- Low loss with bad readout: the loss is being reduced by shrinking the
  target or by a trivial solution.
- Low pixel MSE with blurry predictions: the generative objective is working
  as designed, but it is not evidence of a good representation.

The M2 run is the canonical example: rank looked healthy near 49, but variance
collapsed from 0.396 to 1.16e-5, cosine reached 0.99997, and position R² fell
to 0.008. Rank alone would have told the wrong story.

## 7. Protocol checklist

- Same train/val/test splits, `k`, and state normalization across arms.
- Same readout capacity, optimizer, steps, and early-stopping rule.
- Train readouts on train features; select on validation; never tune on test.
- Report readout MSE, position/velocity MAE, and R².
- Report variance, cosine, uncentered/top-1 rank, and target variance.
- Use multiple seeds before making fine-grained rankings.
- Do not compare pixel MSE with latent MSE.
