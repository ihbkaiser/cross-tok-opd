# MP-OPD locality and pooling diagnostics

The optional MP-OPD instrumentation is designed to test whether contiguous
atomic spans help because nearby credits are locally correlated, rather than
because changing the span length accidentally rescales the loss.

Enable it for a training run with:

```powershell
$env:MP_OPD_DIAGNOSTICS = "1"
$env:MP_OPD_DIAGNOSTICS_EVERY = "1"
$env:MP_OPD_DIAGNOSTICS_LOGIT_GRAD = "1"
```

`MP_OPD_DIAGNOSTICS_LOGIT_GRAD=1` performs extra autograd traversals. It
reports exact gradients with respect to the selected student logits, not the
full transformer parameter gradient after the model Jacobian.

The returned metrics are averaged over valid samples in the microbatch:

- `mp_opd_diag_lag1_corr`, `mp_opd_diag_lag2_corr`: correlation of atomic
  rates at adjacent offsets. Positive values support local credit coherence.
- `mp_opd_diag_lag1_abs_diff` and
  `mp_opd_diag_lag1_sign_flip_fraction`: local disagreement and sign changes.
- `mp_opd_diag_within_group_rate_rmse` and
  `mp_opd_diag_group_sign_flip_fraction`: heterogeneity hidden by the actual
  contiguous partition.
- `mp_opd_diag_pooled_abs_mass_ratio` and
  `mp_opd_diag_abs_mass_cancelled_fraction`: the absolute credit mass after
  pooling divided by the atomic absolute mass, and the corresponding lost
  fraction. Signed credit mass is conserved by construction, so this is the
  useful cancellation diagnostic.
- `mp_opd_diag_group_rate_std`: dispersion after pooling.
- `mp_opd_diag_shuffled_within_group_rate_rmse`: the same group lengths after
  randomly permuting atomic rates. If contiguous grouping is useful because of
  locality, the actual within-group RMSE should be lower than this control.
- `mp_opd_diag_actual_minus_shuffled_within_rmse`: negative values support the
  locality explanation; values near zero suggest that only group size matters.
- `mp_opd_diag_logit_grad_cosine`: cosine between the atomic and actual pooled
  loss gradients at the student-logit interface.
- `mp_opd_diag_logit_grad_delta_ratio` and
  `mp_opd_diag_logit_grad_norm_ratio`: direction and magnitude changes caused
  by pooling. The reported norms are normalized by the sample's valid student
  token count, matching the token-normalized MP-OPD loss scale.
- `mp_opd_diag_pooled_shuffled_logit_grad_cosine`: whether the actual pooled
  update is close to a same-size but non-local shuffled control.

For a fixed-span ladder, compare the distributions of these metrics across
the `m=1,2,3,4` runs using the same rollout and seed protocol. The expected
local-smoothing pattern is:

```text
m=2/3: positive lag correlation, lower actual-vs-shuffled RMSE,
       high atomic/pooled gradient cosine
m=4:   larger within-group RMSE or sign-flip fraction,
       lower cosine and/or a sharp norm change
```

The production loss and normalization are unchanged by these diagnostics.
