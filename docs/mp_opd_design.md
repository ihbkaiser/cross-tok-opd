# MP-OPD v0 design and implementation contract

## Status and evidence boundary

MP-OPD is exposed as a **KDFlow algorithm mode**, not as a long-lived server
branch:

```text
--kd_algorithm mp_opd
--mp_opd_mode atomic|fixed|random|oracle|soft|gbv|kernel|grass|grass_chunk
```

The isolated worktree used during development protects the completed SimCT and
X-Token evidence. It is not part of the runtime interface. Unit tests and the
toy oracle are labelled `implementation_validation` or `oracle_diagnostic`.
They do not establish an LLM improvement, paper reproduction, or novelty.

Distributional `span_ctkd` and projection-based `xtoken` remain unchanged.
`mp_opd atomic` is a scalar score-credit baseline that uses SimCT boundaries;
it must not be reported as the original distributional SimCT baseline.

## Formal correction to the proposal

The proposal defines `h_i` as a **sum** of token NLLs but its span-loss equation
multiplies those sums by token counts again. That double-counts length. MP-OPD
v0 uses the following single convention throughout.

For atom `a_i`, student prediction range `I_i`, teacher prediction range `J_i`:

```text
lT_i     = sum_(j in J_i) log p_T(t_j | teacher prefix)
lS_i_old = sum_(t in I_i) log p_theta_old(y_t | student prefix)
b_i      = stopgrad(lT_i - lS_i_old)
w_i      = |I_i| > 0
r_i      = b_i / w_i
h_i      = -sum_(t in I_i) log p_theta(y_t | student prefix)
```

For a candidate contiguous span `c=[i,j)`:

```text
B_c = sum_(k=i)^(j-1) b_k
W_c = sum_(k=i)^(j-1) w_k
r_c = B_c / W_c
ell_c = r_c * sum_(k=i)^(j-1) h_k
L_P = sum_(c in P) ell_c
```

There is no extra `w_i` inside `ell_c`. For span marginals `mu(c)`:

```text
rbar_i = sum_(c contains i) mu(c) r_c
L_soft = sum_i rbar_i h_i = sum_c mu(c) ell_c
sum_i w_i rbar_i = sum_i b_i
```

If all rates in a merged span are constant, loss and gradient are identical to
the atomic partition. If maximum span length is one, the exact result is
`sum_i (b_i/w_i) h_i`.

For outer gradient `v=grad F_M(theta)` and realized student path score:

```text
z_i = <v, grad_theta sum_(t in I_i) log p_theta(y_t | ...)>
U_c = <v, grad ell_c> = -r_c * sum_(i in c) z_i
```

The hard oracle maximizes `sum_c U_c`. Tests compare this prefix-sum form with
explicit autograd and check the sign using actual virtual updates. The proposal
needs an erratum in both its span-loss equation and its utility equation: remove
the second token-count factor in each.

## Tensor and detach contract

| Quantity | Shape | Gradient |
|---|---:|---|
| student logits at response predictors | `[S,V_s]` | student |
| teacher logits at response predictors | `[T,V_t]` | none |
| atom `b,w,r` | `[n]` | detached |
| current atom NLL `h` | `[n]` | student |
| atom features | `[n,10]` | detached |
| span energy | `[n,L]` | energy model only |
| span marginal | `[n,L]` | energy model only during meta step |
| rate used by real student loss | `[n]` | detached from energy |
| oracle utility | `[n,L]` | detached target |

Teacher parameters are frozen. The real student optimizer never owns energy
parameters. In `soft` student training, marginals are detached before weighting
`h_i`. A separate energy optimizer is serialized with a separate format,
configuration hash, and step. The one-step claim applies to virtual SGD in the
declared adapter subspace, not to a full Adam trajectory.

## Atomization and failure policy

`SimCTAtomizer` consumes response-only, shifted label IDs after the caller's
loss masks. It creates only minimal synchronized segments whose cumulative
decoded UTF-8 bytes are identical. Each atom records:

- stable sample ID;
- half-open student and teacher prediction ranges;
- half-open response byte interval;
- student/teacher token counts;
- one-to-one or multi-token boundary type;
- validity/failure reason; and
- optional detached scalar credit fields.

Atoms are ordered, non-overlapping and gap-free on covered events. A transient
replacement character from a token prefix that ends inside a multi-byte UTF-8
scalar is skipped as a candidate boundary; the complete decode must still be
replacement-free and byte-identical across tokenizers. The v0 atomizer fails
closed on a replacement character in the complete decode, normalization
mismatch, empty decode, unsupported added-token semantics, empty responses, or
an unaligned suffix. Padding never reaches the atomizer. Terminal EOS is masked
and counted: v0 does not pretend two unrelated EOS IDs decode to the same
response bytes.

An invalid sample contributes a differentiable zero loss and explicit failure
telemetry. This matters when `micro_train_batch_size=1`: an unrepresentable
teacher decode must be excluded without aborting the other valid samples in the
same gradient-accumulation window. The effective valid-sample count and each
failure reason are logged, so exclusion cannot be mistaken for training signal.

Prompt/completion ambiguity is prevented structurally: the API accepts only
labels selected by response loss masks. A caller that cannot prove that
boundary must not call MP-OPD.

## Partition modes

- `atomic`: every atom is a length-one span.
- `fixed`: deterministic length `mp_opd_fixed_span_length`; the tail may be
  shorter.
- `random`: a full-cover contiguous partition drawn from the documented
  seeded uniform-next-length procedure. This is a matched-capacity control.
- `oracle`: hard max-sum DP over detached utilities. Normal training fails
  closed unless instrumentation supplies per-atom directional scores.
- `soft`: a learned semi-Markov distribution. Normal training requires an
  audited energy checkpoint; random initialization is rejected.
- `gbv`: Gradient Bias-Variance Span. Each candidate span pays
  `D_g(c)/E_g + beta*(Q_c/W_c)/Q_0` — observed squared gradient distortion of
  pooling the span's rates, plus `beta` times the span's retained estimator
  degrees of freedom — and the exact minimum-cost full-cover partition is found by
  the same `O(nL)` dynamic program as `oracle`. No extra parameters, no auxiliary
  checkpoint. See the section below.
- `kernel`: **not a partition.** It leaves the partition at length one and re-weights
  the atomic credits with a cross-atom operator `A = K r`. `mp_opd_mode=atomic` is the
  same path with `K = I`. See the cross-atom credit section below.

No mode splits a SimCT atom.

## GBV-Span (`mp_opd_mode=gbv`)

Scale-free objective, from the method note *Gradient Bias Variance Adaptive Span
for MP OPD*:

```text
C_beta(c) = D_g(c)/E_g + beta * (Q_c/W_c)/Q_0
D_g(c)    = 0.5 * sum_{i in c} q_i (r_i - rbar_c)^2
E_g       = 0.5 * sum_i q_i (r_i - rbar)^2
Q_0       = sum_i q_i / w_i
```

`r_i = b_i/w_i` is the atomic credit rate, `w_i` the atom token mass and `q_i` the
atom logit sensitivity. Only the partition changes; the loss is still the hard
pooled loss, so signed-credit conservation is exact for every selected partition.

Atom sensitivity has two settings, and the difference is a claim, not a detail:

| `mp_opd_gbv_geometry` | `q_i` | Cost |
|---|---|---|
| `token_count` (default) | `w_i` | none; this is the weighted Potts special case (`Q_c/W_c = 1`, so the second term is `beta*|pi|/n`) |
| `exact_logit` | `sum_t (1 - 2 p_t(y_t) + sum_v p_t(v)^2)` | one chunked pass over the student logits per sample; no `autograd.grad` |

`beta` is a locked recipe knob (`mp_opd_gbv_beta`); the method note's pilot grid is
`{0.1, 0.3, 1.0, 3.0}` and it must be fixed before a primary multi-seed comparison.
`MP_OPD_GBV_VOCAB_CHUNK` (default 16384) is a memory knob for the exact pass only.

Numerical contract: cost tables are float64, invalid cells are `+inf` behind an
explicit validity mask, and if `E_g` falls below a relative floor the draw is
reported degenerate and the coarsest admissible tiling is returned instead of
dividing by a numerically-zero energy. Ties fall to the shortest admissible span.

Telemetry: `mp_opd_gbv_beta`, `mp_opd_gbv_exact_geometry`, `mp_opd_gbv_total_cost`,
`mp_opd_gbv_distortion_term`, `mp_opd_gbv_dof_term`,
`mp_opd_gbv_retained_dof_fraction`, `mp_opd_gbv_selected_span_count`,
`mp_opd_gbv_selected_span_length_mean`, `mp_opd_gbv_span_length_max`,
`mp_opd_gbv_span_{1,2,3,4}_fraction`, `mp_opd_gbv_boundary_strength_mean` (mean
absolute pooled-rate jump across selected boundaries) and `mp_opd_gbv_degenerate`.

Evidence boundary: the exact parts are the credit conservation, the selected-logit
geometry and the dynamic program. The token-noise covariance model, the claim that
the logit-space optimum is also a parameter-space optimum, and any claim that GBV
reduces training variance or beats fixed span 3 are **not** established by this
implementation and must not be reported as results.

## GRASS (`mp_opd_mode=grass`)

Gradient-Risk Adaptive Span Shrinkage, from the method note *GRASS: Gradient-Risk
Adaptive Span Shrinkage*. It keeps the SimCT atoms and the same `O(nL)` contiguous
dynamic program as GBV, and changes two things: **how a candidate span is scored**, and
**what happens inside the selected span**.

Scoring is the SURE estimate of a local update-MSE risk in the exact output-head
gradient geometry:

```text
D_c     = (r - rbar_c*1)^T H_c (r - rbar_c*1)          observed heterogeneity
V_c     = tr(H_c Sigma_c) = sigma^2*( sum_i H_ii/w_i - (1^T H_c 1)/W_c )
alpha_c = clip(V_c / D_c, 0, 1)                       SURE-optimal shrinkage strength
C(c)    = alpha_c^2 D_c - 2 alpha_c V_c               excess cost over a singleton
r~_i    = (1 - alpha_c) r_i + alpha_c rbar_c
```

The DP minimises `sum_pi C(pi)`. Because `tr(H_c Sigma_c)` is constant in `pi`, using
the excess cost leaves every singleton at exactly zero, so the selected partition never
has to pay a positive price for splitting. `H_c` is the local band of the exact atom
Gram `H_ij = <grad_W l_i, grad_W l_j>_F`, which is what makes `alpha` *estimated*
rather than a second hand-tuned trade-off knob like GBV's `beta`.

| `mp_opd_grass_geometry` | `H_c` | Cost |
|---|---|---|
| `exact_head` (default) | full local band of the output-head atom Gram | one chunked pass over the student logits and the head-input hidden states per sample; no `autograd.grad` |
| `diag` | only `diag(H_c)`; the off-diagonal band is zeroed | same pass; this is the section 15 GRASS-Diag ablation |

| Knob | Default | Meaning |
|---|---|---|
| `mp_opd_grass_sigma_rho` | 0.99 | bias-corrected EMA decay of the credit-noise scale |
| `mp_opd_grass_sigma_min_pairs` | 8 | valid adjacent atom pairs needed to update that scale |
| `mp_opd_grass_eps_d` | 0.0 | floor under `D_c` for the no-heterogeneity branch |
| `mp_opd_grass_negative_tol_rel` | 1e-6 | relative tolerance that turns a negative `D_c`/`V_c` into a counter instead of silence |
| `mp_opd_grass_shadow` | false | report what Atomic / Fixed-2 / Fixed-3 / GBV would have done on the same batch; never changes the update |

`MP_OPD_GRASS_ROW_CHUNK` (default 32) is a memory/latency knob for the Gram pass and
changes how `H` is computed, never which `H`. `MP_OPD_GRASS_SOFTCAP` overrides the
head's `final_logit_softcapping` when the model config is not reachable.

### The geometry is exact, including the softcapped head

`delta_t = m_t * (p_t - e_{y_t}) * (1 - (z_t/sc)^2)`, where `m_t` is the per-token unit
credit coefficient (1 in the current OPD loss) and the last factor is
`d/du [sc*tanh(u/sc)]` read straight off the returned logits — no second
`[tokens, vocab] x [vocab, hidden]` matmul through the head weight is needed. The `e_y`
half of `delta_t` is scaled by `s_t[y_t]`, **not** by `q_t[y_t]`; conflating the two puts
`p_t[y_t]` where `1` belongs and shifts every diagonal block by `-2 p_t[y_t] + 1`.
Getting this wrong is invisible in the Gram's positive definiteness and fatal to `alpha`.

The Gram is accumulated in fp32 with `torch.backends.cuda.matmul.allow_tf32` disabled
for the duration of the pass (restored afterwards, override with
`MP_OPD_GRASS_EXACT_FP32=0`): at ~1e-3 relative error TF32 is larger than the fp32
round-off `D_c` and `V_c` are supposed to resolve, and an exactly PSD head Gram starts
producing materially negative `D_c`.

### Numerical contract

Cost tables are float64, invalid cells are `+inf` behind an explicit validity mask, and
`V_c >= 0` holds by construction because `A_c Sigma_c = sigma^2 (diag(1/w) - 11^T/W)` is
PSD by Cauchy-Schwarz. A materially negative `D_c`/`V_c` is therefore an implementation
fault, not a tunable: it is counted and reported, never clipped away. Ties fall to the
shortest admissible span. The Gram is computed under `torch.no_grad()` — it is a
selector input, and letting it carry a graph would backpropagate the variance estimate
into the student.

### Credit conservation is asserted, not assumed

`sum_i w_i r~_i == sum_i b_i` is exact for every partition because the shrink only
redistributes a span's credit. The mode checks this on the float32 coefficients the
loss actually multiplies by, against `CREDIT_CONSERVATION_REL_TOL * max(total, 1)`, and
raises rather than training on a run that is not comparable with Atomic/Fixed/GBV. The
noise-scale state is checkpointed with the trainer state, so a resume does not restart
GRASS from a deflated `sigma^2` (the collapse-to-atomic failure signature).

### Evidence boundary

Implemented: the exact head Gram, the SURE statistics, the dynamic program, the soft
shrinkage, the conservation assertion, and the Tier-A/Tier-B scalar telemetry. **Not**
implemented: the Tier-C artifacts of the method note (`span_samples.jsonl`,
`matrix_samples_stepXXXX.npz`, full-Transformer Gram calibration) — `gram_local_blocks()`
and `grass_gram_diagnostics()` already return exactly the objects those files would
contain, so adding the writers needs no algorithm change.

Claimed but unproven here: that the local update-MSE surrogate predicts downstream
benchmark behaviour, that SURE stays unbiased after searching over candidate partitions,
and that output-head geometry is predictive of full-model dynamics. Report them as
limitations, not results.

## GRASS-Chunk (`mp_opd_mode=grass_chunk`)

Same statistic as GRASS, one hypothesis removed. GRASS-DP bundles two claims: that
risk-adaptive shrinkage helps, and that SURE-driven *boundary search* helps. A win for
GRASS over a pooled baseline cannot say which one carried it. GRASS-Chunk keeps the first
and deletes the second — the partition arrives from the upstream cross-tokenizer
alignment, and the SURE machinery only decides how hard to shrink inside each chunk.

```text
chunk    = upstream alignment (or a fixed-run baseline)
D_c, V_c = the GRASS closed forms, computed on chunk c
alpha_c  = clip(V_c / D_c, 0, 1)          singleton chunk => 0
r~_i     = r_i - alpha_c (r_i - rbar_c)   for i in chunk c
```

### What is deliberately absent

| removed | why |
|---|---|
| candidate-span enumeration | nothing to search over |
| `mp_opd_max_span_length` | a native chunk is used exactly as delivered; §6.3 measures its cost instead of truncating it |
| dynamic program | no partition to optimise |
| partition-selection cost | no selection means no selection bias and no SURE-after-search caveat |

`tests/mp_opd/test_grass_chunk.py` asserts these structurally: a chunk mode that quietly
grew a search back would still pass every numerical test.

### The gain is reported, never obeyed

`G_c = 2*alpha_c*V_c - alpha_c^2*D_c` is computed and logged
(`mp_opd_grass_chunk_sure_gain_*`) because it is the natural diagnostic, but §10 forbids
letting it accept, reject, split, merge or reorder a chunk. A test asserts that the only
function which decides the update never reads it.

### Chunk source

| `mp_opd_grass_chunk_source` | meaning |
|---|---|
| `xtoken` (default) | the native synchronized chunk from the audited aligner used by `kd_algorithm='xtoken'`, re-verified by digest here. The method's intended definition. Requires `xtoken_projection_path` + a 64-char `xtoken_projection_sha256`, or the run **fails closed**. |
| `run` | fixed runs of `mp_opd_grass_chunk_run_length` atoms. A baseline for hosts without an audited projection — **not** an alignment chunk. Every metric carries `mp_opd_grass_chunk_source = 0`. |

The aligner labels *tokens* and credit is carried by *atoms*, so a chunk boundary can fall
inside an atom. `mp_opd_grass_chunk_straddle` decides that case explicitly —
`singleton` (default) leaves the atom's credit untouched, `majority` joins it to the chunk
holding most of its tokens — and both cases are counted in
`mp_opd_grass_chunk_mapping_straddling_atoms` / `_unaligned_atoms` rather than smoothed
over. A chunk id that reappears later cannot form one group; it is split and counted in
`mp_opd_grass_chunk_mapping_noncontiguous_splits`.

### Reuse, not duplication

`GrassNoiseEstimator`, the exact head Gram, `grass_shrink` and the conservation assertion
all come from `_mp_opd_grass_span`. The per-chunk strengths are laid out in that module's
`[start, length - 1]` table shape, so GRASS-Chunk trains through the *same* shrink and the
*same* invariant rather than a second copy that could drift.

Two deliberate differences:

* **Within-chunk adjacency only** (§11). A pair crossing an alignment boundary is exactly
  where the latent credit may jump, so it must not inform `sigma^2`; the noise estimator
  takes a per-adjacent-pair group mask and `mp_opd_grass_chunk_noise_pairs_excluded`
  reports how many pairs were dropped.
* **Blockwise Gram** (§6.2). One band wide enough for the longest chunk would evaluate
  every atom pair within that distance regardless of chunk; computing chunk by chunk is
  the same exact routine on the chunk's own slice.

### Telemetry

§18.2–18.8, with `mp_opd_grass_chunk_boundary_*` for §18.7: within-chunk and cross-chunk
adjacent pairs are compared on absolute rate difference, variance-normalized difference
and neighbour gradient cosine. If those two populations are indistinguishable, the fixed
chunk structure is a weak inductive bias for credit shrinkage — worth knowing *before*
attributing a result to the method. §18.8 conservation is asserted on the float32
coefficients the loss multiplies by, exactly as in GRASS-DP.

With `mp_opd_grass_chunk_shadow`, `mp_opd_grass_chunk_shadow_*` reports hard chunk pooling
(`alpha = 1`, §15.2) and atomic on the same batches: same boundaries, different shrinkage.
That is the single most diagnostic comparison in the method note (§20).

### Evidence boundary

Implemented: the chunk projection, the straddle policies, the per-chunk exact Gram, the
shared SURE shrinkage and its conservation assertion, and the §18 telemetry.
**Not** implemented: exact full-Transformer gradient geometry, and a global risk optimum —
this is a local, chunk-wise rule and must be reported as such. Claimed but unproven: that
upstream alignment chunks are the right inductive bias for credit shrinkage at all. The
§18.7 diagnostics exist to test that claim rather than to assume it.

<a id="temporal-grass-chunk"></a>

## GRASS-Chunk Temporal (`mp_opd_mode=grass_chunk_temporal`)

The partition, D/V formula and running noise estimator are GRASS-Chunk's. Only
the applied shrinkage strength changes: `alpha_effective = s(t) * alpha_raw`,
where `s(t) = clip((t - start_step) / (end_step - start_step), 0, 1)`.
Defaults are **20 -> 160**, with **t the one-based optimizer update being
computed**. Updates 1-20 preserve atomic credits; update 90 applies half the
GRASS strength; update 160 and later apply the full strength. Noise EMA still
observes every valid response during warmup. A complete accumulation window
uses one t, independent of response count and microbatch count.

The node runner defaults this mode to `source=run`, `run_length=2`; the direct
KDFlow CLI keeps the shared source default, so pass `--mp_opd_grass_chunk_source
run` explicitly. Length 3 is also supported. Neither native `grass_chunk` nor
ordinary `grass_chunk/source=run` receives the temporal gate.

```text
--mp_opd_mode grass_chunk_temporal
--mp_opd_grass_chunk_source run
--mp_opd_grass_chunk_run_length 2
--mp_opd_grass_chunk_temporal_start_step 20
--mp_opd_grass_chunk_temporal_end_step 160
```

Node runner environment: `MP_GRASS_CHUNK_SOURCE=run`,
`MP_GRASS_CHUNK_RUN_LENGTH=2`, `MP_GRASS_CHUNK_TEMPORAL_START_STEP=20`, and
`MP_GRASS_CHUNK_TEMPORAL_END_STEP=160`. Use a fresh output and the same starting
SFT checkpoint for a comparison against ungated run2/run3.

Checkpoints retain completed `student_updates`, noise EMA, and the temporal
recipe (schedule, source, run length, straddle policy, step convention). Resume
rejects missing noise state or a missing/different temporal recipe rather than silently replaying the
warmup. Telemetry reports the upcoming update, s(t), raw/effective alpha; the
existing alpha, gain, credit-change and geometry diagnostics describe the
**effective** update. Weighted credit conservation holds at every s(t).
This is an implementation of a research hypothesis, not evidence of improved
benchmark accuracy. Existing conditional per-length metrics retain their old
aggregation behaviour; use raw/effective alpha fields for this ablation.

## Cross-atom credit operators (`mp_opd_mode=kernel`)

GBV replaced the atomic credit by a pooled one (`A = P r`). This mode keeps the atomic
scalar interface intact and instead re-weights it with an operator that pulls credit
*across* atoms:

```text
L_K = sum_i stopgrad(A_i) * NLL_i ,     A_i = (K r)_i
```

`mp_opd_mode=atomic` is this same code path with `K = I`, so no special-case atomic loss
remains. `b_i`, `w_i` and `r_i = b_i / w_i` are not redefined, and this mode selects no
span, runs no dynamic program, and reads no gradient.

### Two families, deliberately distinct

`mix` — a normalized redistribution of a fixed credit mass. Boundary fallback to atomic
is correct, and reachable-neighbour renormalization is allowed.

| `mp_opd_credit_transform` | operator |
|---|---|
| `identity` | `A_i = r_i` (Atomic) |
| `forward` | `(1-lam) r_i + lam r_{i+1}` |
| `backward` | `(1-lam) r_i + lam r_{i-1}` — matched anti-causal control |
| `shuffle` | `(1-lam) r_i + lam r_{pi(i)}`, `pi` a seeded bijection within the sequence |
| `causal_kernel` | static kernel `sum_k lambda_k r_{i + s*k}`; `uniform` or `exponential`, `direction` forward/backward/symmetric |

`future_return` — additive accumulation, **not implemented and not trained here**:
`A_i = r_i + lam r_{i+1} + lam^2 r_{i+2} + ...` is not normalized, so a short sequence
simply has fewer future terms. Do not call the normalized `causal_kernel` a future
return: the two have different scale behaviour and need separate namespaces.

### Mask and boundary rules

- A masked atom is neither a source nor a destination: it keeps its own atomic credit,
  and its neighbours fall back to atomic for that direction.
- No operator transfers credit across a sequence boundary; reachability requires the
  whole chain `i -> i+k` to stay inside one sequence and entirely on valid atoms.
- A kernel window truncated by a boundary is renormalized by the weights actually
  reachable, so boundary atoms fall back to atomic instead of shrinking toward zero.
- Terminal handling is unchanged: the masked EOS stays outside every atom.
- `symmetric` is the mean of the forward and backward kernels with identical weights,
  so the three directions carry the same kernel mass and the comparison isolates
  temporal direction rather than kernel mass.

### Scale discipline

Convex mixtures, permutation sources and normalized kernels are weighted means of the
same atomic credits, so they preserve the credit mean and cannot inflate its RMS.
`mp_opd_credit_scale_match=rms` rescales the additive `external` operator to the atomic
RMS, so an apparent effect cannot come from a larger credit or gradient magnitude.

### Knobs

`mp_opd_credit_{transform,lambda,convex,horizon,kernel,decay,direction,shuffle_seed,alpha,scale_match}`,
validated fail-closed in `kdflow/arguments/distillation_args.py`; `mp_opd_mode=atomic`
rejects a non-identity transform rather than silently ignoring it.

### Telemetry

Fifteen `mp_opd_credit_*` fields (transform code, valid atom count, mean/std/rms of the
atomic and the effective credit, correlation between them, mean absolute delta, transfer
fraction, sign-flip fraction, neighbour product mean, neighbour sign agreement, neighbour
pair count) plus the operator's own parameters. All values are finite by construction,
including empty and zero-variance draws, because the trainer aborts on a non-finite
metric.

### Regression gate

`MP_OPD_CREDIT_IDENTITY_CHECK=1` recomputes the historical pooled Atomic loss beside the
identity-operator loss for every sample and raises when the relative drift exceeds
`1e-5`, logging both the absolute and the relative drift.

### Evidence boundary

Phase 0 (identity equivalence on the credit path) and Phase 0.5 (the real Gemma<->Qwen
tokenizer mismatch, where `w_i > 1` atoms actually occur) are closed; the artifacts are
`CROSS_ATOM_CREDIT_STATUS.md` and `CROSS_ATOM_PHASE05_REPORT.md`. Operator loss
magnitudes are **not** evidence that any operator is better: they differ because the
operator attaches a different weight to the same `NLL`. Whether Atomic leaves useful
future credit on the table, and whether direction matters, require the same-prefix branch
probe and its gated follow-ups — none of which has run yet, and none of which may be
reported as a result before it does.

## Semi-Markov dynamic program

Candidate span `(i,j)` is valid when `1 <= j-i <= L`. Energies are laid out as
`energy[i,j-i-1]`. Invalid cells are `-inf`. The forward and backward recurrences
run in float32 log space:

```text
alpha[0] = 0
alpha[j] = logsumexp_i(alpha[i] + energy[i,j]/tau)
beta[n] = 0
beta[i] = logsumexp_j(energy[i,j]/tau + beta[j])
mu(i,j) = exp(alpha[i] + energy[i,j]/tau + beta[j] - alpha[n])
```

Time is `O(nL)` and marginal storage is `O(nL)`. Every atom must have marginal
coverage one. Partition entropy is `logZ - sum_c mu(c)s(c)/tau`; expected span
count is `sum_c mu(c)` and expected span length is total marginal length divided
by expected count. `n=0` returns the neutral empty partition. `n=1` and `L>n`
are handled geometrically. An all-invalid full-cover graph raises an error.

## Energy model and bilevel order

The v0 energy network is a two-layer bidirectional GRU followed by a span
scorer over start, end, prefix-pooled interior and normalized span length.
Inputs are detached atom features: rate, total credit, student count, teacher
count, byte length, teacher/student average log probability, boundary-type
indicators and validity. Reference/meta answers are never features.

The reusable model, checkpoint and semi-Markov mechanics are implemented. The
current KDFlow rollout batch does not carry an independent stable-ID meta batch,
so distributed `soft` training is deliberately fail-closed unless a separately
trained energy checkpoint is supplied. The next integration must add a B/M
loader with disjoint stable IDs, then perform:

1. freeze current adapter state;
2. compute detached atoms/features on rollout batch B;
3. compute `mu_phi` and expected virtual SGD update;
4. evaluate reference NLL on independent meta batch M;
5. form exact one-step hypergradient or
   `-eta sum_c mu(c) stopgrad(U_c)`;
6. step only the energy optimizer;
7. recompute/fix marginals and step the real student optimizer exactly once.

The toy runner validates this surrogate's directional derivative but is not a
substitute for the missing real-data B/M loader.

## Oracle falsification gate and data splits

`experiments/mp_opd/toy_oracle.py` uses disjoint stable fixture IDs for rollout,
meta-train, validation and test. It computes every candidate utility in an
adapter-only linear model, runs hard DP, enumerates all small partitions,
performs actual mutation-free virtual updates for atomic and oracle branches,
and reports predicted/actual headroom plus Spearman rank correlation.

The real gate must pin dataset and model revisions and record exact ID hashes.
Rollout remains student generated; M uses high-quality reference responses.
Benchmark test data cannot select partitions. Adapter-only headroom does not
prove full-parameter headroom.

## Metrics and evidence labels

Training emits scalar, finite, W&B-safe values for atom counts, valid/invalid
sample counts and ratio, per-reason exclusions, covered student/teacher events,
one-to-one/multi-token ratio, candidate span count, masked EOS count and `b/r`
distribution. Soft mode additionally
emits logZ, partition entropy, expected span count/length, marginal coverage
error and credit-conservation residual. Oracle instrumentation emits predicted
utility. The local JSON also records source identity, config hash, pinned toy
revisions, split IDs, seeds, timing and gate results.

Use exactly these evidence labels:

- `implementation_validation`: unit/integration consistency only;
- `oracle_diagnostic`: local headroom/falsification evidence only;
- `training_evidence`: only a terminal audited training run with finite metrics.

## Known v0 limitations

- Byte equality is established through exact cumulative decode. Tokenizers
  with context-dependent decode that cannot expose stable response bytes fail
  closed.
- EOS is masked, not learned as an endpoint atom.
- The current online trainer has no independent meta batch and stable record
  IDs, so learned-energy training is not yet an end-to-end KDFlow feature.
- Oracle computations are adapter/subspace diagnostics; they make no claim
  about full-parameter FSDP optimization.
- The toy positive headroom is an analytic fixture, not an LLM result.


## A/B/C real-data diagnostic (2026-09-09)

[Implementation and operating contract](mp_opd_abc_20260909.md) adds a standalone
HF real-data runner with disjoint B/M_select/M_eval, a zero-initialized low-rank
probe, actual functional virtual updates, matched-length/LR/norm/skip/SFT controls
and prequential learned atom weighting. It does not wire the production KDFlow
soft optimizer or validate full-parameter Adam. The existing toy harness remains
a mathematical fixture. No GPU result is implied by the new runner's existence.

## Alternating adapter pilot (2026-09-14)

The standalone real-data runner now supports `--alternating-student`. This is a
persistent adapter-B training loop, not an integration into the Ray/FSDP actor.
It uses exactly one virtual SGD step with a differentiable reference NLL to
update energy (exact hypergradient), recomputes marginals with updated energy,
detaches pooled rates, and applies exactly one real SGD step. The next HF
rollout uses this updated student. The existing diagnostic remains the default.

Use prepared disjoint rollout/select/eval groups from `real_oracle.py prepare`.
Prompt identity as well as record IDs is checked across all roles/groups.
Reference eval is measured after the real update and never used for gradients.
The caller must exclude benchmark test prompts during data preparation; the
loader cannot infer benchmark membership from arbitrary text.

Example, on an authorized remote GPU with existing local models/runtime:

```bash
python experiments/mp_opd/real_oracle.py run \
  --student /path/to/student --teacher /path/to/teacher \
  --data /path/to/disjoint-groups.json --output /path/to/new-pilot \
  --adapter-module model.layers.25.mlp.down_proj --rank 4 \
  --learn-partition --alternating-student --select-counts 4 \
  --energy-checkpoint /path/to/energy-select-4.pt \
  --energy-sha256 VERIFIED_SHA256 --max-span 2 \
  --virtual-lr 0.001 --energy-lr 0.001 --seed 42
```

The adapter module is model-specific: verify it on the target model before use.
The example LR is a pilot choice, not tuned or qualified for Gemma. Virtual and
real SGD use the same LR and unnormalized summed atom loss. Energy starts from
verified checkpoint weights with fresh AdamW moments at the requested LR. One
energy update occurs per valid group; diagnostic `--energy-steps` does not set
this schedule. Add `--freeze-energy` for the matched persistent-student frozen
control. Both use deterministic energy forwards with dropout disabled; this
does not disable the energy gradient. They are separate adapter-pilot groups,
not the full-parameter company campaign groups.

`latest.pt` now uses `mp-alternating-resume-v2`. It contains adapter A/B,
energy, energy AdamW moments/step, Python/NumPy/Torch CPU/all-CUDA RNG states,
the next group cursor, valid/invalid and energy update counters, committed
results/trajectories, and source/data/model/runtime provenance. The real
student is plain fixed-LR SGD without momentum, scheduler or GradScaler;
there is no additional optimizer state to reconstruct for that update.

Resume uses the same command and output directory, with `--resume` added.
All training/data/model/energy parameters must stay the same. `--stop-after-groups K`
pauses before absolute group K (including invalid groups); omit it when
resuming to the end. This is a controlled checkpoint-boundary pause, not a
change to the dataset or training budget. A finished run resumes without
another update. The initial state is checkpointed before the first rollout.
Invalid groups also checkpoint cursor and RNG. No automatic queue restart or
W&B upload is performed.

```bash
# Repeat the original full pilot command, changing only these operational flags:
# First segment: add --stop-after-groups 10
# Continuation: remove --stop-after-groups, add --resume; keep --output unchanged.
```

A checkpoint is the authoritative commit record. Write + fsync + atomic rename
precedes projection of its journals. Resume reconstructs missing/torn/excess
journal tails from the checkpoint, so a death before checkpoint commit replays
that group from previous RNG/state; a death after commit does not repeat it.
Only this pilot's results/trajectories/summary are reconstructed. An exclusive
POSIX flock prevents concurrent writers; the deployment filesystem must support
flock and atomic rename. An interrupted `.tmp` is never treated as committed.
State/provenance mismatch or legacy v1 checkpoint is rejected rather than
silently starting again. Load only trusted checkpoints (PyTorch pickle payload).

Runtime, implementation file hashes, model/data hashes and configuration are
checked before state restoration. Resume restores RNG after construction/load.
CPU tests establish bitwise agreement; CUDA kernel nondeterminism means a real
B200 canary is still required before claiming bitwise GPU reproducibility.
Strict provenance checks intentionally reject implementation changes mid-run.

The checkpoint remains adapter evidence, not an HF model directory. Direct
full-model evaluation queue ingestion and production Ray/FSDP resume are not
implemented; do not pass latest.pt as the model path to that queue.

CPU tests cover the exact hypergradient against finite differences, no virtual
student mutation, energy-before-student ordering, one persistent student step,
frozen control, ID overlap rejection, and a two-iteration runner with a tiny
mock model. A real-model B200 canary and Ray/FSDP integration remain unverified.

For a bounded alternating pilot, `--max-student-updates 50` stops after 50 valid
student updates; invalid groups only advance the data cursor. Prepare additional
disjoint groups as reserves. Exhausting data below the budget exits with an error
and `budget_unmet` summary, preserving the checkpoint. The budget is part of the
resume contract and cannot change mid-run. Budget completion resumes without
additional updates even when unprocessed reserve groups remain.

### Queued matched adapter follow-up

`experiments/runai/queue_alternating_followup.py submit --case CASE --manager MANAGER_SOURCE --state EXISTING_STATE --gpu-uuid GPU_UUID`
submits a success-dependent frozen50 -> paired-reference-evaluation -> report DAG
using job-manager on GPU0. An explicit state must already be running. Without
--state, it discovers the active local manager, rejects ambiguous multiple
managers, or creates a host-specific state when none is running. It does not
reconfigure an existing manager or cancel other jobs. Repeated submission checks exact specs and does
not duplicate jobs. The frozen payload reuses the original runner and command,
changing only output and freeze-energy; an existing valid checkpoint resumes.
The evaluator uses the same base model and adapter formula for initial,
alternating and frozen models, verifies matching training provenance, and only
uses eval references beyond both consumed cursors (including invalid groups).
If no unused groups remain it fails instead of evaluating on training groups.
Reports are in CASE/followup/report.json; NLL is mean per-reference token NLL
including EOS, not benchmark accuracy. No automatic long training promotion.
