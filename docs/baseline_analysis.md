# Baseline analysis (Gate 1)

**Experiment:** `baseline_timit_v3` on OptLab-TIMIT ([dataset card](datasets/optlab_timit.md)).
**Code version:** `403c168` for the experiment and 42 of the 50 runs; the first 8 runs were
trained at `f7b8b4d`, which differs only in the fusion solver's fallback, not in training.
**Source of every number below:** the results database, via `optlab stats --experiment
baseline_timit_v3` ([full tables](../results/baseline_timit_v3/summary.md)) and direct queries
of the same tables. Nothing here was tuned or chosen after looking at test results.

## What was run

- 10 optimizers: SGD, SGD + momentum, NAG, RMSProp, Adam, AdamW, RAdam, AdaBelief, Lion, and
  SAM over SGD + momentum.
- Each optimizer had 20 random-search trials over its learning rate and weight decay (plus
  ρ for SAM). Trials ran 10 epochs on the validation-tune split, with seed 0.
- Each optimizer was then trained for 5 seeds at its selected hyperparameters: 50 runs, all
  completed, none diverged.
- Model: one small head per stream (CLIP ViT-B/16 face features, wav2vec 2.0 audio, frequency
  spectrum), combined by weights on the simplex fitted on the validation-fusion split.
- Data: 32 people, 22 for training, 6 for validation and 4 for testing, each with 10 real and
  10 fake clips. The tests are therefore 40 real vs 40 fake videos from 4 unseen people:
  higher-quality fakes (in-distribution) and lower-quality fakes (the "unseen generator" test).
- Total training time for the 50 runs: 85 s on a laptop CPU.

## Results

**Test AUC, video level, mean ± sd over 5 seeds.** 95% bootstrap CIs are in the summary.

| optimizer | HQ (in-distribution) | LQ (unseen fakes) | gap HQ − LQ |
|---|---|---|---|
| NAG | 0.773 ± 0.030 | 0.852 ± 0.031 | −0.079 |
| AdaBelief | 0.768 ± 0.030 | 0.866 ± 0.016 | −0.098 |
| AdamW | 0.767 ± 0.036 | 0.875 ± 0.027 | −0.108 |
| Adam | 0.766 ± 0.023 | 0.875 ± 0.007 | −0.109 |
| RMSProp | 0.762 ± 0.075 | 0.827 ± 0.086 | −0.065 |
| SGD + momentum | 0.757 ± 0.022 | 0.846 ± 0.024 | −0.088 |
| SAM (SGD + M) | 0.756 ± 0.061 | 0.812 ± 0.072 | −0.056 |
| Lion | 0.753 ± 0.034 | 0.855 ± 0.044 | −0.102 |
| RAdam | 0.748 ± 0.036 | 0.833 ± 0.027 | −0.085 |
| SGD | 0.707 ± 0.078 | 0.803 ± 0.094 | −0.096 |

### 1. No optimizer is distinguishable from any other

0 of 45 pairs are distinguishable on either test (a pair counts only if its bootstrap CI
excludes 0 *and* its Holm-adjusted Welch p < 0.05). On the in-distribution test, nine of the
ten optimizers lie within 0.025 AUC of each other. This is a null result and is reported as
one: with 5 seeds and only 4 test people, this experiment cannot rank optimizers by test AUC.

### 2. Plain SGD is the weakest, but not significantly so

SGD has the lowest mean on both tests (0.707 and 0.803) and the largest seed spread (sd 0.078
and 0.094). It is the only optimizer that never reaches the convergence target (0 of 5 seeds).
Its selected learning rate is 0.545, high in its search range, and the validation curve shows
AUC *falling* after epoch 2 at that rate. Early stopping keeps the best early epoch, which
explains the low and noisy result.

### 3. Lion converges fastest; the "adaptive is faster" hypothesis is only partly supported

Convergence target: validation AUC ≥ 0.869 (0.95 × the best mean peak).

| | seeds reaching target | median epochs |
|---|---|---|
| Lion | 5 / 5 | 3 |
| Adam, AdaBelief, RAdam | 3 / 5 each | 8, 16, 7 |
| SGD + momentum | 2 / 5 | 5 |
| NAG, AdamW, RMSProp, SAM | 1 / 5 each | 5, 9, 3, 5 |
| SGD | 0 / 5 | — |

Lion is clearly the most reliable at reaching the target quickly. Among the Adam family,
AdaBelief's median of 16 epochs is slower than SGD + momentum's 5. The pre-registered claim
"adaptive optimizers converge faster" therefore holds for Lion but not for adaptive methods
as a group.

### 4. The "unseen" fakes are easier, so the generalization-gap hypothesis can't be tested here

Every optimizer scores *higher* on the lower-quality fakes (gap −0.056 to −0.109): the 64 px
swap model leaves more visible artefacts than the 128 px one. The pre-registered hypothesis
(adaptive optimizers show a larger gap, SAM narrows it) assumed the shifted test is harder.
For the record, SAM has the smallest gap magnitude (−0.056) and Adam the largest (−0.109),
but their CIs overlap other optimizers' and the direction is opposite to the assumed one.
This dataset does not test generalization to *harder* unseen fakes.

### 5. The fusion weights do not measure how useful a stream is

This is the most important methodological finding.

- **Audio is provably uninformative.** Every audio clip appears once as real and once as fake
  with identical bytes, so an audio-only detector must score AUC 0.5. Measured: exactly
  0.5000 in all 50 runs on both validation splits.
- **The frequency stream is near chance:** mean validation AUC 0.506 and 0.489. All the signal
  is in the visual stream (mean validation AUC 0.84 and 0.79).
- **The uninformative streams still get large weights:** audio averages 0.354 (exactly 0 in
  only 27 of 50 runs) and frequency 0.374. Visual averages only 0.272.

The reason: the weights must be non-negative and sum to 1, and the visual head's logits are
often overconfident (on validation-fusion their standard deviation averages 3.8 and reaches
25 in some runs, against about 0.01 for audio and frequency). The only way the solver can
reduce that confidence is to move weight onto a stream whose logits are almost constant,
which scales down the visual logit. So the fitted weights mix **calibration** with **usefulness**, and they cannot be read
as stream importance. The constrained solver itself is correct: every solution has KKT
residual < 1e-6. 41 came from cvxpy and 9 from projected gradient with a Newton polish, which
took over where cvxpy hit its precision limit on these badly scaled logits.

### 6. Validation is optimistic

Mean fused video AUC is 0.840 on validation-tune but 0.756 on the in-distribution test. Both
contain 80 videos from 2 swap pairs (4 people), so most of the difference is likely the
particular people. Selecting hyperparameters on so few people is noisy.

## Where each optimizer is weak (input to the hybrid design)

| optimizer | weakness seen in this data |
|---|---|
| SGD | Unstable at its selected learning rate (validation AUC drops after epoch 2); lowest and noisiest results; never reaches the target |
| SGD + momentum, NAG | Reach the target in only 2/5 and 1/5 seeds; otherwise unremarkable |
| RMSProp | Large seed spread (sd 0.075 HQ, 0.086 LQ); oscillating validation curve |
| SAM | Large seed spread (sd 0.061 HQ, 0.072 LQ); about twice the cost per step; no measurable test benefit |
| Adam, AdamW, RAdam, AdaBelief | Best means on the unseen fakes, but slower than Lion to the target (AdaBelief median 16 epochs) |
| Lion | Fastest and most reliable to the target, with mid-table test AUC |

Because no difference is significant, these are weak signals. Any hybrid motivated by them
should be framed as *exploratory*.

## Recommendation before Phase 3 (needs a decision)

The data supports these next steps more than it supports designing a hybrid now:

1. **Fix the fusion formulation before relying on it.** Calibrate each stream's logits (for
   example temperature scaling on validation-fusion) before fitting simplex weights, or let
   the weights sum to something other than 1, so that weights can mean importance.
2. **Get more statistical power.** Use more seeds (cheap: 85 s for all 50 runs) and, above
   all, more test people. Leave-one-pair-out cross-validation over the 16 swap pairs would
   test every person, instead of 4.
3. **Find a harder unseen test.** The higher- to lower-quality shift made detection easier.
   A genuinely different generator is needed (FaceForensics++ / Celeb-DF once access is
   granted).

## History: superseded runs (kept in the database, not reported as results)

| experiment | problem found (from validation diagnostics) | fix |
|---|---|---|
| `baseline_timit` | Fakes used DeepfakeTIMIT's 16 kHz resampled audio, reals the 32 kHz originals; audio alone reached validation AUC up to 0.72 | Each fake gets its real twin's original audio |
| `baseline_timit_v2` | 11 never-swapped people were always real, so cues identifying them predicted the label; audio alone reached validation AUC 0.64 despite identical audio | Keep only the 32 swapped people |

Both problems were found from per-stream validation AUCs and fusion weights, not from test
results. The test rows of those runs exist in the database but played no part in the fixes.

## Limitations

- 5 seeds and 4 test people. With 40 + 40 test videos, EER moves in steps of 0.025.
- Frozen backbones; DeepfakeTIMIT's 2018 face swaps; frontal studio recordings.
- "Area under validation loss" in the summary is summed over each run's own length, which
  early stopping makes different per run, so it is not compared here.
