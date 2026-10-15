# Frozen hybrid design — OptLab blend v1

Status: design specified before hybrid test evaluation. This document, implementation,
optimizer YAMLs and experiment configs are committed together as Gate 2. The hash of that
commit is recorded in the subsequent run metadata. Changing the design after those runs
requires a new named version and another evaluation.

## Motivation and prior work

The baseline study showed noisy seed/fold performance and no Holm-significant optimizer
winner. Adaptive methods and momentum provide different update directions; slow-weight
averaging is a low-overhead stability component. SAM was costly and showed no clear local
benefit, so the proposed hybrid uses one gradient evaluation per step.

This is an exploratory engineering combination of established techniques. It is not a claim
of a new published optimization algorithm or guaranteed better deepfake detection.
Related work: [Adam](https://arxiv.org/abs/1412.6980),
[AdamW](https://arxiv.org/abs/1711.05101),
[Lookahead](https://arxiv.org/abs/1907.08610), and
[SWATS](https://arxiv.org/abs/1712.07628). Unlike SWATS, this design blends directions
rather than triggering a discrete optimizer switch.

## Exact update

For each parameter group independently, with gradient g:

1. `m = beta1*m + (1-beta1)*g`; `v = beta2*v + (1-beta2)*g²`.
2. Bias-correct by each parameter's gradient-update count: `mhat=m/(1-beta1^t)`,
   `vhat=v/(1-beta2^t)`.
3. Adam direction `a=mhat/(sqrt(vhat)+eps)`.
4. Group scalar `r=||a||₂/(||mhat||₂+eps)`, concatenating active parameters conceptually.
5. `d=blend*a + (1-blend)*r*mhat`.
6. Decoupled decay and update: `theta=(1-lr*wd)*theta-lr*d`.
7. Every k group updates, `slow += lookahead_alpha*(theta-slow)`; `theta=slow`.
   Slow parameters start at the initial parameters and are serialized with m/v/counts.

Constants: betas=(0.9,0.999), eps=1e-8, blend=0.5, k=5, lookahead_alpha=0.5.
Only learning rate and weight decay are tuned; learning rate is log-uniform [1e-5,1e-1],
weight decay has 50% probability zero and otherwise log-uniform [1e-6,1e-2].
This normalization matches direction norms, not a theoretical optimum. No convergence or
robustness guarantee is asserted. Sparse gradients are explicitly unsupported.

## Ablations

| Optimizer | Removed component |
|---|---|
| hybrid_blend | Full design |
| hybrid_no_preconditioning | blend=0: remove coordinate preconditioning, retain global RMS matching |
| hybrid_no_momentum | beta1=0: remove first-moment history |
| hybrid_no_lookahead | k=0: disable slow-weight interpolation |

Children are stored through `optimizers.parent_optimizer_id` and an ablation note. Each
ablation receives its own equal-budget tuning; selected hyperparameters are not copied
from the full hybrid.

## Frozen evaluation plan

- Unit checks: AdamW limiting case, manual first-step calculation, Lookahead interpolation,
  multi-group independence, zero gradients, checkpoint continuation and invalid arguments.
- Toy validation: five known-minimum functions, five seeds, common LR grid; 5,000-step cap,
  recorded under distinct `hybrid_*` benchmark names. Toy references remain Newton/L-BFGS.
- Detector study: 8 identity-disjoint folds, 5 seeds, 20 tuning trials × 10 epochs,
  30 final epochs, early-stopping patience 5, batch size 256. Same cached features,
  calibration, fusion, folds and seeds as the previous calibrated study.
- Comparators: AdamW, Lion and momentum SGD. Four hybrid/ablation variants yield 280 runs.
- Report seed-averaged video AUC by fold, paired fold differences with Holm correction,
  divergence and fit time. Do not revise the algorithm or search spaces after inspecting
  hybrid test metrics. A null or negative result is a valid completed phase.

Earlier TIMIT test results have already informed this project. Even with this frozen design,
new TIMIT runs are exploratory; FF++/Celeb-DF after access is the outstanding external check.
Standalone media predictions are a separate inference workflow, not independent evidence
of image/video accuracy. Images without audio must use a trained visual+frequency model.
