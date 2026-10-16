# Additional-phase research report

Generated from the results database on 2026-10-16 03:07 UTC.

The experimental hybrid combines Adam-style preconditioning, RMS-matched momentum
and Lookahead slow weights. The rule and ablations were frozen in commit `5155285`
before hybrid TIMIT test evaluation. It combines established methods; this work does
not establish algorithmic novelty. TIMIT was already observed during baseline work,
so all subsequent TIMIT results are exploratory.

## Main finding

Full-hybrid HQ mean video AUC is 0.6475; AdamW is 0.6738.
0 of 21 HQ optimizer pairs pass the paired CI and Holm criteria.
These observations do not establish improved hybrid detection or equivalence.

## Execution coverage

| Experiment | Model | Optimizer | Status | Runs |
|---|---|---|---|---|
| hybrid_timit_cv_f0 | clip_vit_b16__wav2vec2__freq__weighted_sum | adamw | completed | 5 |
| hybrid_timit_cv_f0 | clip_vit_b16__wav2vec2__freq__weighted_sum | lion | completed | 5 |
| hybrid_timit_cv_f0 | clip_vit_b16__wav2vec2__freq__weighted_sum | sgd_momentum | completed | 5 |
| hybrid_timit_cv_f0 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_blend | completed | 5 |
| hybrid_timit_cv_f0 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_lookahead | completed | 5 |
| hybrid_timit_cv_f0 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_momentum | completed | 5 |
| hybrid_timit_cv_f0 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_preconditioning | completed | 5 |
| hybrid_timit_cv_f1 | clip_vit_b16__wav2vec2__freq__weighted_sum | adamw | completed | 5 |
| hybrid_timit_cv_f1 | clip_vit_b16__wav2vec2__freq__weighted_sum | lion | completed | 5 |
| hybrid_timit_cv_f1 | clip_vit_b16__wav2vec2__freq__weighted_sum | sgd_momentum | completed | 5 |
| hybrid_timit_cv_f1 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_blend | completed | 5 |
| hybrid_timit_cv_f1 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_lookahead | completed | 5 |
| hybrid_timit_cv_f1 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_momentum | completed | 5 |
| hybrid_timit_cv_f1 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_preconditioning | completed | 5 |
| hybrid_timit_cv_f2 | clip_vit_b16__wav2vec2__freq__weighted_sum | adamw | completed | 5 |
| hybrid_timit_cv_f2 | clip_vit_b16__wav2vec2__freq__weighted_sum | lion | completed | 5 |
| hybrid_timit_cv_f2 | clip_vit_b16__wav2vec2__freq__weighted_sum | sgd_momentum | completed | 5 |
| hybrid_timit_cv_f2 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_blend | completed | 5 |
| hybrid_timit_cv_f2 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_lookahead | completed | 5 |
| hybrid_timit_cv_f2 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_momentum | completed | 5 |
| hybrid_timit_cv_f2 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_preconditioning | completed | 5 |
| hybrid_timit_cv_f3 | clip_vit_b16__wav2vec2__freq__weighted_sum | adamw | completed | 5 |
| hybrid_timit_cv_f3 | clip_vit_b16__wav2vec2__freq__weighted_sum | lion | completed | 5 |
| hybrid_timit_cv_f3 | clip_vit_b16__wav2vec2__freq__weighted_sum | sgd_momentum | completed | 5 |
| hybrid_timit_cv_f3 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_blend | completed | 5 |
| hybrid_timit_cv_f3 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_lookahead | completed | 5 |
| hybrid_timit_cv_f3 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_momentum | completed | 5 |
| hybrid_timit_cv_f3 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_preconditioning | completed | 5 |
| hybrid_timit_cv_f4 | clip_vit_b16__wav2vec2__freq__weighted_sum | adamw | completed | 5 |
| hybrid_timit_cv_f4 | clip_vit_b16__wav2vec2__freq__weighted_sum | lion | completed | 5 |
| hybrid_timit_cv_f4 | clip_vit_b16__wav2vec2__freq__weighted_sum | sgd_momentum | completed | 5 |
| hybrid_timit_cv_f4 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_blend | completed | 5 |
| hybrid_timit_cv_f4 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_lookahead | completed | 5 |
| hybrid_timit_cv_f4 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_momentum | completed | 5 |
| hybrid_timit_cv_f4 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_preconditioning | completed | 5 |
| hybrid_timit_cv_f5 | clip_vit_b16__wav2vec2__freq__weighted_sum | adamw | completed | 5 |
| hybrid_timit_cv_f5 | clip_vit_b16__wav2vec2__freq__weighted_sum | lion | completed | 5 |
| hybrid_timit_cv_f5 | clip_vit_b16__wav2vec2__freq__weighted_sum | sgd_momentum | completed | 5 |
| hybrid_timit_cv_f5 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_blend | completed | 5 |
| hybrid_timit_cv_f5 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_lookahead | completed | 5 |
| hybrid_timit_cv_f5 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_momentum | completed | 5 |
| hybrid_timit_cv_f5 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_preconditioning | completed | 5 |
| hybrid_timit_cv_f6 | clip_vit_b16__wav2vec2__freq__weighted_sum | adamw | completed | 5 |
| hybrid_timit_cv_f6 | clip_vit_b16__wav2vec2__freq__weighted_sum | lion | completed | 5 |
| hybrid_timit_cv_f6 | clip_vit_b16__wav2vec2__freq__weighted_sum | sgd_momentum | completed | 5 |
| hybrid_timit_cv_f6 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_blend | completed | 5 |
| hybrid_timit_cv_f6 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_lookahead | completed | 5 |
| hybrid_timit_cv_f6 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_momentum | completed | 5 |
| hybrid_timit_cv_f6 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_preconditioning | completed | 5 |
| finetune_timit | clip_vit_b16__raw_last0 | adamw | completed | 3 |
| finetune_timit | clip_vit_b16__raw_last0 | hybrid_blend | completed | 3 |
| finetune_timit | clip_vit_b16__raw_last1 | adamw | completed | 3 |
| finetune_timit | clip_vit_b16__raw_last1 | hybrid_blend | completed | 3 |
| hybrid_heads_weighted_sum | clip_vit_b16__freq__weighted_sum | adamw | completed | 5 |
| hybrid_heads_weighted_sum | clip_vit_b16__freq__weighted_sum | hybrid_blend | completed | 5 |
| hybrid_timit_cv_f7 | clip_vit_b16__wav2vec2__freq__weighted_sum | adamw | completed | 5 |
| hybrid_timit_cv_f7 | clip_vit_b16__wav2vec2__freq__weighted_sum | lion | completed | 5 |
| hybrid_timit_cv_f7 | clip_vit_b16__wav2vec2__freq__weighted_sum | sgd_momentum | completed | 5 |
| hybrid_timit_cv_f7 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_blend | completed | 5 |
| hybrid_timit_cv_f7 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_lookahead | completed | 5 |
| hybrid_timit_cv_f7 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_momentum | completed | 5 |
| hybrid_timit_cv_f7 | clip_vit_b16__wav2vec2__freq__weighted_sum | hybrid_no_preconditioning | completed | 5 |
| hybrid_heads_concat_mlp | clip_vit_b16__freq__concat_mlp | adamw | completed | 5 |
| hybrid_heads_concat_mlp | clip_vit_b16__freq__concat_mlp | hybrid_blend | completed | 5 |
| hybrid_heads_svm_head | clip_vit_b16__freq__svm_head | adamw | completed | 5 |
| hybrid_heads_svm_head | clip_vit_b16__freq__svm_head | hybrid_blend | completed | 5 |

## Equal-budget hybrid and component ablations

Eight identity-component folds cover all 32 people. Five seeds are averaged
within each fold before statistical comparison. Every optimizer gets 20 tuning
trials of 10 epochs and up to 30 final epochs, with the same stopping rule.
Confidence intervals resample fold means, not correlated frames.

### optlab_timit_hq (in_distribution)

| Optimizer | Mean video AUC | 95% bootstrap CI | Folds |
|---|---|---|---|
| lion | 0.6789 | 0.6099–0.7412 | 8 |
| adamw | 0.6738 | 0.6067–0.7340 | 8 |
| hybrid_no_lookahead | 0.6708 | 0.6070–0.7322 | 8 |
| sgd_momentum | 0.6626 | 0.6046–0.7250 | 8 |
| hybrid_blend | 0.6475 | 0.5780–0.7188 | 8 |
| hybrid_no_preconditioning | 0.6275 | 0.5841–0.6696 | 8 |
| hybrid_no_momentum | 0.5812 | 0.5507–0.6177 | 8 |

Paired fold comparisons involving the full hybrid. Holm adjustment
covers all optimizer pairs within this dataset/protocol.

| A | B | A−B AUC | Paired 95% CI | Holm p | Distinguishable |
|---|---|---|---|---|---|
| adamw | hybrid_blend | 0.0263 | -0.0188–0.0713 | 1.0000 | False |
| lion | hybrid_blend | 0.0314 | -0.0168–0.0853 | 1.0000 | False |
| sgd_momentum | hybrid_blend | 0.0152 | -0.0235–0.0542 | 1.0000 | False |
| hybrid_blend | hybrid_no_preconditioning | 0.0200 | -0.0352–0.0740 | 1.0000 | False |
| hybrid_blend | hybrid_no_momentum | 0.0663 | 0.0079–0.1324 | 1.0000 | False |
| hybrid_blend | hybrid_no_lookahead | -0.0233 | -0.0684–0.0184 | 1.0000 | False |

### optlab_timit_lq (cross_dataset)

| Optimizer | Mean video AUC | 95% bootstrap CI | Folds |
|---|---|---|---|
| adamw | 0.7431 | 0.6725–0.8164 | 8 |
| lion | 0.7408 | 0.6671–0.7992 | 8 |
| hybrid_no_lookahead | 0.7139 | 0.6396–0.7813 | 8 |
| sgd_momentum | 0.7001 | 0.6221–0.7803 | 8 |
| hybrid_blend | 0.6903 | 0.6037–0.7766 | 8 |
| hybrid_no_preconditioning | 0.6651 | 0.6083–0.7255 | 8 |
| hybrid_no_momentum | 0.6099 | 0.5705–0.6561 | 8 |

Paired fold comparisons involving the full hybrid. Holm adjustment
covers all optimizer pairs within this dataset/protocol.

| A | B | A−B AUC | Paired 95% CI | Holm p | Distinguishable |
|---|---|---|---|---|---|
| adamw | hybrid_blend | 0.0528 | 0.0013–0.1049 | 1.0000 | False |
| lion | hybrid_blend | 0.0504 | -0.0021–0.1105 | 1.0000 | False |
| sgd_momentum | hybrid_blend | 0.0098 | -0.0267–0.0494 | 1.0000 | False |
| hybrid_blend | hybrid_no_preconditioning | 0.0252 | -0.0350–0.0814 | 1.0000 | False |
| hybrid_blend | hybrid_no_momentum | 0.0804 | 0.0083–0.1563 | 1.0000 | False |
| hybrid_blend | hybrid_no_lookahead | -0.0236 | -0.0701–0.0194 | 1.0000 | False |

## Joint heads and limited backbone fine-tuning

Head variants use visual/frequency inputs, five seeds and the primary search
budget. The raw CLIP ablation uses one face frame per video, three seeds, three
epochs and a shared fixed LR of 1e-4; it is a smaller feasibility study. Frozen
prefix tokens stop before the final transformer block. The thawed model updates
that block, final layer norm and projection; controls freeze all three.
Models and datasets are kept separate. Seed intervals condition on one fixed
identity split and do not measure uncertainty over the population of identities.

### finetune_timit

| Model | Dataset | Protocol | Optimizer | Video AUC | 95% seed CI | Seeds |
|---|---|---|---|---|---|---|
| clip_vit_b16__raw_last0 | optlab_timit_hq | in_distribution | adamw | 0.6050 | 0.5488–0.7037 | 3 |
| clip_vit_b16__raw_last0 | optlab_timit_hq | in_distribution | hybrid_blend | 0.5258 | 0.4475–0.6325 | 3 |
| clip_vit_b16__raw_last1 | optlab_timit_hq | in_distribution | adamw | 0.8323 | 0.8138–0.8419 | 3 |
| clip_vit_b16__raw_last1 | optlab_timit_hq | in_distribution | hybrid_blend | 0.7708 | 0.7312–0.8381 | 3 |

### hybrid_heads_weighted_sum

| Model | Dataset | Protocol | Optimizer | Video AUC | 95% seed CI | Seeds |
|---|---|---|---|---|---|---|
| clip_vit_b16__freq__weighted_sum | optlab_timit_hq | in_distribution | adamw | 0.7631 | 0.7478–0.7790 | 5 |
| clip_vit_b16__freq__weighted_sum | optlab_timit_lq | cross_dataset | adamw | 0.8694 | 0.8546–0.8838 | 5 |
| clip_vit_b16__freq__weighted_sum | optlab_timit_hq | in_distribution | hybrid_blend | 0.7516 | 0.7358–0.7629 | 5 |
| clip_vit_b16__freq__weighted_sum | optlab_timit_lq | cross_dataset | hybrid_blend | 0.8610 | 0.8488–0.8742 | 5 |

### hybrid_heads_concat_mlp

| Model | Dataset | Protocol | Optimizer | Video AUC | 95% seed CI | Seeds |
|---|---|---|---|---|---|---|
| clip_vit_b16__freq__concat_mlp | optlab_timit_hq | in_distribution | adamw | 0.8054 | 0.7730–0.8377 | 5 |
| clip_vit_b16__freq__concat_mlp | optlab_timit_lq | cross_dataset | adamw | 0.8626 | 0.8352–0.8900 | 5 |
| clip_vit_b16__freq__concat_mlp | optlab_timit_hq | in_distribution | hybrid_blend | 0.8116 | 0.7715–0.8545 | 5 |
| clip_vit_b16__freq__concat_mlp | optlab_timit_lq | cross_dataset | hybrid_blend | 0.8750 | 0.8357–0.9143 | 5 |

### hybrid_heads_svm_head

| Model | Dataset | Protocol | Optimizer | Video AUC | 95% seed CI | Seeds |
|---|---|---|---|---|---|---|
| clip_vit_b16__freq__svm_head | optlab_timit_hq | in_distribution | adamw | 0.7400 | 0.6829–0.7966 | 5 |
| clip_vit_b16__freq__svm_head | optlab_timit_lq | cross_dataset | adamw | 0.8256 | 0.7915–0.8590 | 5 |
| clip_vit_b16__freq__svm_head | optlab_timit_hq | in_distribution | hybrid_blend | 0.7110 | 0.6447–0.7772 | 5 |
| clip_vit_b16__freq__svm_head | optlab_timit_lq | cross_dataset | hybrid_blend | 0.7502 | 0.6760–0.8245 | 5 |

## Toy validation

See [the toy report](hybrid_toy_summary.md) for all 215 runs, tolerance failures,
learning rates, conditioning curves and trajectories. Newton and L-BFGS are
references on convex/Rosenbrock tasks. Their iterations have different function
evaluation costs from first-order optimizer steps. Success on these functions
does not establish better deepfake generalization.

## Delivered software

- Full hybrid and three child optimizer ablations with checkpointable state.
- Calibrated weighted fusion, concatenated MLP and primal linear SVM heads.
- Paired frozen/final-block CLIP runner with verified parameter-update checks.
- Hybrid dashboard, model export and opt-in local image/video prediction.
- Explicit identity metadata adapter for licensed media and image collections.
- Schema v5, API/snapshot parity, bounded temporary uploads and atomic checkpoints.

## Limits and pending external evidence

HQ and LQ share the same manipulation source and identities; LQ is a degradation
test, not a harder unseen generator. FF++, Celeb-DF and FakeAVCeleb studies remain
blocked on author-granted dataset copies and real-layout verification. No results
are fabricated for those datasets. Standalone image accuracy is unmeasured until
an identity-labelled image corpus is supplied. A sigmoid output is not verified
proof of authenticity. Three-epoch final-block results do not establish full
end-to-end training performance. OpenCLIP emitted a QuickGELU/pretrained-tag mismatch
warning: the same configured ViT-B-16 implementation was used consistently, but
these results should not be treated as exact reference-CLIP reproduction. Changing
its activation requires fresh caches and a new experiment. Compare measured intervals;
a new combination alone provides no evidence that the hybrid is more optimal.

Protocols and commands: [research workflows](../docs/RESEARCH_WORKFLOWS.md).
Update rule: [frozen hybrid design](../docs/HYBRID_DESIGN.md).
