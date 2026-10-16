"""Generate the additional-phase report from recorded results, without pooling models/datasets."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from optlab.export.queries import hybrid


def write_phase_report(conn: sqlite3.Connection, out: Path) -> None:
    data = hybrid(conn)
    hq = next(
        (
            t
            for t in (data["cross_validation"] or {}).get("tables", [])
            if t["dataset"] == "optlab_timit_hq"
        ),
        None,
    )
    finding = []
    if hq:
        means = {r["optimizer"]: r["auc_mean"] for r in hq["rows"]}
        if {"hybrid_blend", "adamw"} <= means.keys():
            finding = [
                "## Main finding",
                "",
                f"Full-hybrid HQ mean video AUC is {means['hybrid_blend']:.4f}; "
                f"AdamW is {means['adamw']:.4f}.",
                f"{sum(r['distinguishable'] for r in hq['pairwise'])} of "
                f"{len(hq['pairwise'])} HQ optimizer pairs pass the paired CI and Holm criteria.",
                "These observations do not establish improved hybrid detection or equivalence.",
                "",
            ]
    lines = [
        "# Additional-phase research report",
        "",
        f"Generated from the results database on {datetime.now(UTC):%Y-%m-%d %H:%M UTC}.",
        "",
        "The experimental hybrid combines Adam-style preconditioning, RMS-matched momentum",
        "and Lookahead slow weights. The rule and ablations were frozen in commit `5155285`",
        "before hybrid TIMIT test evaluation. It combines established methods; this work does",
        "not establish algorithmic novelty. TIMIT was already observed during baseline work,",
        "so all subsequent TIMIT results are exploratory.",
        "",
        "## Execution coverage",
        "",
        "| Experiment | Model | Optimizer | Status | Runs |",
        "|---|---|---|---|---|",
    ]
    lines[lines.index("## Execution coverage") : lines.index("## Execution coverage")] = finding
    for r in data["coverage"]:
        lines.append(
            f"| {r['experiment']} | {r['model']} | {r['optimizer']} | {r['status']} | {r['n']} |"
        )
    lines += [
        "",
        "## Equal-budget hybrid and component ablations",
        "",
        "Eight identity-component folds cover all 32 people. Five seeds are averaged",
        "within each fold before statistical comparison. Every optimizer gets 20 tuning",
        "trials of 10 epochs and up to 30 final epochs, with the same stopping rule.",
        "Confidence intervals resample fold means, not correlated frames.",
        "",
    ]
    for table in (data["cross_validation"] or {}).get("tables", []):
        lines += [
            f"### {table['dataset']} ({table['protocol']})",
            "",
            "| Optimizer | Mean video AUC | 95% bootstrap CI | Folds |",
            "|---|---|---|---|",
        ]
        for r in table["rows"]:
            lo, hi = r["auc_ci95"]
            lines.append(
                f"| {r['optimizer']} | {r['auc_mean']:.4f} | {lo:.4f}–{hi:.4f} | {r['n_folds']} |"
            )
        lines += [
            "",
            "Paired fold comparisons involving the full hybrid. Holm adjustment",
            "covers all optimizer pairs within this dataset/protocol.",
            "",
            "| A | B | A−B AUC | Paired 95% CI | Holm p | Distinguishable |",
            "|---|---|---|---|---|---|",
        ]
        for r in table["pairwise"]:
            if "hybrid_blend" not in (r["a"], r["b"]):
                continue
            lo, hi = r["ci95"]
            lines.append(
                f"| {r['a']} | {r['b']} | {r['mean_diff']:.4f} | "
                f"{lo:.4f}–{hi:.4f} | {r['p_holm']:.4f} | {r['distinguishable']} |"
            )
        lines += [""]
    lines += [
        "## Joint heads and limited backbone fine-tuning",
        "",
        "Head variants use visual/frequency inputs, five seeds and the primary search",
        "budget. The raw CLIP ablation uses one face frame per video, three seeds, three",
        "epochs and a shared fixed LR of 1e-4; it is a smaller feasibility study. Frozen",
        "prefix tokens stop before the final transformer block. The thawed model updates",
        "that block, final layer norm and projection; controls freeze all three.",
        "Models and datasets are kept separate. Seed intervals condition on one fixed",
        "identity split and do not measure uncertainty over the population of identities.",
        "",
    ]
    for study in data["studies"]:
        lines += [
            f"### {study['experiment']}",
            "",
            "| Model | Dataset | Protocol | Optimizer | Video AUC | 95% seed CI | Seeds |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in study["rows"]:
            lo, hi = r["auc_ci95"]
            lines.append(
                f"| {r['model']} | {r['dataset']} | {r['protocol']} | {r['optimizer']} | "
                f"{r['auc_mean']:.4f} | {lo:.4f}–{hi:.4f} | {r['n_seeds']} |"
            )
        lines += [""]
    lines += [
        "## Toy validation",
        "",
        "See [the toy report](hybrid_toy_summary.md) for all 215 runs, tolerance failures,",
        "learning rates, conditioning curves and trajectories. Newton and L-BFGS are",
        "references on convex/Rosenbrock tasks. Their iterations have different function",
        "evaluation costs from first-order optimizer steps. Success on these functions",
        "does not establish better deepfake generalization.",
        "",
        "## Delivered software",
        "",
        "- Full hybrid and three child optimizer ablations with checkpointable state.",
        "- Calibrated weighted fusion, concatenated MLP and primal linear SVM heads.",
        "- Paired frozen/final-block CLIP runner with verified parameter-update checks.",
        "- Hybrid dashboard, model export and opt-in local image/video prediction.",
        "- Explicit identity metadata adapter for licensed media and image collections.",
        "- Schema v5, API/snapshot parity, bounded temporary uploads and atomic checkpoints.",
        "",
        "## Limits and pending external evidence",
        "",
        "HQ and LQ share the same manipulation source and identities; LQ is a degradation",
        "test, not a harder unseen generator. FF++, Celeb-DF and FakeAVCeleb studies remain",
        "blocked on author-granted dataset copies and real-layout verification. No results",
        "are fabricated for those datasets. Standalone image accuracy is unmeasured until",
        "an identity-labelled image corpus is supplied. A sigmoid output is not verified",
        "proof of authenticity. Three-epoch final-block results do not establish full",
        "end-to-end training performance. OpenCLIP emitted a QuickGELU/pretrained-tag mismatch",
        "warning: the same configured ViT-B-16 implementation was used consistently, but",
        "these results should not be treated as exact reference-CLIP reproduction. Changing",
        "its activation requires fresh caches and a new experiment. Compare measured intervals;",
        "a new combination alone provides no evidence that the hybrid is more optimal.",
        "",
        "Protocols and commands: [research workflows](../docs/RESEARCH_WORKFLOWS.md).",
        "Update rule: [frozen hybrid design](../docs/HYBRID_DESIGN.md).",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
