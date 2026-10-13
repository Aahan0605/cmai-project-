"""`optlab` command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DB = "results/results.db"
DATASET_ALIASES = {"ff++": "faceforensics_pp", "celeb-df": "celeb_df_v2"}


def cmd_init_db(args: argparse.Namespace) -> None:
    from optlab import config
    from optlab.db.connection import connect, init_schema
    from optlab.db.repository import seed_catalogues

    conn = connect(args.db)
    version = init_schema(conn, args.schema)
    toy = config.load_toy_experiment(args.toy_config)
    seed_catalogues(
        conn,
        config.load_optimizers(args.configs).values(),
        config.load_models(args.configs).values(),
        config.load_datasets(args.configs).values(),
        toy.benchmarks,
    )
    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ["optimizers", "model_configs", "datasets", "toy_benchmarks"]
    }
    conn.close()
    print(f"{args.db}: schema v{version}, catalogues {counts}")


def cmd_merge_db(args: argparse.Namespace) -> None:
    from optlab.db.merge import merge_databases

    added = merge_databases(args.inputs, args.out)
    print(f"{args.out}: merged {len(args.inputs)} databases, added {added}")


def cmd_toy(args: argparse.Namespace) -> None:
    from optlab import config
    from optlab.db.connection import connect
    from optlab.toy.runner import run_suite

    conn = connect(args.db)
    run_suite(
        conn,
        config.load_toy_experiment(args.config),
        config.load_optimizers(args.configs),
        force=args.force,
        only_benchmarks=set(args.benchmarks) if args.benchmarks else None,
        only_optimizers=set(args.optimizers) if args.optimizers else None,
    )
    conn.close()


def cmd_toy_report(args: argparse.Namespace) -> None:
    from optlab import config
    from optlab.db.connection import connect
    from optlab.toy.report import write_report

    conn = connect(args.db, readonly=True)
    write_report(conn, config.load_toy_experiment(args.config), args.out, args.summary)
    conn.close()
    print(f"wrote {args.summary} and tables/plots under {args.out}")


def cmd_prepare(args: argparse.Namespace) -> None:
    from collections import Counter

    from optlab import config
    from optlab.data import manifest as m
    from optlab.db.connection import connect
    from optlab.db.repository import id_of, upsert_dataset_split

    name = DATASET_ALIASES.get(args.dataset, args.dataset)
    ds = config.load_datasets(args.configs)[name]
    split_cfg = config.load_splits(args.configs)
    if name == "faceforensics_pp":
        rows = m.assign_splits(
            m.parse_faceforensics(args.root, ds.compression or "c23", ds.methods or []), split_cfg
        )
        seed = split_cfg.seed
    elif name == "celeb_df_v2":
        rows, seed = m.parse_celeb_df(args.root), None
    else:
        raise SystemExit(f"no parser for {name} yet (dataset layout not verified)")
    path = args.out / f"{name}.csv"
    m.write_manifest(rows, path)
    counts = Counter(r.split for r in rows)
    conn = connect(args.db)
    for split, n in counts.items():
        upsert_dataset_split(conn, id_of(conn, "datasets", name), split, path.as_posix(), n, seed)
    conn.commit()
    conn.close()
    print(f"{path}: {dict(counts)}")


def cmd_extract(args: argparse.Namespace) -> None:
    from optlab import config
    from optlab.data import extract, extractors, faces, preprocess

    cfg = config.load_extraction(args.configs)
    name = DATASET_ALIASES.get(args.dataset, args.dataset)
    manifest = args.manifest or Path("manifests") / f"{name}.csv"
    out = args.out / name
    if args.backbone in cfg.visual_backbones:
        spec = cfg.visual_backbones[args.backbone]
        stats = extract.extract_visual(
            manifest,
            args.root,
            out,
            cfg,
            detector=faces.MTCNNDetector(args.device),
            extractor=extractors.ClipVisualExtractor(
                args.backbone, spec["arch"], spec["pretrained"], args.device
            ),
            frequency=None
            if args.no_frequency
            else extractors.FrequencyExtractor(cfg.freq_size, cfg.freq_bins),
            read_frames=preprocess.read_sampled_frames,
        )
    elif args.backbone in cfg.audio_backbones:
        spec = cfg.audio_backbones[args.backbone]
        stats = extract.extract_audio(
            manifest,
            args.root,
            out,
            cfg,
            extractor=extractors.Wav2Vec2AudioExtractor(
                args.backbone, spec["hf_id"], cfg.audio_sample_rate, args.device
            ),
            load_audio=preprocess.load_audio,
        )
    else:
        raise SystemExit(f"unknown backbone {args.backbone!r}; see configs/data/extraction.yaml")
    print(f"{out}: {stats}")


def _experiment(args: argparse.Namespace, config_path: Path | None = None):
    from optlab import config
    from optlab.db.connection import connect
    from optlab.pipeline import Experiment

    path = config_path or Path(args.config or f"configs/experiments/{args.experiment}.yaml")
    conn = connect(args.db)
    return Experiment(
        conn,
        config.load_experiment(path),
        path,
        config.load_optimizers(args.configs),
        config.load_models(args.configs),
    )


def _experiment_of_run(args: argparse.Namespace):
    from optlab.db.connection import connect

    conn = connect(args.db)
    row = conn.execute("SELECT config_path FROM runs WHERE id = ?", (args.run_id,)).fetchone()
    conn.close()
    if row is None:
        raise SystemExit(f"no run with id {args.run_id}")
    return _experiment(args, Path(row[0]))


def cmd_tune(args: argparse.Namespace) -> None:
    exp = _experiment(args)
    best = exp.tune(args.optimizer)
    print(
        f"selected trial {best['trial_index']}: val AUC {best['val_auc']},"
        f" {best['hyperparams_json']}"
    )


def cmd_train(args: argparse.Namespace) -> None:
    exp = _experiment(args)
    run_id = exp.train_run(args.optimizer, args.seed, exp.tune(args.optimizer))
    print(f"run {run_id}")


def cmd_fuse(args: argparse.Namespace) -> None:
    _experiment_of_run(args).fuse(args.run_id, args.solver)


def cmd_evaluate(args: argparse.Namespace) -> None:
    _experiment_of_run(args).evaluate(args.run_id, final=args.final, force=args.force)


def cmd_sweep(args: argparse.Namespace) -> None:
    _experiment(args).sweep()


def cmd_stats(args: argparse.Namespace) -> None:
    from optlab.db.connection import connect
    from optlab.eval.report import write_stats

    conn = connect(args.db, readonly=True)
    out = write_stats(conn, args.experiment, args.out or Path("results") / args.experiment)
    conn.close()
    print(f"wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optlab")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="apply schema.sql and seed catalogues from configs")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--schema", default="backend/schema.sql")
    p.add_argument("--configs", default="configs")
    p.add_argument("--toy-config", default="configs/experiments/toy.yaml")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("merge-db", help="merge partial results databases")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=cmd_merge_db)

    p = sub.add_parser("prepare", help="identity-disjoint split manifests from a dataset copy")
    p.add_argument("--dataset", required=True, help="faceforensics_pp (ff++) or celeb_df_v2")
    p.add_argument("--root", required=True, type=Path, help="local dataset root")
    p.add_argument("--out", type=Path, default=Path("manifests"))
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--configs", default="configs")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("extract", help="face crops -> frozen-backbone feature cache")
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", required=True, help="clip_vit_b16 or wav2vec2_base")
    p.add_argument("--root", required=True, type=Path, help="local dataset root")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path, default=Path("features"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-frequency", action="store_true", help="skip the frequency stream")
    p.add_argument("--configs", default="configs")
    p.set_defaults(func=cmd_extract)

    def experiment_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--experiment", help="name of configs/experiments/<name>.yaml")
        p.add_argument("--config", help="explicit experiment config path")
        p.add_argument("--db", default=DEFAULT_DB)
        p.add_argument("--configs", default="configs")

    p = sub.add_parser("tune", help="equal-budget hyperparameter search for one optimizer")
    experiment_args(p)
    p.add_argument("--optimizer", required=True)
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("train", help="one seed at the selected hyperparameters (no-op if done)")
    experiment_args(p)
    p.add_argument("--optimizer", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("fuse", help="fit simplex fusion weights on val_fusion")
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument(
        "--solver",
        default="qp_cvxpy",
        choices=["qp_cvxpy", "kkt_closed_form", "projected_gradient"],
    )
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--configs", default="configs")
    p.set_defaults(func=cmd_fuse, config=None)

    p = sub.add_parser("evaluate", help="validation metrics; --final writes test rows once")
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument("--final", action="store_true", help="also evaluate the test splits")
    p.add_argument("--force", action="store_true", help="overwrite existing test rows (flagged)")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--configs", default="configs")
    p.set_defaults(func=cmd_evaluate, config=None)

    p = sub.add_parser("sweep", help="tune, train, fuse and evaluate everything (resumable)")
    experiment_args(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("stats", help="tables, pairwise tests and plots for an experiment")
    p.add_argument("--experiment", required=True)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("toy", help="run optimizers on the toy benchmarks (resumable)")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--config", default="configs/experiments/toy.yaml")
    p.add_argument("--configs", default="configs")
    p.add_argument("--benchmarks", nargs="*", help="subset of benchmark names (default: all)")
    p.add_argument("--optimizers", nargs="*", help="subset of optimizer names (default: all)")
    p.add_argument("--force", action="store_true", help="recompute runs already recorded")
    p.set_defaults(func=cmd_toy)

    p = sub.add_parser("toy-report", help="toy tables, plots and summary from the DB")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--config", default="configs/experiments/toy.yaml")
    p.add_argument("--out", type=Path, default=Path("results/toy"))
    p.add_argument("--summary", type=Path, default=Path("results/toy_summary.md"))
    p.set_defaults(func=cmd_toy_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    from optlab.db.repository import TestSplitAlreadyWritten

    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except TestSplitAlreadyWritten as e:
        raise SystemExit(f"error: {e}") from None
