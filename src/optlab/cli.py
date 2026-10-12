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
    args = build_parser().parse_args(argv)
    args.func(args)
