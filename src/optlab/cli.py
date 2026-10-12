"""`optlab` command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DB = "results/results.db"


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
