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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
