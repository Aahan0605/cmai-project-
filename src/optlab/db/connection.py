"""SQLite connections, schema application and numbered migrations."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
LATEST_SCHEMA_VERSION = max([1] + [int(p.name.split("_")[0]) for p in MIGRATIONS_DIR.glob("*.sql")])


def require_current_schema(conn: sqlite3.Connection) -> None:
    version = schema_version(conn)
    if version != LATEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"results DB is at schema v{version}, code needs v{LATEST_SCHEMA_VERSION}; "
            "run `optlab init-db` with matching code to migrate"
        )


@contextmanager
def atomic(conn: sqlite3.Connection) -> Iterator[None]:
    """Rollback this operation on failure without committing the caller's transaction."""
    name = f"optlab_{uuid4().hex}"
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    else:
        conn.execute(f"RELEASE {name}")


def connect(path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def init_schema(
    conn: sqlite3.Connection, schema_path: str | Path, migrations_dir: Path = MIGRATIONS_DIR
) -> int:
    """Apply schema.sql to an empty database, then any newer `NNN_*.sql` migrations.

    Idempotent: an initialised database only receives migrations above its user_version.
    """
    if schema_version(conn) == 0:
        try:
            conn.executescript("BEGIN;\n" + Path(schema_path).read_text() + "\nCOMMIT;")
        except BaseException:
            conn.rollback()
            raise
    for path in sorted(migrations_dir.glob("*.sql")):
        match = re.match(r"(\d+)_", path.name)
        if match and int(match.group(1)) > schema_version(conn):
            try:
                conn.executescript(
                    "BEGIN;\n"
                    + path.read_text()
                    + f"\nPRAGMA user_version = {int(match.group(1))};\nCOMMIT;"
                )
            except BaseException:
                conn.rollback()
                raise
    conn.commit()
    return schema_version(conn)
