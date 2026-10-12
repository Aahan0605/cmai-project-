"""SQLite connections, schema application and numbered migrations."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def connect(path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
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
        conn.executescript(Path(schema_path).read_text())
    for path in sorted(migrations_dir.glob("*.sql")):
        match = re.match(r"(\d+)_", path.name)
        if match and int(match.group(1)) > schema_version(conn):
            conn.executescript(path.read_text())
            conn.execute(f"PRAGMA user_version = {int(match.group(1))}")
    conn.commit()
    return schema_version(conn)
