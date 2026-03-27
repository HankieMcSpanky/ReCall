"""Backup and restore functionality for NeuroPack databases."""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def create_backup(db_path: str, backup_dir: str | None = None) -> str:
    """Create a timestamped backup of the database.

    Returns the path to the backup file.
    """
    src = Path(db_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Database not found: {src}")

    if backup_dir:
        dest_dir = Path(backup_dir).expanduser()
    else:
        dest_dir = src.parent / "backups"

    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_name = f"neuropack_backup_{timestamp}.db"
    dest = dest_dir / backup_name

    # Use SQLite's backup API for a consistent snapshot
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dest))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    # Write metadata alongside
    meta = {
        "source": str(src),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backup_file": str(dest),
    }
    meta_path = dest.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return str(dest)


def restore_backup(backup_path: str, db_path: str) -> None:
    """Restore a database from a backup file.

    Creates a safety backup of the current database before restoring.
    """
    src = Path(backup_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Backup not found: {src}")

    dest = Path(db_path).expanduser()

    # Safety backup of current database
    if dest.exists():
        safety = dest.with_suffix(".db.pre_restore")
        shutil.copy2(str(dest), str(safety))

    # Verify the backup is a valid SQLite database
    try:
        conn = sqlite3.connect(str(src))
        conn.execute("SELECT COUNT(*) FROM memories")
        conn.close()
    except sqlite3.Error as e:
        raise ValueError(f"Invalid backup file: {e}")

    # Restore using SQLite backup API
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dest))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()


def list_backups(db_path: str, backup_dir: str | None = None) -> list[dict]:
    """List available backups with metadata."""
    src = Path(db_path).expanduser()
    if backup_dir:
        search_dir = Path(backup_dir).expanduser()
    else:
        search_dir = src.parent / "backups"

    if not search_dir.exists():
        return []

    backups = []
    for f in sorted(search_dir.glob("neuropack_backup_*.db"), reverse=True):
        meta_path = f.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        size_mb = f.stat().st_size / (1024 * 1024)
        backups.append({
            "file": str(f),
            "size_mb": round(size_mb, 2),
            "timestamp": meta.get("timestamp", "unknown"),
        })

    return backups
