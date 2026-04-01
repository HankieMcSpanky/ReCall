"""DB Rotation — monthly database rotation with auto-consolidation.

Keeps the current month's DB fast and lean by archiving older months.

Layout:
    ~/.neuropack/
        memories.db              ← current (hot)
        archive/
            memories_2026_02.db  ← recent (searchable)
            memories_2026_01.db.zst  ← cold (compressed)

On rotation:
    1. Current DB renamed to archive/memories_YYYY_MM.db
    2. Auto-consolidation runs on the archived DB (merges similar memories)
    3. New empty DB created for current month
    4. Archives older than `compress_after_months` get zstd-compressed
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DBRotator:
    """Manages monthly DB rotation and archival."""

    def __init__(
        self,
        db_path: str,
        archive_dir: str | None = None,
        compress_after_months: int = 3,
        keep_months: int = 12,
    ) -> None:
        self.db_path = Path(os.path.expanduser(db_path))
        self.archive_dir = Path(archive_dir) if archive_dir else self.db_path.parent / "archive"
        self.compress_after_months = compress_after_months
        self.keep_months = keep_months

    def should_rotate(self) -> bool:
        """Check if rotation is needed (current DB is from a previous month)."""
        if not self.db_path.exists():
            return False

        db_mtime = datetime.fromtimestamp(self.db_path.stat().st_mtime)
        now = datetime.now()
        return (now.year, now.month) != (db_mtime.year, db_mtime.month)

    def rotate(self, consolidate: bool = True) -> str | None:
        """Rotate the current DB to archive.

        Returns the archive path, or None if no rotation needed.
        """
        if not self.db_path.exists():
            return None

        # Determine the month of the current DB
        db_mtime = datetime.fromtimestamp(self.db_path.stat().st_mtime)
        month_tag = db_mtime.strftime("%Y_%m")

        # Create archive dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Archive path
        archive_name = f"memories_{month_tag}.db"
        archive_path = self.archive_dir / archive_name

        if archive_path.exists():
            # Already rotated this month — skip
            logger.info("Archive %s already exists, skipping rotation", archive_path)
            return None

        # Move current DB to archive
        logger.info("Rotating %s -> %s", self.db_path, archive_path)

        # Copy WAL-mode DB files
        shutil.copy2(str(self.db_path), str(archive_path))
        wal_path = str(self.db_path) + "-wal"
        shm_path = str(self.db_path) + "-shm"
        if os.path.exists(wal_path):
            shutil.copy2(wal_path, str(archive_path) + "-wal")
        if os.path.exists(shm_path):
            shutil.copy2(shm_path, str(archive_path) + "-shm")

        # Clear current DB (delete and let store create fresh)
        self.db_path.unlink()
        if os.path.exists(wal_path):
            os.unlink(wal_path)
        if os.path.exists(shm_path):
            os.unlink(shm_path)

        # Run consolidation on archive
        if consolidate:
            self._consolidate_archive(archive_path)

        # Compress old archives
        self._compress_old_archives()

        # Prune very old archives
        self._prune_old_archives()

        return str(archive_path)

    def _consolidate_archive(self, archive_path: Path) -> None:
        """Run memory consolidation on an archived DB."""
        try:
            from neuropack.config import NeuropackConfig
            from neuropack.core.store import MemoryStore

            config = NeuropackConfig()
            config.db_path = str(archive_path)
            store = MemoryStore(config)
            store.initialize()
            result = store.consolidate()
            logger.info(
                "Consolidated archive %s: %d clusters, %d memories merged",
                archive_path.name,
                result.clusters_found,
                result.memories_consolidated,
            )
            store.close()
        except Exception:
            logger.debug("Consolidation failed for %s", archive_path, exc_info=True)

    def _compress_old_archives(self) -> None:
        """Compress archives older than compress_after_months with zstd."""
        if not self.archive_dir.exists():
            return

        try:
            import zstandard as zstd
        except ImportError:
            return  # zstd not available — skip compression

        now = datetime.now()

        for db_file in self.archive_dir.glob("memories_*.db"):
            # Parse month from filename
            try:
                name = db_file.stem  # memories_2026_02
                parts = name.replace("memories_", "").split("_")
                year, month = int(parts[0]), int(parts[1])
                age_months = (now.year - year) * 12 + (now.month - month)
            except (ValueError, IndexError):
                continue

            if age_months >= self.compress_after_months:
                zst_path = db_file.with_suffix(".db.zst")
                if zst_path.exists():
                    continue  # already compressed

                logger.info("Compressing %s", db_file.name)
                cctx = zstd.ZstdCompressor(level=10)
                with open(db_file, "rb") as f_in:
                    with open(zst_path, "wb") as f_out:
                        cctx.copy_stream(f_in, f_out)

                # Remove uncompressed + WAL files
                db_file.unlink()
                for suffix in ("-wal", "-shm"):
                    wal = Path(str(db_file) + suffix)
                    if wal.exists():
                        wal.unlink()

    def _prune_old_archives(self) -> None:
        """Remove archives older than keep_months (keep L3 summaries only)."""
        if not self.archive_dir.exists():
            return

        now = datetime.now()

        for archive_file in self.archive_dir.glob("memories_*"):
            try:
                name = archive_file.stem.replace(".db", "").replace("memories_", "")
                parts = name.split("_")
                year, month = int(parts[0]), int(parts[1])
                age_months = (now.year - year) * 12 + (now.month - month)
            except (ValueError, IndexError):
                continue

            if age_months > self.keep_months:
                logger.info("Pruning old archive %s (age: %d months)", archive_file.name, age_months)
                archive_file.unlink()

    def list_archives(self) -> list[dict]:
        """List all archived DBs with metadata."""
        archives = []
        if not self.archive_dir.exists():
            return archives

        for f in sorted(self.archive_dir.glob("memories_*")):
            compressed = f.suffix == ".zst" or str(f).endswith(".db.zst")
            size_mb = round(f.stat().st_size / (1024 * 1024), 2)
            archives.append({
                "file": f.name,
                "path": str(f),
                "size_mb": size_mb,
                "compressed": compressed,
            })
        return archives

    def search_archives(
        self, query: str, limit: int = 10,
    ) -> list[dict]:
        """Search across all uncompressed archives.

        For compressed archives, decompress to temp file first.
        """
        results = []

        if not self.archive_dir.exists():
            return results

        for db_file in sorted(self.archive_dir.glob("memories_*.db")):
            try:
                from neuropack.config import NeuropackConfig
                from neuropack.core.store import MemoryStore

                config = NeuropackConfig()
                config.db_path = str(db_file)
                store = MemoryStore(config)
                store.initialize()
                hits = store.recall(query=query, limit=limit)
                for hit in hits:
                    results.append({
                        "archive": db_file.name,
                        "id": hit.record.id,
                        "summary": hit.record.l3_abstract,
                        "score": round(hit.score, 4),
                        "tags": hit.record.tags,
                    })
                store.close()
            except Exception:
                logger.debug("Failed to search %s", db_file, exc_info=True)

        # Sort by score and limit
        results.sort(key=lambda x: -x["score"])
        return results[:limit]
