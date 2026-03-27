"""ProfileBuilder: queries memories, runs analyzer, caches results in DB."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from neuropack.profile.analyzer import DeveloperProfileAnalyzer
from neuropack.profile.models import PROFILE_SECTIONS, DeveloperProfile

logger = logging.getLogger(__name__)


class ProfileBuilder:
    """Builds and caches developer DNA profiles from memory store data."""

    def __init__(self, store) -> None:
        """Initialize with a MemoryStore reference.

        Args:
            store: A MemoryStore instance used to query memories and access the DB.
        """
        self._store = store
        self._analyzer = DeveloperProfileAnalyzer()

    def build(self, namespace: str | None = None) -> DeveloperProfile:
        """Build a developer profile, using cache if available.

        Returns cached profile if it exists and is not stale, otherwise
        performs a full build and caches the result.
        """
        cached = self.get_cached(namespace=namespace)
        if cached is not None:
            return cached
        return self.rebuild(namespace=namespace)

    def rebuild(self, namespace: str | None = None) -> DeveloperProfile:
        """Force a full rebuild of the developer profile.

        Queries all memories in the namespace, runs the analyzer, and
        caches each section in the developer_profile table.
        """
        ns = namespace or self._store.config.namespace

        # Query all memories for analysis
        memories = self._store.list(limit=10000, namespace=ns if ns != "default" else None)

        # Check minimum evidence threshold
        min_evidence = self._store.config.profile_min_evidence
        if len(memories) < min_evidence:
            logger.info(
                "Insufficient evidence for profile: %d memories (need %d)",
                len(memories), min_evidence,
            )
            profile = DeveloperProfile(
                last_updated=datetime.now(timezone.utc).isoformat(),
                confidence=0.0,
                evidence_count=len(memories),
            )
        else:
            profile = self._analyzer.analyze(memories)

        # Cache each section
        self._cache_profile(profile, ns)
        return profile

    def get_cached(self, namespace: str | None = None) -> DeveloperProfile | None:
        """Read the cached profile from the developer_profile table.

        Returns None if no cached profile exists.
        """
        ns = namespace or self._store.config.namespace
        conn = self._store._db.connect()

        rows = conn.execute(
            "SELECT section, data, evidence_count, updated_at FROM developer_profile WHERE namespace = ?",
            (ns,),
        ).fetchall()

        if not rows:
            return None

        sections: dict[str, object] = {}
        evidence_count = 0
        last_updated = ""

        for row in rows:
            d = dict(row)
            section_name = d["section"]
            try:
                section_data = json.loads(d["data"])
            except (json.JSONDecodeError, TypeError):
                section_data = d["data"]
            sections[section_name] = section_data
            evidence_count = max(evidence_count, d["evidence_count"])
            if d["updated_at"] > last_updated:
                last_updated = d["updated_at"]

        # Reconstruct DeveloperProfile from cached sections
        try:
            profile = DeveloperProfile(
                naming_conventions=sections.get("naming_conventions", {
                    "variable_style": "unknown", "class_style": "unknown",
                    "file_style": "unknown", "common_prefixes": [],
                }),
                architecture_patterns=sections.get("architecture_patterns", []),
                error_handling=sections.get("error_handling", {
                    "style": "unknown", "patterns": [], "examples": [],
                }),
                preferred_libraries=sections.get("preferred_libraries", {}),
                code_style=sections.get("code_style", {
                    "line_length_pref": "unknown", "import_style": "unknown",
                    "docstring_style": "unknown", "type_hints_usage": "unknown",
                }),
                review_feedback=sections.get("review_feedback", []),
                anti_patterns=sections.get("anti_patterns", []),
                last_updated=last_updated,
                confidence=sections.get("_meta", {}).get("confidence", 0.0) if isinstance(sections.get("_meta"), dict) else 0.0,
                evidence_count=evidence_count,
            )
        except Exception:
            logger.warning("Failed to reconstruct cached profile, returning None")
            return None

        return profile

    def invalidate(self, namespace: str | None = None) -> None:
        """Mark the cached profile as stale by deleting it."""
        ns = namespace or self._store.config.namespace
        with self._store._db.transaction() as conn:
            conn.execute(
                "DELETE FROM developer_profile WHERE namespace = ?",
                (ns,),
            )

    def query_section(self, section: str, namespace: str | None = None) -> dict:
        """Return a specific section of the profile.

        Builds the profile first if not cached.
        """
        if section not in PROFILE_SECTIONS:
            return {"error": f"Unknown section: {section}. Valid: {', '.join(PROFILE_SECTIONS)}"}

        profile = self.build(namespace=namespace)
        section_data = profile.get_section(section)

        return {
            "section": section,
            "data": section_data,
            "confidence": profile.confidence,
            "evidence_count": profile.evidence_count,
            "last_updated": profile.last_updated,
        }

    def _cache_profile(self, profile: DeveloperProfile, namespace: str) -> None:
        """Persist all profile sections to the developer_profile table."""
        now = datetime.now(timezone.utc).isoformat()
        profile_dict = profile.to_dict()

        with self._store._db.transaction() as conn:
            # Clear existing cached sections for this namespace
            conn.execute(
                "DELETE FROM developer_profile WHERE namespace = ?",
                (namespace,),
            )

            # Insert each section
            for section_name in PROFILE_SECTIONS:
                section_data = profile_dict.get(section_name, {})
                conn.execute(
                    """INSERT INTO developer_profile (id, namespace, section, data, evidence_count, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        uuid.uuid4().hex,
                        namespace,
                        section_name,
                        json.dumps(section_data),
                        profile.evidence_count,
                        now,
                    ),
                )

            # Store metadata section for confidence
            conn.execute(
                """INSERT INTO developer_profile (id, namespace, section, data, evidence_count, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex,
                    namespace,
                    "_meta",
                    json.dumps({
                        "confidence": profile.confidence,
                        "last_updated": profile.last_updated,
                    }),
                    profile.evidence_count,
                    now,
                ),
            )
