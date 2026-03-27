from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Valid profile section names
PROFILE_SECTIONS = (
    "naming_conventions",
    "architecture_patterns",
    "error_handling",
    "preferred_libraries",
    "code_style",
    "review_feedback",
    "anti_patterns",
)


@dataclass(frozen=True, slots=True)
class DeveloperProfile:
    """Full developer DNA profile built from memory analysis."""
    naming_conventions: dict = field(default_factory=lambda: {
        "variable_style": "unknown",
        "class_style": "unknown",
        "file_style": "unknown",
        "common_prefixes": [],
    })
    architecture_patterns: list[str] = field(default_factory=list)
    error_handling: dict = field(default_factory=lambda: {
        "style": "unknown",
        "patterns": [],
        "examples": [],
    })
    preferred_libraries: dict[str, dict] = field(default_factory=dict)
    code_style: dict = field(default_factory=lambda: {
        "line_length_pref": "unknown",
        "import_style": "unknown",
        "docstring_style": "unknown",
        "type_hints_usage": "unknown",
    })
    review_feedback: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    last_updated: str = ""
    confidence: float = 0.0
    evidence_count: int = 0

    def to_dict(self) -> dict:
        """Serialize the profile to a plain dictionary."""
        return {
            "naming_conventions": self.naming_conventions,
            "architecture_patterns": self.architecture_patterns,
            "error_handling": self.error_handling,
            "preferred_libraries": self.preferred_libraries,
            "code_style": self.code_style,
            "review_feedback": self.review_feedback,
            "anti_patterns": self.anti_patterns,
            "last_updated": self.last_updated,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
        }

    def get_section(self, section: str) -> dict | list[str]:
        """Return a specific section by name."""
        if section not in PROFILE_SECTIONS:
            return {}
        return getattr(self, section)
