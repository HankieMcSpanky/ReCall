"""Memory DNA — detects behavioral patterns from memory history.

Patterns detected:
- Frequency: "User discusses auth every Monday"
- Oscillation: "User switches frameworks every 2 weeks"
- Correlation: "User's activity peaks on weekends"
- Trend: "User is gradually moving from backend to frontend"

Usage:
    from neuropack.agents.pattern_detector import PatternDetector
    detector = PatternDetector(store)
    patterns = detector.analyze()
    warnings = detector.check_pattern("I'm switching to Vue")
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from neuropack.types import MemoryRecord

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Keyword groups used for trend/oscillation detection
_TECH_GROUPS: dict[str, list[str]] = {
    "frontend": ["react", "vue", "angular", "svelte", "nextjs", "css", "html", "tailwind"],
    "backend": ["django", "flask", "fastapi", "express", "rails", "spring", "node"],
    "language": ["python", "javascript", "typescript", "rust", "go", "java", "ruby", "c++"],
    "database": ["postgres", "mysql", "sqlite", "mongo", "redis", "dynamo"],
    "infra": ["docker", "kubernetes", "terraform", "aws", "gcp", "azure", "ci/cd"],
}


@dataclass
class Pattern:
    """A detected behavioral pattern."""

    pattern_type: str  # frequency, oscillation, trend, correlation
    description: str
    confidence: float  # 0.0-1.0
    evidence: list[dict[str, Any]] = field(default_factory=list)


class PatternDetector:
    """Detects behavioral patterns from stored memories."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._records: list[MemoryRecord] | None = None
        self._patterns: list[Pattern] = []

    def _load_records(self) -> list[MemoryRecord]:
        if self._records is None:
            self._records = self._store.repo.list_all(limit=10000)
        return self._records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> list[Pattern]:
        """Scan all memories and detect patterns."""
        self._patterns = []
        self._patterns.extend(self.detect_frequency_patterns())
        self._patterns.extend(self.detect_oscillation_patterns())
        self._patterns.extend(self.detect_trends())
        return self._patterns

    def check_pattern(self, action: str) -> list[str]:
        """Check if *action* matches a known pattern; return human-readable warnings."""
        if not self._patterns:
            self.analyze()

        action_lower = action.lower()
        warnings: list[str] = []

        for pat in self._patterns:
            if pat.pattern_type == "oscillation":
                # Warn if the action mentions a keyword from an oscillation pattern
                for ev in pat.evidence:
                    for kw in ev.get("keywords", []):
                        if kw in action_lower:
                            warnings.append(
                                f"Pattern detected: {pat.description} "
                                f"(confidence {pat.confidence:.0%})"
                            )
                            break

            elif pat.pattern_type == "trend":
                for ev in pat.evidence:
                    from_kw = ev.get("from", "")
                    if from_kw and from_kw in action_lower:
                        warnings.append(
                            f"Note: You have been trending away from '{from_kw}'. "
                            f"{pat.description} (confidence {pat.confidence:.0%})"
                        )

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for w in warnings:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique

    # ------------------------------------------------------------------
    # Frequency patterns
    # ------------------------------------------------------------------

    def detect_frequency_patterns(self) -> list[Pattern]:
        """Find topics that recur on specific days or intervals."""
        records = self._load_records()
        if len(records) < 3:
            return []

        tag_days: dict[str, list[int]] = defaultdict(list)  # tag -> list of weekdays
        tag_dates: dict[str, list[datetime]] = defaultdict(list)

        for r in records:
            weekday = r.created_at.weekday()
            for tag in r.tags:
                tag_days[tag].append(weekday)
                tag_dates[tag].append(r.created_at)

        patterns: list[Pattern] = []
        for tag, days in tag_days.items():
            if len(days) < 3:
                continue
            counter = Counter(days)
            most_common_day, count = counter.most_common(1)[0]
            ratio = count / len(days)
            if ratio >= 0.5 and count >= 3:
                confidence = min(ratio, 1.0)
                patterns.append(Pattern(
                    pattern_type="frequency",
                    description=f"Topic '{tag}' frequently appears on {DAY_NAMES[most_common_day]}s "
                                f"({count}/{len(days)} occurrences)",
                    confidence=confidence,
                    evidence=[{"tag": tag, "day": DAY_NAMES[most_common_day],
                               "count": count, "total": len(days)}],
                ))

            # Interval detection: check if memories are roughly evenly spaced
            dates_sorted = sorted(tag_dates[tag])
            if len(dates_sorted) >= 3:
                deltas = [(dates_sorted[i + 1] - dates_sorted[i]).days
                          for i in range(len(dates_sorted) - 1)]
                deltas = [d for d in deltas if d > 0]
                if deltas:
                    avg_delta = sum(deltas) / len(deltas)
                    variance = sum((d - avg_delta) ** 2 for d in deltas) / len(deltas)
                    cv = (variance ** 0.5) / avg_delta if avg_delta > 0 else 999
                    if cv < 0.4 and avg_delta >= 2:
                        patterns.append(Pattern(
                            pattern_type="frequency",
                            description=f"Topic '{tag}' recurs roughly every {avg_delta:.0f} days",
                            confidence=max(0.0, 1.0 - cv),
                            evidence=[{"tag": tag, "avg_interval_days": round(avg_delta, 1),
                                       "occurrences": len(dates_sorted)}],
                        ))
        return patterns

    # ------------------------------------------------------------------
    # Oscillation patterns
    # ------------------------------------------------------------------

    def detect_oscillation_patterns(self) -> list[Pattern]:
        """Find attributes that keep switching back and forth."""
        records = self._load_records()
        if len(records) < 4:
            return []

        sorted_records = sorted(records, key=lambda r: r.created_at)
        patterns: list[Pattern] = []

        for group_name, keywords in _TECH_GROUPS.items():
            timeline: list[tuple[datetime, str]] = []
            for r in sorted_records:
                text = (r.content + " " + " ".join(r.tags)).lower()
                for kw in keywords:
                    if re.search(rf"\b{re.escape(kw)}\b", text):
                        timeline.append((r.created_at, kw))

            if len(timeline) < 4:
                continue

            # Detect switches: consecutive mentions of *different* keywords
            switches: list[tuple[datetime, str, str]] = []
            prev_kw = timeline[0][1]
            for dt, kw in timeline[1:]:
                if kw != prev_kw:
                    switches.append((dt, prev_kw, kw))
                    prev_kw = kw
                else:
                    prev_kw = kw

            if len(switches) >= 2:
                avg_gap = 0.0
                if len(switches) >= 2:
                    gaps = [(switches[i + 1][0] - switches[i][0]).days
                            for i in range(len(switches) - 1)]
                    avg_gap = sum(gaps) / len(gaps) if gaps else 0

                confidence = min(len(switches) / 6, 1.0)
                kw_set = list({s[1] for s in switches} | {s[2] for s in switches})
                last_switch = switches[-1]
                patterns.append(Pattern(
                    pattern_type="oscillation",
                    description=(
                        f"Oscillation in {group_name}: switched between "
                        f"{', '.join(kw_set)} {len(switches)} times"
                        + (f" (~every {avg_gap:.0f} days)" if avg_gap > 0 else "")
                    ),
                    confidence=confidence,
                    evidence=[{"group": group_name, "switches": len(switches),
                               "keywords": kw_set,
                               "last_switch": f"{last_switch[1]} -> {last_switch[2]}"}],
                ))
        return patterns

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def detect_trends(self) -> list[Pattern]:
        """Find directional trends (e.g., moving from backend to frontend)."""
        records = self._load_records()
        if len(records) < 4:
            return []

        sorted_records = sorted(records, key=lambda r: r.created_at)
        midpoint = len(sorted_records) // 2
        first_half = sorted_records[:midpoint]
        second_half = sorted_records[midpoint:]

        patterns: list[Pattern] = []

        for group_name, keywords in _TECH_GROUPS.items():
            first_counts: Counter[str] = Counter()
            second_counts: Counter[str] = Counter()

            for r in first_half:
                text = (r.content + " " + " ".join(r.tags)).lower()
                for kw in keywords:
                    if re.search(rf"\b{re.escape(kw)}\b", text):
                        first_counts[kw] += 1

            for r in second_half:
                text = (r.content + " " + " ".join(r.tags)).lower()
                for kw in keywords:
                    if re.search(rf"\b{re.escape(kw)}\b", text):
                        second_counts[kw] += 1

            # Look for a keyword rising and another falling
            all_kws = set(first_counts) | set(second_counts)
            for kw in all_kws:
                fc = first_counts.get(kw, 0)
                sc = second_counts.get(kw, 0)
                total = fc + sc
                if total < 3:
                    continue
                shift = (sc - fc) / total  # -1 to +1

                if shift >= 0.4:
                    patterns.append(Pattern(
                        pattern_type="trend",
                        description=f"Increasing focus on '{kw}' ({group_name}): "
                                    f"{fc} mentions early -> {sc} mentions recently",
                        confidence=min(abs(shift), 1.0),
                        evidence=[{"keyword": kw, "group": group_name,
                                   "early": fc, "recent": sc, "from": ""}],
                    ))
                elif shift <= -0.4:
                    patterns.append(Pattern(
                        pattern_type="trend",
                        description=f"Decreasing focus on '{kw}' ({group_name}): "
                                    f"{fc} mentions early -> {sc} mentions recently",
                        confidence=min(abs(shift), 1.0),
                        evidence=[{"keyword": kw, "group": group_name,
                                   "early": fc, "recent": sc, "from": kw}],
                    ))
        return patterns
