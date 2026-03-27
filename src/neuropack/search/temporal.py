"""Temporal retrieval: date-aware memory search for LongMemEval."""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Callable

import sqlite3


# Day-of-week map for resolving "last Tuesday" etc.
_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_SEASON_MONTHS = {
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
    "fall": (9, 11),
    "winter": (12, 2),
}

# Named holidays (month, day)
_HOLIDAYS = {
    "christmas": (12, 25),
    "christmas eve": (12, 24),
    "new year": (1, 1),
    "new years": (1, 1),
    "new year's": (1, 1),
    "valentine's day": (2, 14),
    "valentines day": (2, 14),
    "halloween": (10, 31),
    "independence day": (7, 4),
    "july 4th": (7, 4),
    "4th of july": (7, 4),
}


def _last_day_of_month(year: int, month: int) -> int:
    """Return last day of the given month."""
    if month == 12:
        return 31
    next_month = datetime(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def _now() -> datetime:
    """Current UTC time. Separated for testability."""
    return datetime.now(timezone.utc)


class TemporalRetriever:
    """Date-aware memory retrieval — the 4th signal in hybrid RRF fusion."""

    def __init__(self, db_connection_factory: Callable[[], sqlite3.Connection]):
        self._connect = db_connection_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_temporal_query(self, query: str) -> dict:
        """Extract date references from a query string.

        Returns:
            {"dates": [datetime, ...], "ranges": [(start, end), ...],
             "raw_refs": ["May 2023", ...], "has_temporal": bool}
        """
        dates: list[datetime] = []
        ranges: list[tuple[datetime, datetime]] = []
        raw_refs: list[str] = []
        now = _now()

        # --- LongMemEval format: "2023/05/30 (Tue) 23:40" ---
        for m in re.finditer(
            r"(\d{4})/(\d{1,2})/(\d{1,2})\s*\(\w+\)\s*(\d{1,2}):(\d{2})", query
        ):
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            hour, minute = int(m.group(4)), int(m.group(5))
            try:
                dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                dates.append(dt)
                raw_refs.append(m.group(0))
            except ValueError:
                pass

        # --- ISO dates: 2023-05-30 or 2023-05-30T12:00:00 ---
        for m in re.finditer(
            r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?", query
        ):
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            hour = int(m.group(4)) if m.group(4) else 0
            minute = int(m.group(5)) if m.group(5) else 0
            second = int(m.group(6)) if m.group(6) else 0
            try:
                dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
                dates.append(dt)
                raw_refs.append(m.group(0))
            except ValueError:
                pass

        # --- Slash dates without day-of-week: 2023/05/30 ---
        for m in re.finditer(r"(\d{4})/(\d{1,2})/(\d{1,2})(?!\s*\()", query):
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                dates.append(dt)
                raw_refs.append(m.group(0))
            except ValueError:
                pass

        # --- "N days/weeks/months ago" ---
        for m in re.finditer(
            r"(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago",
            query, re.IGNORECASE,
        ):
            n = int(m.group(1))
            unit = m.group(2).lower().rstrip("s")
            if unit == "day":
                dt = now - timedelta(days=n)
                dates.append(dt)
            elif unit == "week":
                start = now - timedelta(weeks=n, days=now.weekday())
                end = start + timedelta(days=6)
                ranges.append((start.replace(hour=0, minute=0, second=0),
                               end.replace(hour=23, minute=59, second=59)))
            elif unit == "month":
                target_month = now.month - n
                target_year = now.year
                while target_month <= 0:
                    target_month += 12
                    target_year -= 1
                start = datetime(target_year, target_month, 1, tzinfo=timezone.utc)
                last_day = _last_day_of_month(target_year, target_month)
                end = datetime(target_year, target_month, last_day, 23, 59, 59, tzinfo=timezone.utc)
                ranges.append((start, end))
            elif unit == "year":
                target_year = now.year - n
                start = datetime(target_year, 1, 1, tzinfo=timezone.utc)
                end = datetime(target_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                ranges.append((start, end))
            raw_refs.append(m.group(0))

        # --- "yesterday" ---
        if re.search(r"\byesterday\b", query, re.IGNORECASE):
            dt = now - timedelta(days=1)
            dates.append(dt.replace(hour=12, minute=0, second=0))
            raw_refs.append("yesterday")

        # --- "today" ---
        if re.search(r"\btoday\b", query, re.IGNORECASE):
            dates.append(now.replace(hour=12, minute=0, second=0))
            raw_refs.append("today")

        # --- "last week" ---
        if re.search(r"\blast\s+week\b", query, re.IGNORECASE):
            start = now - timedelta(days=now.weekday() + 7)
            end = start + timedelta(days=6)
            ranges.append((start.replace(hour=0, minute=0, second=0),
                           end.replace(hour=23, minute=59, second=59)))
            raw_refs.append("last week")

        # --- "last month" ---
        if re.search(r"\blast\s+month\b", query, re.IGNORECASE):
            target_month = now.month - 1
            target_year = now.year
            if target_month <= 0:
                target_month += 12
                target_year -= 1
            start = datetime(target_year, target_month, 1, tzinfo=timezone.utc)
            last_day = _last_day_of_month(target_year, target_month)
            end = datetime(target_year, target_month, last_day, 23, 59, 59, tzinfo=timezone.utc)
            ranges.append((start, end))
            raw_refs.append("last month")

        # --- "last year" ---
        if re.search(r"\blast\s+year\b", query, re.IGNORECASE):
            target_year = now.year - 1
            start = datetime(target_year, 1, 1, tzinfo=timezone.utc)
            end = datetime(target_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            ranges.append((start, end))
            raw_refs.append("last year")

        # --- "recently" → last 30 days ---
        if re.search(r"\brecently\b", query, re.IGNORECASE):
            start = now - timedelta(days=30)
            ranges.append((start, now))
            raw_refs.append("recently")

        # --- Day of week: "on Tuesday", "last Friday" ---
        for m in re.finditer(
            r"\b(?:last|on|this past)\s+(" + "|".join(_WEEKDAY_MAP.keys()) + r")\b",
            query, re.IGNORECASE,
        ):
            day_name = m.group(1).lower()
            target_weekday = _WEEKDAY_MAP[day_name]
            days_back = (now.weekday() - target_weekday) % 7
            if days_back == 0:
                days_back = 7  # "last Tuesday" means previous week if today is Tuesday
            dt = now - timedelta(days=days_back)
            dates.append(dt.replace(hour=12, minute=0, second=0))
            raw_refs.append(m.group(0))

        # --- Named month + year: "in January 2023", "March 2024" ---
        month_names = "|".join(_MONTH_MAP.keys())
        for m in re.finditer(
            r"\b(?:in\s+)?(" + month_names + r")\s+(\d{4})\b",
            query, re.IGNORECASE,
        ):
            month_name = m.group(1).lower()
            year = int(m.group(2))
            month_num = _MONTH_MAP[month_name]
            start = datetime(year, month_num, 1, tzinfo=timezone.utc)
            last_day = _last_day_of_month(year, month_num)
            end = datetime(year, month_num, last_day, 23, 59, 59, tzinfo=timezone.utc)
            ranges.append((start, end))
            raw_refs.append(m.group(0))

        # --- Named month alone: "in March", "in September" ---
        for m in re.finditer(
            r"\bin\s+(" + month_names + r")\b(?!\s+\d{4})",
            query, re.IGNORECASE,
        ):
            month_name = m.group(1).lower()
            month_num = _MONTH_MAP[month_name]
            # Use current year if the month hasn't passed yet, otherwise previous year
            year = now.year if month_num <= now.month else now.year - 1
            start = datetime(year, month_num, 1, tzinfo=timezone.utc)
            last_day = _last_day_of_month(year, month_num)
            end = datetime(year, month_num, last_day, 23, 59, 59, tzinfo=timezone.utc)
            ranges.append((start, end))
            raw_refs.append(m.group(0))

        # --- Seasons: "in the summer", "last summer", "last spring" ---
        for m in re.finditer(
            r"\b(?:in\s+(?:the\s+)?|last\s+)(spring|summer|autumn|fall|winter)\b",
            query, re.IGNORECASE,
        ):
            season = m.group(1).lower()
            start_month, end_month = _SEASON_MONTHS[season]
            is_last = "last" in m.group(0).lower()

            if season == "winter":
                # Winter spans Dec-Feb
                year = now.year - 1 if is_last else now.year
                start = datetime(year, 12, 1, tzinfo=timezone.utc)
                end = datetime(year + 1, 2, _last_day_of_month(year + 1, 2), 23, 59, 59, tzinfo=timezone.utc)
            else:
                year = now.year - 1 if is_last else now.year
                start = datetime(year, start_month, 1, tzinfo=timezone.utc)
                end = datetime(year, end_month, _last_day_of_month(year, end_month), 23, 59, 59, tzinfo=timezone.utc)
            ranges.append((start, end))
            raw_refs.append(m.group(0))

        # --- Just a year: "in 2019", "in 2023" ---
        for m in re.finditer(r"\bin\s+(\d{4})\b", query, re.IGNORECASE):
            year = int(m.group(1))
            if 1900 <= year <= 2100:
                start = datetime(year, 1, 1, tzinfo=timezone.utc)
                end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                ranges.append((start, end))
                raw_refs.append(m.group(0))

        # --- Holidays: "before Christmas", "on Halloween" ---
        for holiday, (hmonth, hday) in _HOLIDAYS.items():
            pattern = r"\b(?:before|after|on|around)\s+" + re.escape(holiday) + r"\b"
            hm = re.search(pattern, query, re.IGNORECASE)
            if hm:
                year = now.year
                holiday_dt = datetime(year, hmonth, hday, tzinfo=timezone.utc)
                if holiday_dt > now:
                    year -= 1
                    holiday_dt = datetime(year, hmonth, hday, tzinfo=timezone.utc)

                if "before" in hm.group(0).lower():
                    # Range: start of year to holiday
                    start = datetime(year, 1, 1, tzinfo=timezone.utc)
                    ranges.append((start, holiday_dt))
                elif "after" in hm.group(0).lower():
                    end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                    ranges.append((holiday_dt, end))
                else:
                    # "on" or "around" — exact date
                    dates.append(holiday_dt)
                raw_refs.append(hm.group(0))

        has_temporal = len(dates) > 0 or len(ranges) > 0

        return {
            "dates": dates,
            "ranges": ranges,
            "raw_refs": raw_refs,
            "has_temporal": has_temporal,
        }

    def extract_dates_from_content(self, content: str) -> list[dict]:
        """Extract all date/time references from text content.

        Returns list of {"text": "May 30, 2023", "date": datetime, "type": "exact|month|year|relative"}
        """
        results: list[dict] = []

        # LongMemEval format: "2023/05/30 (Tue) 23:40"
        for m in re.finditer(
            r"(\d{4})/(\d{1,2})/(\d{1,2})\s*\(\w+\)\s*(\d{1,2}):(\d{2})", content
        ):
            try:
                dt = datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)), tzinfo=timezone.utc,
                )
                results.append({"text": m.group(0), "date": dt, "type": "exact"})
            except ValueError:
                pass

        # ISO dates
        for m in re.finditer(
            r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?", content
        ):
            try:
                dt = datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0),
                    tzinfo=timezone.utc,
                )
                results.append({"text": m.group(0), "date": dt, "type": "exact"})
            except ValueError:
                pass

        # Slash dates: 2023/05/30
        for m in re.finditer(r"(\d{4})/(\d{1,2})/(\d{1,2})(?!\s*\()", content):
            try:
                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
                results.append({"text": m.group(0), "date": dt, "type": "exact"})
            except ValueError:
                pass

        # "Month Day, Year": "May 30, 2023", "January 5, 2024"
        month_names = "|".join(_MONTH_MAP.keys())
        for m in re.finditer(
            r"\b(" + month_names + r")\s+(\d{1,2}),?\s+(\d{4})\b",
            content, re.IGNORECASE,
        ):
            month_num = _MONTH_MAP[m.group(1).lower()]
            day = int(m.group(2))
            year = int(m.group(3))
            try:
                dt = datetime(year, month_num, day, tzinfo=timezone.utc)
                results.append({"text": m.group(0), "date": dt, "type": "exact"})
            except ValueError:
                pass

        # "Month Year": "May 2023", "January 2024"
        for m in re.finditer(
            r"\b(" + month_names + r")\s+(\d{4})\b",
            content, re.IGNORECASE,
        ):
            month_num = _MONTH_MAP[m.group(1).lower()]
            year = int(m.group(2))
            try:
                dt = datetime(year, month_num, 1, tzinfo=timezone.utc)
                results.append({"text": m.group(0), "date": dt, "type": "month"})
            except ValueError:
                pass

        # Standalone years in context: "in 2019", "during 2023"
        for m in re.finditer(r"\b(?:in|during)\s+(\d{4})\b", content, re.IGNORECASE):
            year = int(m.group(1))
            if 1900 <= year <= 2100:
                dt = datetime(year, 1, 1, tzinfo=timezone.utc)
                results.append({"text": m.group(0), "date": dt, "type": "year"})

        return results

    def retrieve(
        self, query: str, limit: int = 20, namespace: str | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve memories ranked by temporal relevance.

        Returns list of (memory_id, score) tuples.
        """
        parsed = self.parse_temporal_query(query)
        if not parsed["has_temporal"]:
            return []

        conn = self._connect()
        memory_scores: dict[str, float] = {}

        target_dates = parsed["dates"]
        target_ranges = parsed["ranges"]

        # Strategy 1: Match memories whose created_at falls within target ranges
        for start, end in target_ranges:
            self._score_by_creation_range(
                conn, start, end, namespace, memory_scores,
            )

        # Strategy 2: Match memories whose created_at is near target dates
        for dt in target_dates:
            self._score_by_creation_proximity(
                conn, dt, namespace, memory_scores,
            )

        # Strategy 3: Content-based date matching
        all_rows = self._fetch_candidate_memories(conn, namespace, limit * 5)
        for row in all_rows:
            mid = row["id"]
            content = row["content"]
            tags_str = row["tags"]

            content_dates = self.extract_dates_from_content(content)
            # Also check tags for date strings
            content_dates.extend(self.extract_dates_from_content(tags_str))

            if not content_dates:
                continue

            best_score = 0.0
            for cd in content_dates:
                cd_date = cd["date"]
                cd_type = cd["type"]

                # Score against target dates
                for td in target_dates:
                    s = self._temporal_similarity(td, cd_date, cd_type)
                    best_score = max(best_score, s)

                # Score against target ranges
                for start, end in target_ranges:
                    mid_range = start + (end - start) / 2
                    s = self._temporal_similarity(mid_range, cd_date, cd_type)
                    # Boost if date falls within range
                    if start <= cd_date <= end:
                        s = max(s, 0.9)
                    best_score = max(best_score, s)

            if best_score > 0:
                memory_scores[mid] = max(memory_scores.get(mid, 0.0), best_score)

        # Sort by score descending, return top-limit
        ranked = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _temporal_similarity(
        self, target: datetime, candidate: datetime, match_type: str,
        sigma: float = 30.0,
    ) -> float:
        """Score temporal proximity between two dates.

        match_type affects scoring:
          - "exact": date is precise, use day-level comparison
          - "month": date is month-level, be more lenient
          - "year": date is year-level, only match by year
        """
        # Ensure both are offset-aware for comparison
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        if candidate.tzinfo is None:
            candidate = candidate.replace(tzinfo=timezone.utc)

        days_diff = abs((target - candidate).total_seconds()) / 86400.0

        if match_type == "year":
            return 0.3 if target.year == candidate.year else 0.0

        if match_type == "month":
            if target.year == candidate.year and target.month == candidate.month:
                return 0.7
            if target.year == candidate.year:
                return 0.3
            return 0.0

        # Exact type — fine-grained scoring
        if days_diff < 0.5:
            return 1.0
        if days_diff <= 3.5:
            return 0.8  # same week-ish
        if target.year == candidate.year and target.month == candidate.month:
            return 0.6
        if target.year == candidate.year:
            return 0.3

        # Exponential decay
        return math.exp(-days_diff / sigma)

    def _score_by_creation_range(
        self,
        conn: sqlite3.Connection,
        start: datetime,
        end: datetime,
        namespace: str | None,
        scores: dict[str, float],
    ) -> None:
        """Score memories whose created_at falls in [start, end]."""
        params: list[object] = [start.isoformat(), end.isoformat()]
        ns_clause = ""
        if namespace:
            ns_clause = " AND namespace = ?"
            params.append(namespace)

        rows = conn.execute(
            f"SELECT id, created_at FROM memories WHERE created_at >= ? AND created_at <= ?{ns_clause}",
            params,
        ).fetchall()

        mid_range = start + (end - start) / 2
        for row in rows:
            d = dict(row)
            mid = d["id"]
            created = datetime.fromisoformat(d["created_at"])
            # In-range memories get a high base score
            score = 0.9
            # Boost for being near the center of the range
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            days_from_center = abs((created - mid_range).total_seconds()) / 86400.0
            range_days = max((end - start).total_seconds() / 86400.0, 1.0)
            center_boost = 0.1 * (1.0 - min(days_from_center / range_days, 1.0))
            final = score + center_boost
            scores[mid] = max(scores.get(mid, 0.0), final)

    def _score_by_creation_proximity(
        self,
        conn: sqlite3.Connection,
        target: datetime,
        namespace: str | None,
        scores: dict[str, float],
        sigma: float = 30.0,
    ) -> None:
        """Score memories by how close their created_at is to target date."""
        # Fetch memories within a reasonable window (sigma * 3 days)
        window = timedelta(days=sigma * 3)
        start = (target - window).isoformat()
        end = (target + window).isoformat()

        params: list[object] = [start, end]
        ns_clause = ""
        if namespace:
            ns_clause = " AND namespace = ?"
            params.append(namespace)

        rows = conn.execute(
            f"SELECT id, created_at FROM memories WHERE created_at >= ? AND created_at <= ?{ns_clause}",
            params,
        ).fetchall()

        for row in rows:
            d = dict(row)
            mid = d["id"]
            created = datetime.fromisoformat(d["created_at"])
            score = self._temporal_similarity(target, created, "exact", sigma)
            if score > 0:
                scores[mid] = max(scores.get(mid, 0.0), score)

    def _fetch_candidate_memories(
        self, conn: sqlite3.Connection, namespace: str | None, limit: int,
    ) -> list[dict]:
        """Fetch recent memories for content-based date matching."""
        params: list[object] = []
        ns_clause = ""
        if namespace:
            ns_clause = " WHERE namespace = ?"
            params.append(namespace)

        params.append(limit)
        rows = conn.execute(
            f"SELECT id, content, tags, created_at FROM memories{ns_clause} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
