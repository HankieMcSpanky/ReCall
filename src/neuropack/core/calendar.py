"""Virtual Calendar — maps all events to a searchable timeline.

Every memory gets placed on a calendar with:
- session_date: when the conversation happened
- event_dates: dates mentioned IN the conversation
- relative_dates: "last Tuesday", "3 weeks ago" resolved to actual dates

The calendar can answer temporal questions directly:
- "How many days between X and Y?" → look up both events, subtract
- "What happened first?" → sort by date, return first
- "What changed?" → find same topic at different dates, show progression
"""
from __future__ import annotations

import calendar as _cal
import re
from datetime import datetime, timedelta
from typing import Optional

# Month-name lookup for parsing temporal cues in questions
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


class VirtualCalendar:
    """A timeline of events extracted from conversations."""

    def __init__(self):
        self.events: list[CalendarEvent] = []
        self._by_date: dict[str, list[CalendarEvent]] = {}  # date_str -> events
        self._by_topic: dict[str, list[CalendarEvent]] = {}  # keyword -> events

    def add_event(
        self,
        description: str,
        session_date: datetime,
        event_date: Optional[datetime] = None,
        session_id: str = "",
        session_idx: int = 0,
        speaker: str = "user",
        raw_expression: str = "",
    ) -> None:
        """Add an event to the calendar."""
        actual_date = event_date or session_date
        event = CalendarEvent(
            description=description,
            session_date=session_date,
            event_date=actual_date,
            session_id=session_id,
            session_idx=session_idx,
            speaker=speaker,
            raw_expression=raw_expression,
        )
        self.events.append(event)

        # Index by date
        date_key = actual_date.strftime("%Y-%m-%d")
        self._by_date.setdefault(date_key, []).append(event)

        # Index by keywords
        for word in _extract_keywords(description):
            self._by_topic.setdefault(word, []).append(event)

    def build_from_sessions(
        self,
        sessions: list,
        dates: list[str],
        session_ids: list = None,
    ) -> None:
        """Build calendar from LongMemEval sessions."""
        for i, sess in enumerate(sessions):
            date_str = dates[i] if i < len(dates) else ""
            session_date = _parse_date(date_str)
            if not session_date:
                continue

            sid = str(session_ids[i]) if session_ids and i < len(session_ids) else str(i)

            for turn in sess:
                if isinstance(turn, dict):
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                elif isinstance(turn, list) and len(turn) >= 2:
                    role, content = turn[0], turn[1]
                else:
                    continue

                if not content or not content.strip():
                    continue

                content = content.strip()

                # Extract sentences with factual content
                for sentence in _split_sentences(content):
                    if len(sentence) < 15:
                        continue

                    # Check if sentence contains a date reference
                    event_date, raw_expression = _extract_event_date(sentence, session_date)

                    # Only add meaningful sentences (not greetings/filler)
                    if _is_meaningful(sentence):
                        self.add_event(
                            description=sentence,
                            session_date=session_date,
                            event_date=event_date,
                            session_id=sid,
                            session_idx=i,
                            speaker=role,
                            raw_expression=raw_expression,
                        )

    # ------------------------------------------------------------------
    # Time-window pre-filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_time_window(
        question: str, question_date: datetime,
    ) -> Optional[tuple[datetime, datetime]]:
        """Extract a date range from temporal cues in *question*.

        Returns ``(start, end)`` or ``None`` when no temporal cue is found.

        Supported patterns:
        - "in May 2023"          -> (2023-05-01, 2023-05-31)
        - "in May"               -> same month, year inferred from question_date
        - "last week"            -> (question_date - 14d, question_date)
        - "last month"           -> first-to-last day of previous month
        - "in the past month"    -> (question_date - 45d, question_date)
        - "3 weeks ago"          -> (question_date - 28d, question_date - 14d)
        - "N days/months ago"    -> similar sliding windows
        - "in 2023"              -> (2023-01-01, 2023-12-31)
        """
        q = question.lower()

        # --- "in <Month> <Year>" -----------------------------------------------
        month_names_re = "|".join(_MONTH_NAMES.keys())
        m = re.search(
            r'\bin\s+(' + month_names_re + r')\s+(\d{4})\b', q,
        )
        if m:
            mn = _MONTH_NAMES[m.group(1)]
            yr = int(m.group(2))
            last_day = _cal.monthrange(yr, mn)[1]
            return (datetime(yr, mn, 1), datetime(yr, mn, last_day, 23, 59, 59))

        # --- "<Month> <Year>" without leading "in" ----------------------------
        m = re.search(
            r'\b(' + month_names_re + r')\s+(\d{4})\b', q,
        )
        if m:
            mn = _MONTH_NAMES[m.group(1)]
            yr = int(m.group(2))
            last_day = _cal.monthrange(yr, mn)[1]
            return (datetime(yr, mn, 1), datetime(yr, mn, last_day, 23, 59, 59))

        # --- "in <Month>" (no year) -------------------------------------------
        m = re.search(
            r'\bin\s+(' + month_names_re + r')\b(?!\s+\d{4})', q,
        )
        if m:
            mn = _MONTH_NAMES[m.group(1)]
            yr = question_date.year
            last_day = _cal.monthrange(yr, mn)[1]
            return (datetime(yr, mn, 1), datetime(yr, mn, last_day, 23, 59, 59))

        # --- "N weeks/days/months ago" -----------------------------------------
        m = re.search(r'(\d+)\s+(day|days|week|weeks|month|months)\s+ago', q)
        if m:
            n = int(m.group(1))
            unit = m.group(2).rstrip("s")
            if unit == "week":
                end = question_date - timedelta(days=max(n * 7 - 7, 0))
                start = question_date - timedelta(days=n * 7 + 7)
                return (start, end)
            elif unit == "day":
                end = question_date - timedelta(days=max(n - 1, 0))
                start = question_date - timedelta(days=n + 7)
                return (start, end)
            elif unit == "month":
                end = question_date - timedelta(days=max((n - 1) * 30, 0))
                start = question_date - timedelta(days=(n + 1) * 30)
                return (start, end)

        # --- "last week" ------------------------------------------------------
        if re.search(r'\blast\s+week\b', q):
            return (question_date - timedelta(days=14), question_date)

        # --- "last month" -----------------------------------------------------
        if re.search(r'\blast\s+month\b', q):
            first_of_cur = question_date.replace(day=1)
            end = first_of_cur - timedelta(days=1)  # last day of prev month
            start = end.replace(day=1)
            return (start, datetime(end.year, end.month, end.day, 23, 59, 59))

        # --- "in the past month" / "past few weeks" / "recently" --------------
        if re.search(r'\b(?:past|recent)\s+(?:month|few weeks|weeks)\b', q) or \
           re.search(r'\brecently\b', q):
            return (question_date - timedelta(days=45), question_date)

        # --- "in <year>" -----------------------------------------------------
        m = re.search(r'\bin\s+(\d{4})\b', q)
        if m:
            yr = int(m.group(1))
            if 1900 <= yr <= 2100:
                return (datetime(yr, 1, 1), datetime(yr, 12, 31, 23, 59, 59))

        return None

    def filter_events_by_time_window(
        self,
        question: str,
        question_date: Optional[datetime] = None,
        margin_days: int = 14,
    ) -> list[CalendarEvent]:
        """Parse temporal cues from *question* and filter events to that window.

        Returns only events within the parsed time window +/- *margin_days*.
        If no temporal cue is found, returns **all** events (no filtering).
        """
        if question_date is None:
            if self.events:
                question_date = max(e.session_date for e in self.events)
            else:
                return list(self.events)

        window = self._parse_time_window(question, question_date)
        if window is None:
            return list(self.events)

        start, end = window
        margin = timedelta(days=margin_days)
        lo = start - margin
        hi = end + margin

        return [
            e for e in self.events
            if lo <= e.session_date <= hi or lo <= e.event_date <= hi
        ]

    def query(self, question: str, question_date: Optional[datetime] = None) -> str:
        """Answer a temporal question using the calendar.

        Returns a pre-computed answer string, or "" if can't answer.
        """
        q_lower = question.lower()
        keywords = _extract_keywords(question)
        # Use provided question_date or fall back to latest session date
        if question_date is None and self.events:
            question_date = max(e.session_date for e in self.events)

        # --- Time-window pre-filtering ----------------------------------------
        # Narrow events to the relevant time window BEFORE keyword matching.
        # This prevents matching events from the wrong time period (e.g. a
        # "MoMA visit" from January when the question asks about March).
        filtered_events = self.filter_events_by_time_window(
            question, question_date,
        )
        # Build a temporary topic index over only the filtered events
        saved_by_topic = self._by_topic
        self._by_topic = {}
        for ev in filtered_events:
            for word in _extract_keywords(ev.description):
                self._by_topic.setdefault(word, []).append(ev)

        # Find relevant events (now searches only within the time window)
        relevant = self._find_relevant_events(keywords)

        # Restore the full topic index immediately
        self._by_topic = saved_by_topic

        if not relevant:
            return ""

        # Extract proper nouns from query for confidence checking
        proper_nouns = set()
        for word in question.split():
            clean = word.strip("?,!.;:\"'()[]")
            if clean and clean[0].isupper() and len(clean) > 1 and clean.lower() not in {
                'how', 'what', 'when', 'where', 'which', 'who', 'did', 'does',
                'the', 'and', 'was', 'are', 'can', 'has', 'have', 'will',
                'first', 'last', 'many', 'much',
            }:
                proper_nouns.add(clean.lower())

        # If query has proper nouns, filter events to those mentioning them
        if proper_nouns:
            high_confidence = [
                e for e in relevant
                if any(pn in e.description.lower() for pn in proper_nouns)
            ]
            if high_confidence:
                relevant = high_confidence

        # Sort chronologically
        relevant.sort(key=lambda e: e.event_date)

        result_parts: list[str] = []

        # --- "How many days/weeks between X and Y?" ---
        duration_match = re.search(
            r'how many (days|weeks|months)\s+(?:between|from|since|passed)',
            q_lower,
        )
        if duration_match and len(relevant) >= 2:
            unit = duration_match.group(1)
            first = relevant[0]
            last = relevant[-1]
            days = (last.event_date - first.event_date).days
            if unit == "weeks":
                val = round(days / 7, 1)
                result_parts.append(
                    f"CALENDAR ANSWER: {abs(days)} days ({val} weeks) "
                    f"between '{first.description[:60]}' ({first.event_date.strftime('%Y/%m/%d')}) "
                    f"and '{last.description[:60]}' ({last.event_date.strftime('%Y/%m/%d')})"
                )
            elif unit == "months":
                val = round(days / 30.44, 1)
                result_parts.append(
                    f"CALENDAR ANSWER: {abs(days)} days (~{val} months) "
                    f"between '{first.description[:60]}' ({first.event_date.strftime('%Y/%m/%d')}) "
                    f"and '{last.description[:60]}' ({last.event_date.strftime('%Y/%m/%d')})"
                )
            else:
                result_parts.append(
                    f"CALENDAR ANSWER: {abs(days)} days "
                    f"between '{first.description[:60]}' ({first.event_date.strftime('%Y/%m/%d')}) "
                    f"and '{last.description[:60]}' ({last.event_date.strftime('%Y/%m/%d')})"
                )

        # --- "How many days/weeks ago?" ---
        ago_match = re.search(r'how (?:many|long)\s+(?:days|weeks|months)\s+ago', q_lower)
        if ago_match and relevant and question_date:
            last_event = relevant[-1]
            days = (question_date - last_event.event_date).days
            weeks = round(days / 7, 1)
            result_parts.append(
                f"CALENDAR ANSWER: {days} days ago (~{weeks} weeks) — "
                f"'{last_event.description[:60]}' was on "
                f"{last_event.event_date.strftime('%Y/%m/%d')}, "
                f"question date is {question_date.strftime('%Y/%m/%d')}"
            )

        # --- "Which happened first?" / "before or after" / "who X first" ---
        order_match = re.search(
            r'(which.*first)|(before|after)\s+(my|the|i)|(happened.*first)|(more recent)'
            r'|(graduated.*first)|(came.*first)|(did.*first)|(became.*first)'
            r'|(who.*first)|(meet.*first)|(started.*first)',
            q_lower,
        )
        if order_match and len(relevant) >= 2:
            result_parts.append(
                f"CALENDAR ANSWER: Chronological order:"
            )
            for i, ev in enumerate(relevant[:5], 1):
                raw_tag = f" [expr: \"{ev.raw_expression}\"]" if ev.raw_expression else ""
                result_parts.append(
                    f"  {i}. {ev.event_date.strftime('%Y/%m/%d')} — {ev.description[:80]} "
                    f"(Session {ev.session_idx + 1}){raw_tag}"
                )
            result_parts.append(f"  FIRST: {relevant[0].description[:80]}")
            result_parts.append(f"  LAST: {relevant[-1].description[:80]}")

        # --- "When did I first/last do X?" ---
        when_match = re.search(r'when did\s+(?:i|the user)\s+(first|last)', q_lower)
        if when_match and relevant:
            which = when_match.group(1)
            event = relevant[0] if which == "first" else relevant[-1]
            result_parts.append(
                f"CALENDAR ANSWER: {which.title()} occurrence on "
                f"{event.event_date.strftime('%Y/%m/%d')} — "
                f"'{event.description[:80]}' (Session {event.session_idx + 1})"
            )

        # --- "most recently" / "last time" ---
        recent_match = re.search(
            r'(most recent)|(last time)|(latest)|(did i (?:use|take|visit|go|try|eat|buy|watch|read)\s+.*most)',
            q_lower,
        )
        if recent_match and relevant and not result_parts:
            latest = relevant[-1]
            result_parts.append(
                f"CALENDAR ANSWER: Most recent occurrence on "
                f"{latest.event_date.strftime('%Y/%m/%d')} — "
                f"'{latest.description[:80]}' (Session {latest.session_idx + 1})"
            )

        # --- "What is the order of X, Y, Z" / "from earliest to latest" ---
        list_order_match = re.search(
            r'(order of (?:the )?(?:three|3|four|4|two|2))|(earliest to latest)|(first.*second.*third)',
            q_lower,
        )
        if list_order_match and len(relevant) >= 2 and not result_parts:
            result_parts.append("CALENDAR ANSWER: Chronological order:")
            for i, ev in enumerate(relevant[:6], 1):
                raw_tag = f" [expr: \"{ev.raw_expression}\"]" if ev.raw_expression else ""
                result_parts.append(
                    f"  {i}. {ev.event_date.strftime('%Y/%m/%d')} — {ev.description[:80]} "
                    f"(Session {ev.session_idx + 1}){raw_tag}"
                )

        # --- "How long had I been X when Y?" / "How long have I been X before Y?" ---
        how_long_when_match = re.search(
            r'how long (?:had|have) (?:i|the user) been\s+(.+?)\s+(?:when|before|until|by the time)'
            r'|how long (?:did|have) (?:i|the user)\s+(?:use|take|wait|work|practice|play|study|train)'
            r'\s+(.+?)\s+(?:before|until|when)',
            q_lower,
        )
        if how_long_when_match and len(relevant) >= 2 and not result_parts:
            first = relevant[0]
            last = relevant[-1]
            days = (last.event_date - first.event_date).days
            weeks = round(days / 7, 1)
            months = round(days / 30.44, 1)
            result_parts.append(
                f"CALENDAR ANSWER: {days} days (~{weeks} weeks, ~{months} months) "
                f"between first mention ({first.event_date.strftime('%Y/%m/%d')}) "
                f"and the referenced event ({last.event_date.strftime('%Y/%m/%d')})"
            )

        # --- "How many days did I spend on X?" ---
        spend_match = re.search(
            r'how many (days|weeks)\s+(?:did|have)\s+(?:i|the user)\s+(?:spend|spent)',
            q_lower,
        )
        if spend_match and len(relevant) >= 2 and not result_parts:
            first = relevant[0]
            last = relevant[-1]
            days = (last.event_date - first.event_date).days
            result_parts.append(
                f"CALENDAR ANSWER: {days} days (from "
                f"{first.event_date.strftime('%Y/%m/%d')} to "
                f"{last.event_date.strftime('%Y/%m/%d')})"
            )

        # --- "How long did it take to finish X" / "How long did I take" ---
        finish_match = re.search(
            r'how long (?:did (?:i|it)|have i)\s+(?:take|taken)\s+to\s+(?:finish|complete|read)',
            q_lower,
        )
        if finish_match and len(relevant) >= 2 and not result_parts:
            first = relevant[0]
            last = relevant[-1]
            days = abs((last.event_date - first.event_date).days)
            weeks = round(days / 7, 1)
            result_parts.append(
                f"CALENDAR ANSWER: {days} days (~{weeks} weeks) "
                f"from first mention ({first.event_date.strftime('%Y/%m/%d')}: "
                f"{first.description[:60]}) to last mention ("
                f"{last.event_date.strftime('%Y/%m/%d')}: {last.description[:60]})"
            )

        # --- "What X last Saturday?" / "Who did I Y last Tuesday?" ---
        # Resolve relative day references in the question itself
        rel_day_match = re.search(
            r'last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            q_lower,
        )
        if rel_day_match and relevant and not result_parts:
            # Show events around the resolved date
            result_parts.append("CALENDAR ANSWER: Events near that date:")
            for ev in relevant[:8]:
                raw_tag = f" [expr: \"{ev.raw_expression}\"]" if ev.raw_expression else ""
                result_parts.append(
                    f"  {ev.event_date.strftime('%Y/%m/%d')} — {ev.description[:100]} "
                    f"[{ev.speaker}] (Session {ev.session_idx + 1}){raw_tag}"
                )

        # --- "How many X before Y?" (count events before a reference event) ---
        count_before_match = re.search(
            r'how many\s+(.+?)\s+(?:before|prior to|until)',
            q_lower,
        )
        if count_before_match and len(relevant) >= 2 and not result_parts:
            result_parts.append(f"CALENDAR ANSWER: {len(relevant)} relevant events found:")
            for i, ev in enumerate(relevant[:10], 1):
                raw_tag = f" [expr: \"{ev.raw_expression}\"]" if ev.raw_expression else ""
                result_parts.append(
                    f"  {i}. {ev.event_date.strftime('%Y/%m/%d')} — {ev.description[:80]}{raw_tag}"
                )

        # --- "Who X first/second/third" ---
        who_order_match = re.search(
            r'who\s+(?:\w+\s+)?(first|second|third|earliest|latest)',
            q_lower,
        )
        if who_order_match and len(relevant) >= 2 and not result_parts:
            result_parts.append("CALENDAR ANSWER: Chronological order:")
            for i, ev in enumerate(relevant[:5], 1):
                raw_tag = f" [expr: \"{ev.raw_expression}\"]" if ev.raw_expression else ""
                result_parts.append(
                    f"  {i}. {ev.event_date.strftime('%Y/%m/%d')} — {ev.description[:80]}{raw_tag}"
                )

        # --- Timeline summary (always include for temporal context) ---
        if relevant and not result_parts:
            result_parts.append("CALENDAR TIMELINE (chronological):")
            if question_date:
                result_parts.append(f"  Question date: {question_date.strftime('%Y/%m/%d')}")
            for i, ev in enumerate(relevant[:15], 1):
                days_ago = ""
                if question_date:
                    d = (question_date - ev.event_date).days
                    if d > 0:
                        days_ago = f" [{d} days before question date]"
                raw_tag = f" [expr: \"{ev.raw_expression}\"]" if ev.raw_expression else ""
                result_parts.append(
                    f"  {ev.event_date.strftime('%Y/%m/%d')} — {ev.description[:100]} "
                    f"[{ev.speaker}] (Session {ev.session_idx + 1}){days_ago}{raw_tag}"
                )

        # --- Knowledge change detection ---
        changes = self._detect_changes(keywords)
        if changes:
            result_parts.append("\nKNOWLEDGE UPDATE DETECTED (use MOST RECENT value):")
            for old_ev, new_ev in changes:
                result_parts.append(
                    f"  SUPERSEDED ({old_ev.event_date.strftime('%Y/%m/%d')}): {old_ev.description[:100]}"
                )
                result_parts.append(
                    f"  CURRENT ({new_ev.event_date.strftime('%Y/%m/%d')}): {new_ev.description[:100]}"
                )
                result_parts.append(f"  ALWAYS USE THE CURRENT/MOST RECENT VALUE ABOVE.")

        return "\n".join(result_parts)

    def _find_relevant_events(self, keywords: list[str]) -> list[CalendarEvent]:
        """Find events matching multiple keywords (not just any one)."""
        # Count how many keywords each event matches
        event_hits: dict[int, tuple[int, CalendarEvent]] = {}
        for kw in keywords:
            for event in self._by_topic.get(kw, []):
                eid = id(event)
                if eid in event_hits:
                    event_hits[eid] = (event_hits[eid][0] + 1, event)
                else:
                    event_hits[eid] = (1, event)

        # Require at least 2 keyword matches (or 1 if only 1-2 keywords)
        min_matches = 1 if len(keywords) <= 2 else 2
        result = [
            ev for hits, ev in event_hits.values()
            if hits >= min_matches
        ]
        # Sort by number of hits (most relevant first)
        result.sort(key=lambda e: -event_hits.get(id(e), (0,))[0])
        return result[:50]  # cap at 50 to avoid overwhelming output

    def _detect_changes(self, keywords: list[str]) -> list[tuple[CalendarEvent, CalendarEvent]]:
        """Detect when the same topic has different values at different dates.

        Uses SVO extraction to find actual value changes, not just text differences.
        """
        relevant = self._find_relevant_events(keywords)
        if len(relevant) < 2:
            return []

        # Extract values from user statements using SVO patterns
        _SVO_PATTERNS = [
            re.compile(r'\bi\s+(?:work|live|study|teach|moved|switched|changed|started|joined)\s+(?:at|for|in|to|with|as)\s+(.+?)(?:\.|,|$)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:\w+\s+)?(?:is|are|was|were)\s+(.+?)(?:\.|,|$)', re.IGNORECASE),
            re.compile(r'\bi\s+(?:use|own|have|drive|bought|got|switched to|upgraded to|now have)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|,|$)', re.IGNORECASE),
            re.compile(r"\bi(?:'m| am)\s+(?:a |an )?(.+?)(?:\.|,|\band\b|$)", re.IGNORECASE),
            re.compile(r'\b(?:just|recently|now)\s+(?:got|bought|moved to|started|switched to)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|,|$)', re.IGNORECASE),
        ]

        # Find value-bearing events sorted by date
        value_events: list[tuple[datetime, str, CalendarEvent]] = []
        for ev in relevant:
            if ev.speaker != "user":
                continue
            for pat in _SVO_PATTERNS:
                m = pat.search(ev.description)
                if m:
                    value_events.append((ev.event_date, m.group(1).strip()[:60], ev))
                    break

        # Sort by date and find changes
        value_events.sort(key=lambda x: x[0])
        changes: list[tuple[CalendarEvent, CalendarEvent]] = []
        if len(value_events) >= 2:
            # Compare first and last value — if different, it's a change
            first_date, first_val, first_ev = value_events[0]
            last_date, last_val, last_ev = value_events[-1]
            if first_val.lower() != last_val.lower() and first_ev.session_idx != last_ev.session_idx:
                changes.append((first_ev, last_ev))

        # Fallback to original method if SVO didn't find changes
        if not changes:
            by_session: dict[int, list[CalendarEvent]] = {}
            for ev in relevant:
                by_session.setdefault(ev.session_idx, []).append(ev)

            session_idxs = sorted(by_session.keys())
            for i in range(len(session_idxs) - 1):
                old_events = by_session[session_idxs[i]]
                new_events = by_session[session_idxs[-1]]
                if old_events and new_events:
                    old_text = old_events[0].description.lower()
                    new_text = new_events[0].description.lower()
                    if old_text != new_text:
                        changes.append((old_events[0], new_events[0]))
                        break

        return changes


class CalendarEvent:
    """A single event on the timeline."""

    __slots__ = (
        "description", "session_date", "event_date",
        "session_id", "session_idx", "speaker", "raw_expression",
    )

    def __init__(
        self,
        description: str,
        session_date: datetime,
        event_date: datetime,
        session_id: str = "",
        session_idx: int = 0,
        speaker: str = "user",
        raw_expression: str = "",
    ):
        self.description = description
        self.session_date = session_date
        self.event_date = event_date
        self.session_id = session_id
        self.session_idx = session_idx
        self.speaker = speaker
        self.raw_expression = raw_expression


# --- Helper functions ---

def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse LongMemEval date format: '2023/05/30 (Tue) 23:40'"""
    if not date_str:
        return None
    try:
        parts = date_str.split("(")[0].strip()
        return datetime.strptime(parts, "%Y/%m/%d")
    except (ValueError, IndexError):
        return None


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text."""
    stopwords = {
        'what', 'when', 'where', 'which', 'that', 'this', 'have', 'does',
        'from', 'with', 'about', 'your', 'there', 'their', 'been', 'will',
        'would', 'could', 'should', 'much', 'many', 'some', 'more', 'than',
        'also', 'just', 'very', 'most', 'other', 'after', 'before', 'they',
        'them', 'then', 'were', 'being', 'each', 'make', 'like', 'long',
        'between', 'first', 'last', 'happened', 'days', 'weeks', 'months',
        'how', 'did', 'the', 'and', 'for', 'was', 'are', 'but', 'not',
        'you', 'all', 'can', 'had', 'her', 'one', 'our', 'out',
        'visit', 'visited', 'time', 'know', 'think', 'said', 'told',
        'want', 'need', 'help', 'look', 'come', 'take', 'give',
    }
    words = []
    for w in text.lower().split():
        w = w.strip("?,!.;:\"'()[]")
        if len(w) > 2 and w not in stopwords:
            words.append(w)
    return words


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _is_meaningful(sentence: str) -> bool:
    """Check if a sentence contains meaningful factual content."""
    lower = sentence.lower()
    # Skip greetings and filler
    filler = [
        "sure", "of course", "certainly", "that's great", "sounds good",
        "no problem", "you're welcome", "i'd be happy", "let me know",
        "is there anything", "how can i help",
    ]
    if any(lower.startswith(f) for f in filler):
        return False
    # Must have some substance
    if len(sentence.split()) < 4:
        return False
    return True


def _extract_event_date(sentence: str, session_date: datetime) -> tuple[Optional[datetime], str]:
    """Extract a specific date mentioned in a sentence.

    Returns (resolved_date, raw_expression) where raw_expression is the
    original relative date text that was matched (e.g. "last Saturday",
    "3 days ago", "Valentine's day").  Empty string when no temporal
    expression was found and the session_date is used as default.
    """
    # "on May 5th", "on January 15"
    month_match = re.search(
        r'(?:on|in|during)\s+'
        r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+(\d{1,2})(?:st|nd|rd|th)?',
        sentence, re.IGNORECASE,
    )
    if month_match:
        try:
            month_name = month_match.group(1)
            day = int(month_match.group(2))
            year = session_date.year
            dt = datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y")
            return dt, month_match.group(0)
        except ValueError:
            pass

    # "in 2019", "in 2023"
    year_match = re.search(r'\bin (20\d{2})\b', sentence)
    if year_match:
        try:
            return datetime(int(year_match.group(1)), 6, 15), year_match.group(0)  # mid-year estimate
        except ValueError:
            pass

    # "last week", "3 days ago", "yesterday"
    lower = sentence.lower()

    if "yesterday" in lower:
        return session_date - timedelta(days=1), "yesterday"

    ago_match = re.search(r'(\d+)\s+(days?|weeks?|months?)\s+ago', sentence, re.IGNORECASE)
    if ago_match:
        num = int(ago_match.group(1))
        unit = ago_match.group(2).lower().rstrip("s")
        raw = ago_match.group(0)
        if unit == "day":
            return session_date - timedelta(days=num), raw
        elif unit == "week":
            return session_date - timedelta(weeks=num), raw
        elif unit == "month":
            return session_date - timedelta(days=num * 30), raw

    if "last week" in lower:
        return session_date - timedelta(weeks=1), "last week"
    if "last month" in lower:
        return session_date - timedelta(days=30), "last month"
    if "this past weekend" in lower:
        days_since_sat = (session_date.weekday() - 5) % 7
        if days_since_sat == 0:
            days_since_sat = 7
        return session_date - timedelta(days=days_since_sat), "this past weekend"
    if "last weekend" in lower:
        # Find the most recent Saturday before session_date
        days_since_sat = (session_date.weekday() - 5) % 7
        if days_since_sat == 0:
            days_since_sat = 7  # if today is Saturday, go to last Saturday
        return session_date - timedelta(days=days_since_sat), "last weekend"

    # "last Saturday", "last Tuesday", etc.
    day_names = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    last_day_match = re.search(
        r'last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        lower,
    )
    if last_day_match:
        target_day = day_names[last_day_match.group(1)]
        days_back = (session_date.weekday() - target_day) % 7
        if days_back == 0:
            days_back = 7
        return session_date - timedelta(days=days_back), last_day_match.group(0)

    # "this Monday", "this Friday"
    this_day_match = re.search(
        r'this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        lower,
    )
    if this_day_match:
        target_day = day_names[this_day_match.group(1)]
        days_fwd = (target_day - session_date.weekday()) % 7
        return session_date + timedelta(days=days_fwd), this_day_match.group(0)

    # Named holidays — resolve to session year
    year = session_date.year
    if "valentine" in lower:
        return datetime(year, 2, 14), "Valentine's day"
    if "christmas" in lower:
        return datetime(year, 12, 25), "Christmas"
    if "new year" in lower:
        # "new year's" usually means Jan 1
        return datetime(year, 1, 1), "New Year's"
    if "halloween" in lower:
        return datetime(year, 10, 31), "Halloween"
    if "thanksgiving" in lower:
        # US Thanksgiving: 4th Thursday of November
        nov1 = datetime(year, 11, 1)
        first_thu = (3 - nov1.weekday()) % 7
        return datetime(year, 11, 1 + first_thu + 21), "Thanksgiving"
    if "independence day" in lower or "fourth of july" in lower or "4th of july" in lower:
        return datetime(year, 7, 4), "Independence Day"

    return session_date, ""  # default to session date, no raw expression
