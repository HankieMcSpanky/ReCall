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

import re
from datetime import datetime, timedelta
from typing import Optional


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
                    event_date = _extract_event_date(sentence, session_date)

                    # Only add meaningful sentences (not greetings/filler)
                    if _is_meaningful(sentence):
                        self.add_event(
                            description=sentence,
                            session_date=session_date,
                            event_date=event_date,
                            session_id=sid,
                            session_idx=i,
                            speaker=role,
                        )

    def query(self, question: str, question_date: Optional[datetime] = None) -> str:
        """Answer a temporal question using the calendar.

        Returns a pre-computed answer string, or "" if can't answer.
        """
        q_lower = question.lower()
        keywords = _extract_keywords(question)
        # Use provided question_date or fall back to latest session date
        if question_date is None and self.events:
            question_date = max(e.session_date for e in self.events)

        # Find relevant events
        relevant = self._find_relevant_events(keywords)
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

        # --- "Which happened first?" / "before or after" ---
        order_match = re.search(
            r'(which.*first)|(before|after)\s+(my|the|i)|(happened.*first)|(more recent)'
            r'|(graduated.*first)|(came.*first)|(did.*first)',
            q_lower,
        )
        if order_match and len(relevant) >= 2:
            result_parts.append(
                f"CALENDAR ANSWER: Chronological order:"
            )
            for i, ev in enumerate(relevant[:5], 1):
                result_parts.append(
                    f"  {i}. {ev.event_date.strftime('%Y/%m/%d')} — {ev.description[:80]} "
                    f"(Session {ev.session_idx + 1})"
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
                result_parts.append(
                    f"  {i}. {ev.event_date.strftime('%Y/%m/%d')} — {ev.description[:80]} "
                    f"(Session {ev.session_idx + 1})"
                )

        # --- "How long had I been X when Y?" ---
        how_long_when_match = re.search(
            r'how long (?:had|have) (?:i|the user) been\s+(.+?)\s+when',
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

        # --- "Who X first/second/third" ---
        who_order_match = re.search(
            r'who\s+(?:\w+\s+)?(first|second|third|earliest|latest)',
            q_lower,
        )
        if who_order_match and len(relevant) >= 2 and not result_parts:
            result_parts.append("CALENDAR ANSWER: Chronological order:")
            for i, ev in enumerate(relevant[:5], 1):
                result_parts.append(
                    f"  {i}. {ev.event_date.strftime('%Y/%m/%d')} — {ev.description[:80]}"
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
                result_parts.append(
                    f"  {ev.event_date.strftime('%Y/%m/%d')} — {ev.description[:100]} "
                    f"[{ev.speaker}] (Session {ev.session_idx + 1}){days_ago}"
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
        "session_id", "session_idx", "speaker",
    )

    def __init__(
        self,
        description: str,
        session_date: datetime,
        event_date: datetime,
        session_id: str = "",
        session_idx: int = 0,
        speaker: str = "user",
    ):
        self.description = description
        self.session_date = session_date
        self.event_date = event_date
        self.session_id = session_id
        self.session_idx = session_idx
        self.speaker = speaker


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


def _extract_event_date(sentence: str, session_date: datetime) -> Optional[datetime]:
    """Extract a specific date mentioned in a sentence."""
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
            return dt
        except ValueError:
            pass

    # "in 2019", "in 2023"
    year_match = re.search(r'\bin (20\d{2})\b', sentence)
    if year_match:
        try:
            return datetime(int(year_match.group(1)), 6, 15)  # mid-year estimate
        except ValueError:
            pass

    # "last week", "3 days ago", "yesterday"
    if "yesterday" in sentence.lower():
        return session_date - timedelta(days=1)

    ago_match = re.search(r'(\d+)\s+(days?|weeks?|months?)\s+ago', sentence, re.IGNORECASE)
    if ago_match:
        num = int(ago_match.group(1))
        unit = ago_match.group(2).lower().rstrip("s")
        if unit == "day":
            return session_date - timedelta(days=num)
        elif unit == "week":
            return session_date - timedelta(weeks=num)
        elif unit == "month":
            return session_date - timedelta(days=num * 30)

    if "last week" in sentence.lower():
        return session_date - timedelta(weeks=1)
    if "last month" in sentence.lower():
        return session_date - timedelta(days=30)

    return session_date  # default to session date
