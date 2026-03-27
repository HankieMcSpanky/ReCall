"""Structured Event Extraction — extracts subject-verb-object tuples from memory content.

Inspired by Chronos (95.6% on LongMemEval) which builds an "event calendar"
alongside raw memory storage for improved temporal recall.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from neuropack.storage.database import Database
from neuropack.validation import sanitize_fts_query


# ---------------------------------------------------------------------------
# Verb synonym map (heuristic, no LLM)
# ---------------------------------------------------------------------------

VERB_SYNONYMS: dict[str, list[str]] = {
    "graduated": ["got degree", "completed studies", "finished school"],
    "bought": ["purchased", "got", "acquired"],
    "purchased": ["bought", "got", "acquired"],
    "moved": ["relocated", "went to live in", "transferred to"],
    "relocated": ["moved", "went to live in", "transferred to"],
    "started": ["began", "commenced", "initiated"],
    "began": ["started", "commenced", "initiated"],
    "prefer": ["like better", "favor", "choose"],
    "prefers": ["likes better", "favors", "chooses"],
    "decided": ["chose", "picked", "selected", "went with"],
    "chose": ["decided", "picked", "selected"],
    "picked": ["decided", "chose", "selected"],
    "selected": ["decided", "chose", "picked"],
    "work": ["employed at", "job at", "career at"],
    "works": ["employed at", "job at", "career at"],
    "worked": ["was employed at", "had job at", "had career at"],
    "live": ["reside in", "stay in", "based in"],
    "lives": ["resides in", "stays in", "based in"],
    "lived": ["resided in", "stayed in", "was based in"],
    "like": ["enjoy", "fond of", "into"],
    "likes": ["enjoys", "is fond of", "is into"],
    "love": ["adore", "really like", "passionate about"],
    "loves": ["adores", "really likes", "passionate about"],
    "hate": ["dislike", "can't stand", "detest"],
    "hates": ["dislikes", "can't stand", "detests"],
    "built": ["created", "made", "developed"],
    "created": ["built", "made", "developed"],
    "joined": ["started at", "became member of", "entered"],
    "left": ["departed from", "quit", "exited"],
    "quit": ["left", "departed from", "resigned from"],
    "married": ["wed", "got married to", "tied the knot with"],
    "born": ["came into the world", "was born"],
    "studied": ["learned", "majored in", "took courses in"],
    "learned": ["studied", "picked up", "acquired knowledge of"],
    "traveled": ["visited", "went to", "journeyed to"],
    "visited": ["traveled to", "went to", "stopped by"],
    "adopted": ["took in", "got", "welcomed"],
    "named": ["called", "known as", "goes by"],
    "is_allergic_to": ["has allergy to", "can't have", "reacts to"],
    "has_dog_named": ["owns a dog called", "dog is named", "has a dog called"],
    "has_cat_named": ["owns a cat called", "cat is named", "has a cat called"],
    "has_pet_named": ["owns a pet called", "pet is named", "has a pet called"],
    "favorite_color": ["preferred color", "best-liked color", "color preference"],
    "favorite_food": ["preferred food", "best-liked food", "go-to food"],
    "favorite_movie": ["preferred movie", "best-liked movie", "top movie"],
    "favorite_book": ["preferred book", "best-liked book", "top book"],
    "moved_to": ["relocated to", "went to live in", "transferred to"],
}


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# Date patterns for temporal extraction
_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
)

_DATE_PATTERNS = [
    # "in 2020", "in 2020-01", "in January 2020"
    re.compile(
        r"\bin\s+(?:(?:" + _MONTH_NAMES + r")\s+)?(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?\b",
        re.IGNORECASE,
    ),
    # "on January 5, 2020", "on 5 January 2020"
    re.compile(
        r"\bon\s+(?:(" + _MONTH_NAMES + r")\s+(\d{1,2}),?\s+(\d{4})|(\d{1,2})\s+(" + _MONTH_NAMES + r")\s+(\d{4}))\b",
        re.IGNORECASE,
    ),
    # "last week", "last month", "last year", "yesterday"
    re.compile(r"\b(last\s+(?:week|month|year)|yesterday|today)\b", re.IGNORECASE),
    # "2020-01-05"
    re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2})\b"),
    # "since 2020", "from 2020"
    re.compile(r"\b(?:since|from)\s+(\d{4})\b", re.IGNORECASE),
]


_MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def _resolve_relative_date(text: str) -> str | None:
    """Resolve relative date references to ISO date strings."""
    low = text.lower().strip()
    now = datetime.now(timezone.utc)
    if low == "yesterday":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    if low == "today":
        return now.strftime("%Y-%m-%d")
    if low == "last week":
        return (now - timedelta(weeks=1)).strftime("%Y-%m-%d")
    if low == "last month":
        m = now.month - 1 or 12
        y = now.year if now.month > 1 else now.year - 1
        return f"{y}-{m:02d}"
    if low == "last year":
        return str(now.year - 1)
    return None


def _extract_date_from_text(text: str) -> str | None:
    """Extract date from a piece of text, return ISO-ish string or None."""
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if m is None:
            continue
        groups = m.groups()
        full_match = m.group(0)

        # Relative dates
        if any(kw in full_match.lower() for kw in ("last ", "yesterday", "today")):
            return _resolve_relative_date(full_match.strip())

        # ISO date 2020-01-05
        if re.match(r"\d{4}-\d{1,2}-\d{1,2}", groups[0] if groups[0] else ""):
            return groups[0]

        # "in 2020" or "in January 2020" or "since 2020"
        # Try to build a date from the first match
        year = None
        month = None
        day = None
        for g in groups:
            if g is None:
                continue
            gl = g.lower()
            if gl in _MONTH_MAP:
                month = _MONTH_MAP[gl]
            elif g.isdigit():
                v = int(g)
                if v > 1900:
                    year = v
                elif v <= 31 and day is None and year is not None:
                    day = v
                elif v <= 31:
                    day = v
                elif v > 31:
                    year = v
        if year:
            parts = [str(year)]
            if month:
                parts.append(f"{month:02d}")
            if day:
                parts.append(f"{day:02d}")
            return "-".join(parts)

    return None


# ---------------------------------------------------------------------------
# SVO extraction patterns
# ---------------------------------------------------------------------------

# Each pattern is (compiled_regex, subject_group, verb_str, object_group)
# or a callable that returns list of (subject, verb, obj, leftover_text_for_date)

def _build_svo_patterns() -> list:
    """Build compiled SVO extraction patterns."""
    patterns = []

    # --- Action verbs: "I/we/he/she/they <verb> <object>" ---
    action_verbs = [
        "graduated", "bought", "purchased", "started", "began", "moved",
        "relocated", "decided", "chose", "picked", "selected", "joined",
        "left", "quit", "built", "created", "made", "developed", "adopted",
        "married", "studied", "learned", "traveled", "visited",
        "sold", "wrote", "published", "launched", "founded", "installed",
        "deleted", "removed", "fixed", "broke", "lost", "found",
        "earned", "won", "received", "completed", "finished",
    ]
    verb_alt = "|".join(action_verbs)
    patterns.append((
        re.compile(
            r"\b(I|we|he|she|they|my\s+\w+)\s+(" + verb_alt + r")\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "action",
    ))

    # --- Present tense actions: "I work at X", "I live in X" ---
    patterns.append((
        re.compile(
            r"\b(I|we|he|she|they)\s+(work|live|teach|study|volunteer)\s+"
            r"(?:at|in|for|with)\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "present_action",
    ))

    # --- "I am <adjective/noun>" patterns ---
    patterns.append((
        re.compile(
            r"\b(I)\s+am\s+(?:a\s+|an\s+)?(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "identity",
    ))

    # --- "I have <object>" ---
    patterns.append((
        re.compile(
            r"\b(I|we)\s+have\s+(?:a\s+|an\s+)?(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "have",
    ))

    # --- "I was <something>" ---
    patterns.append((
        re.compile(
            r"\b(I|he|she)\s+was\s+(?:a\s+|an\s+)?(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "was",
    ))

    # --- Preferences: "I prefer X over Y", "I like X", "my favorite X is Y" ---
    patterns.append((
        re.compile(
            r"\b(I|we)\s+(prefer|like|love|enjoy|hate|dislike|avoid)\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "preference",
    ))

    patterns.append((
        re.compile(
            r"\bmy\s+favorite\s+(\w+)\s+is\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "favorite",
    ))

    # --- Facts: "my X is Y" ---
    patterns.append((
        re.compile(
            r"\bmy\s+(\w+(?:'s)?)\s+(?:name\s+)?is\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "my_is",
    ))

    # --- Allergies: "allergic to X" ---
    patterns.append((
        re.compile(
            r"\b(?:I(?:'m|\s+am)\s+)?allergic\s+to\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "allergic",
    ))

    # --- Decisions: "we decided to X", "we chose to X" ---
    patterns.append((
        re.compile(
            r"\b(I|we|they|the\s+team)\s+(decided|chose|picked|selected|agreed)\s+to\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "decision",
    ))

    # --- "I moved to X" ---
    patterns.append((
        re.compile(
            r"\b(I|we|he|she|they)\s+moved\s+to\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "moved_to",
    ))

    # --- "My dog/cat/pet's name is X" or "My dog/cat/pet is named X" ---
    patterns.append((
        re.compile(
            r"\bmy\s+(dog|cat|pet|bird|fish|hamster|rabbit)(?:'s\s+name)?\s+is\s+(?:named\s+)?(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ),
        "pet_named",
    ))

    return patterns


_SVO_PATTERNS = _build_svo_patterns()


def _extract_svos(text: str) -> list[dict]:
    """Extract subject-verb-object tuples from text using regex patterns.

    Returns list of dicts with keys: subject, verb, object, date, confidence.
    """
    results: list[dict] = []
    seen_spans: list[tuple[int, int]] = []

    # Split into sentences for better extraction
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 5:
            continue

        date = _extract_date_from_text(sentence)

        for pattern, ptype in _SVO_PATTERNS:
            for m in pattern.finditer(sentence):
                # Avoid overlapping matches
                span = (m.start(), m.end())
                if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                    continue

                event = _match_to_event(m, ptype, date)
                if event is not None:
                    seen_spans.append(span)
                    results.append(event)

    return results


def _clean(s: str) -> str:
    """Strip and clean extracted text."""
    s = s.strip().rstrip(".,;:!?")
    # Remove trailing prepositions/articles that got captured
    s = re.sub(r'\s+(in|on|at|to|from|for|with|the|a|an)$', '', s, flags=re.IGNORECASE)
    return s.strip()


def _match_to_event(
    m: re.Match, ptype: str, date: str | None,
) -> dict | None:
    """Convert a regex match to an event dict based on pattern type."""
    confidence = 0.7  # default

    if ptype == "action":
        subject = _normalize_subject(m.group(1))
        verb = m.group(2).lower()
        obj = _clean(m.group(3))
        if len(obj) < 2:
            return None
        confidence = 0.8

    elif ptype == "present_action":
        subject = _normalize_subject(m.group(1))
        raw_verb = m.group(2).lower()
        obj = _clean(m.group(3))
        # Reconstruct verb with preposition
        # Find the preposition between verb and object in original text
        after_verb = m.group(0)[m.group(0).lower().find(raw_verb) + len(raw_verb):]
        prep_match = re.match(r'\s+(at|in|for|with)\s+', after_verb, re.IGNORECASE)
        prep = prep_match.group(1) if prep_match else "at"
        verb = f"{raw_verb}_{prep}"
        if len(obj) < 2:
            return None
        confidence = 0.75

    elif ptype == "identity":
        subject = "user"
        verb = "is"
        obj = _clean(m.group(2))
        if len(obj) < 2:
            return None
        confidence = 0.7

    elif ptype == "have":
        subject = _normalize_subject(m.group(1))
        verb = "has"
        obj = _clean(m.group(2))
        if len(obj) < 2:
            return None
        confidence = 0.65

    elif ptype == "was":
        subject = _normalize_subject(m.group(1))
        verb = "was"
        obj = _clean(m.group(2))
        if len(obj) < 2:
            return None
        confidence = 0.65

    elif ptype == "preference":
        subject = _normalize_subject(m.group(1))
        verb = m.group(2).lower()
        obj = _clean(m.group(3))
        if len(obj) < 2:
            return None
        confidence = 0.75

    elif ptype == "favorite":
        subject = "user"
        category = m.group(1).lower()
        value = _clean(m.group(2))
        verb = f"favorite_{category}"
        obj = value
        if len(obj) < 1:
            return None
        confidence = 0.85

    elif ptype == "my_is":
        subject = "user"
        what = m.group(1).lower()
        value = _clean(m.group(2))
        # Handle possessives: "my dog's name is Max"
        if what.endswith("'s"):
            what = what[:-2]
        verb = f"has_{what}_named" if what in ("dog", "cat", "pet", "bird", "fish") else what
        obj = value
        if len(obj) < 1:
            return None
        confidence = 0.8

    elif ptype == "allergic":
        subject = "user"
        verb = "is_allergic_to"
        obj = _clean(m.group(1))
        if len(obj) < 2:
            return None
        confidence = 0.85

    elif ptype == "decision":
        subject = _normalize_subject(m.group(1))
        verb = m.group(2).lower()
        obj = _clean(m.group(3))
        if len(obj) < 2:
            return None
        confidence = 0.8

    elif ptype == "moved_to":
        subject = _normalize_subject(m.group(1))
        verb = "moved_to"
        obj = _clean(m.group(2))
        if len(obj) < 2:
            return None
        confidence = 0.8

    elif ptype == "pet_named":
        subject = "user"
        pet_type = m.group(1).lower()
        name = _clean(m.group(2))
        verb = f"has_{pet_type}_named"
        obj = name
        if len(obj) < 1:
            return None
        confidence = 0.85

    else:
        return None

    return {
        "subject": subject,
        "verb": verb,
        "object": obj,
        "date": date,
        "confidence": confidence,
    }


def _normalize_subject(raw: str) -> str:
    """Normalize subject pronoun to canonical form."""
    low = raw.strip().lower()
    if low in ("i", "i'm", "i've", "i'd"):
        return "user"
    if low in ("we", "we're", "we've"):
        return "team"
    if low in ("he", "she"):
        return low
    if low.startswith("my "):
        return "user"
    if low.startswith("the "):
        return low[4:]
    return low


# ---------------------------------------------------------------------------
# EventExtractor
# ---------------------------------------------------------------------------

class EventExtractor:
    """Extracts structured SVO events from memory content and stores them
    in the ``memory_events`` / ``events_fts`` tables for calendar-style
    retrieval."""

    def __init__(self, db: Database):
        self._db = db

    # -- public API --

    def extract_events(
        self, content: str, source_memory_id: str,
    ) -> list[dict]:
        """Extract structured events from *content*.

        Returns a list of event dicts ready for ``store_events``:
        ``{id, memory_id, subject, verb, object, event_date, aliases, confidence}``.
        """
        raw_events = _extract_svos(content)
        events: list[dict] = []
        for ev in raw_events:
            aliases = self._generate_aliases(ev["subject"], ev["verb"], ev["object"])
            events.append({
                "id": uuid.uuid4().hex,
                "memory_id": source_memory_id,
                "subject": ev["subject"],
                "verb": ev["verb"],
                "object": ev["object"],
                "event_date": ev.get("date"),
                "event_date_end": None,
                "aliases": aliases,
                "confidence": ev.get("confidence", 0.5),
            })
        return events

    def store_events(
        self,
        events: list[dict],
        memory_id: str,
        namespace: str = "default",
    ) -> None:
        """Persist extracted events and keep FTS in sync."""
        if not events:
            return

        now = datetime.now(timezone.utc).isoformat()
        conn = self._db.connect()

        for ev in events:
            aliases_json = json.dumps(ev["aliases"])
            with self._db.transaction() as tx:
                tx.execute(
                    """INSERT INTO memory_events
                       (id, memory_id, subject, verb, object, event_date,
                        event_date_end, aliases, confidence, namespace, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ev["id"],
                        memory_id,
                        ev["subject"],
                        ev["verb"],
                        ev["object"],
                        ev.get("event_date"),
                        ev.get("event_date_end"),
                        aliases_json,
                        ev.get("confidence", 0.5),
                        namespace,
                        now,
                    ),
                )

    def delete_events_for_memory(self, memory_id: str) -> None:
        """Remove all events associated with a memory (cascade helper)."""
        with self._db.transaction() as tx:
            tx.execute("DELETE FROM memory_events WHERE memory_id = ?", (memory_id,))

    def search_events(
        self,
        query: str,
        limit: int = 20,
        namespace: str | None = None,
    ) -> list[tuple[str, float]]:
        """Search events via FTS5 and optional date-range filtering.

        Returns ``(memory_id, score)`` tuples.
        """
        import sqlite3

        safe_query = sanitize_fts_query(query)
        if not safe_query:
            return []

        conn = self._db.connect()

        # --- Date-range filter (heuristic) ---
        date_filter_clause = ""
        date_params: list[object] = []
        extracted_date = _extract_date_from_text(query)
        if extracted_date:
            # If we got a year only, search whole year
            parts = extracted_date.split("-")
            if len(parts) == 1:
                date_filter_clause = " AND e.event_date LIKE ?"
                date_params = [f"{parts[0]}%"]
            elif len(parts) == 2:
                date_filter_clause = " AND e.event_date LIKE ?"
                date_params = [f"{parts[0]}-{parts[1]}%"]
            else:
                date_filter_clause = " AND e.event_date = ?"
                date_params = [extracted_date]

        try:
            if namespace:
                rows = conn.execute(
                    f"""SELECT e.memory_id, fts.rank
                        FROM events_fts fts
                        JOIN memory_events e ON e.rowid = fts.rowid
                        WHERE events_fts MATCH ?
                          AND e.namespace = ?
                          {date_filter_clause}
                        ORDER BY fts.rank
                        LIMIT ?""",
                    (safe_query, namespace, *date_params, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""SELECT e.memory_id, fts.rank
                        FROM events_fts fts
                        JOIN memory_events e ON e.rowid = fts.rowid
                        WHERE events_fts MATCH ?
                          {date_filter_clause}
                        ORDER BY fts.rank
                        LIMIT ?""",
                    (safe_query, *date_params, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []

        # Deduplicate by memory_id, keep best rank
        best: dict[str, float] = {}
        for r in rows:
            d = dict(r)
            mid = d["memory_id"]
            rank = d["rank"]
            if mid not in best or rank < best[mid]:
                best[mid] = rank

        return [(mid, rank) for mid, rank in best.items()]

    def get_events_for_memory(self, memory_id: str) -> list[dict]:
        """Return all events linked to a memory."""
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT * FROM memory_events WHERE memory_id = ? ORDER BY created_at",
            (memory_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["aliases"] = json.loads(d["aliases"])
            results.append(d)
        return results

    # -- alias generation --

    def _generate_aliases(
        self, subject: str, verb: str, obj: str,
    ) -> list[str]:
        """Generate 2-4 lexical aliases for an SVO event (no LLM)."""
        aliases: list[str] = []

        # 1. Replace verb with synonyms
        verb_key = verb.lower().replace("_", " ").strip()
        # Also try without trailing prepositions for lookup
        verb_base = re.split(r'[_ ](?:to|at|in|for|with)$', verb_key)[0]

        synonyms = VERB_SYNONYMS.get(verb_key, VERB_SYNONYMS.get(verb_base, []))
        for syn in synonyms[:2]:
            alias = f"{subject} {syn} {obj}"
            aliases.append(alias.strip())

        # 2. Object-first rephrasing: "<obj> — <verb> by <subject>"
        if subject not in ("user", "team"):
            aliases.append(f"{obj} {verb.replace('_', ' ')} by {subject}")
        else:
            aliases.append(f"{obj} was {verb.replace('_', ' ')}")

        # 3. Simplified version without subject
        simple = f"{verb.replace('_', ' ')} {obj}"
        aliases.append(simple)

        # Deduplicate and limit
        seen: set[str] = set()
        unique: list[str] = []
        for a in aliases:
            a_low = a.lower().strip()
            if a_low and a_low not in seen:
                seen.add(a_low)
                unique.append(a)
        return unique[:4]
