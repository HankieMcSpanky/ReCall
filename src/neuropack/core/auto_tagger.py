"""Auto-tagging: extract tags and classify memory type from content.

Uses LLM when available, falls back to keyword heuristics.
"""
from __future__ import annotations

import re
from typing import Optional

from neuropack.types import MEMORY_TYPES, STALENESS_CATEGORIES


# --- Keyword-based heuristics (zero-dependency fallback) ---

_TYPE_PATTERNS: dict[str, list[str]] = {
    "decision": [
        r"\bdecided\b", r"\bchose\b", r"\bwill use\b", r"\bgoing with\b",
        r"\bpicked\b", r"\bselected\b", r"\bsettled on\b", r"\bdecision\b",
    ],
    "fact": [
        r"\bis\b.*\bversion\b", r"\breleased\b", r"\bdefault\sto\b",
        r"\bsupports\b", r"\brequires\b", r"\bcompatible\b", r"\bequals\b",
    ],
    "preference": [
        r"\bprefer\b", r"\balways use\b", r"\bnever use\b", r"\bfavorite\b",
        r"\blike to\b", r"\bstyle\b.*\bguide\b", r"\bconvention\b",
    ],
    "procedure": [
        r"\bstep\s*\d", r"\bfirst\b.*\bthen\b", r"\bhow to\b",
        r"\bworkflow\b", r"\bprocess\b", r"\brecipe\b", r"\binstructions?\b",
    ],
    "code": [
        r"```", r"\bdef\s+\w+", r"\bclass\s+\w+", r"\bimport\s+\w+",
        r"\bfunction\b", r"\breturn\b", r"=>", r"\bconsole\.log\b",
    ],
    "observation": [
        r"\bnoticed\b", r"\bfound that\b", r"\bseems like\b", r"\bturns out\b",
        r"\blearned\b", r"\brealized\b", r"\bdiscovered\b",
    ],
}

_STALENESS_VOLATILE_PATTERNS = [
    r"\bversion\s+\d", r"\bv\d+\.\d+", r"\bprice\b", r"\bcost\b",
    r"\b\d{4}[-/]\d{2}", r"\bcurrently\b", r"\blatest\b", r"\btoday\b",
    r"\bright now\b", r"\bthis week\b", r"\bthis month\b",
]

_STALENESS_SEMI_STABLE_PATTERNS = [
    r"\bAPI\b", r"\bendpoint\b", r"\bconfig\b", r"\bsettings?\b",
    r"\bdefault\b", r"\bport\b", r"\bURL\b", r"\bschema\b",
]

# Common topic keywords for tag extraction
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "python": ["python", "pip", "pytest", "django", "flask", "fastapi", "pydantic"],
    "javascript": ["javascript", "node", "npm", "react", "vue", "typescript", "deno"],
    "rust": ["rust", "cargo", "tokio", "serde"],
    "database": ["sql", "sqlite", "postgres", "mysql", "mongodb", "redis", "database"],
    "docker": ["docker", "container", "kubernetes", "k8s", "compose"],
    "git": ["git", "github", "gitlab", "branch", "merge", "rebase", "commit"],
    "api": ["api", "rest", "graphql", "grpc", "endpoint", "webhook"],
    "security": ["security", "auth", "encryption", "token", "password", "ssl", "tls"],
    "testing": ["test", "pytest", "jest", "unittest", "coverage", "mock"],
    "devops": ["ci/cd", "pipeline", "deploy", "terraform", "ansible"],
    "ai": ["llm", "gpt", "claude", "embedding", "vector", "transformer", "model"],
    "linux": ["linux", "bash", "shell", "ubuntu", "systemd", "cron"],
    "networking": ["http", "tcp", "dns", "proxy", "nginx", "cors"],
    "performance": ["performance", "optimization", "cache", "latency", "throughput"],
    "architecture": ["architecture", "microservice", "monolith", "pattern", "design"],
}


def classify_memory_type(content: str) -> str:
    """Classify content into a memory type using keyword heuristics."""
    lower = content.lower()
    scores: dict[str, int] = {}
    for mtype, patterns in _TYPE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, lower))
        if score > 0:
            scores[mtype] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


def classify_staleness(content: str) -> str:
    """Classify how time-sensitive content is."""
    lower = content.lower()
    volatile_hits = sum(1 for p in _STALENESS_VOLATILE_PATTERNS if re.search(p, lower))
    if volatile_hits >= 2:
        return "volatile"
    semi_hits = sum(1 for p in _STALENESS_SEMI_STABLE_PATTERNS if re.search(p, lower))
    if semi_hits >= 2 or volatile_hits >= 1:
        return "semi-stable"
    return "stable"


def extract_tags(content: str, existing_tags: list[str] | None = None) -> list[str]:
    """Extract topic tags from content using keyword matching."""
    lower = content.lower()
    existing = set(existing_tags or [])
    found: list[str] = []
    for tag, keywords in _TOPIC_KEYWORDS.items():
        if tag in existing:
            continue
        if any(kw in lower for kw in keywords):
            found.append(tag)
    # Cap at 5 auto-tags
    return found[:5]


class AutoTagger:
    """Tag and classify memories. Uses LLM when available, keyword fallback otherwise."""

    def __init__(self, llm_provider=None):
        self._llm = llm_provider

    def tag_and_classify(
        self, content: str, existing_tags: list[str] | None = None,
    ) -> dict:
        """Return dict with 'tags', 'memory_type', 'staleness'."""
        # Always start with keyword heuristics (fast, reliable)
        auto_tags = extract_tags(content, existing_tags)
        memory_type = classify_memory_type(content)
        staleness = classify_staleness(content)

        # LLM enhancement (optional, best-effort)
        if self._llm is not None:
            try:
                llm_result = self._llm_classify(content)
                if llm_result.get("tags"):
                    # Merge LLM tags with keyword tags, dedup
                    seen = set(auto_tags)
                    for t in llm_result["tags"]:
                        t = t.lower().strip().replace(" ", "-")
                        if t and t not in seen and len(t) <= 30:
                            auto_tags.append(t)
                            seen.add(t)
                if llm_result.get("memory_type") in MEMORY_TYPES:
                    memory_type = llm_result["memory_type"]
                if llm_result.get("staleness") in STALENESS_CATEGORIES:
                    staleness = llm_result["staleness"]
            except Exception:
                pass  # LLM failure is fine, heuristics still work

        return {
            "tags": auto_tags[:5],
            "memory_type": memory_type,
            "staleness": staleness,
        }

    def _llm_classify(self, content: str) -> dict:
        """Use LLM to classify content. Returns partial dict."""
        import json as _json

        prompt = (
            "Classify this text. Respond with ONLY valid JSON, no markdown.\n"
            '{"tags": ["tag1", "tag2"], '
            '"memory_type": "one of: fact|decision|preference|procedure|code|observation|general", '
            '"staleness": "one of: volatile|semi-stable|stable"}\n\n'
            f"Text: {content[:500]}"
        )
        response = self._llm.complete(prompt)
        # Parse JSON from response
        text = response.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return _json.loads(text)
