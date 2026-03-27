"""PII and secret detection for memory content.

Detects common patterns of sensitive data: API keys, emails, credit cards,
phone numbers, SSNs, and other secrets. Can warn or redact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class PIIAction(str, Enum):
    WARN = "warn"       # Return warnings but store as-is
    REDACT = "redact"   # Replace sensitive data with [REDACTED]
    BLOCK = "block"     # Refuse to store


@dataclass
class PIIMatch:
    """A detected PII/secret pattern."""
    category: str
    pattern_name: str
    start: int
    end: int
    redacted: str  # The replacement text


# Regex patterns for sensitive data
_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    # API Keys
    ("api_key", "OpenAI API Key", re.compile(r'sk-[a-zA-Z0-9]{20,}')),
    ("api_key", "AWS Access Key", re.compile(r'AKIA[0-9A-Z]{16}')),
    ("api_key", "GitHub Token", re.compile(r'gh[ps]_[a-zA-Z0-9]{36,}')),
    ("api_key", "Anthropic Key", re.compile(r'sk-ant-[a-zA-Z0-9\-]{20,}')),
    ("api_key", "Slack Token", re.compile(r'xox[bpas]-[a-zA-Z0-9\-]+')),
    ("api_key", "Generic API Key", re.compile(r'(?:api[_-]?key|apikey|secret[_-]?key)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE)),

    # Passwords in config
    ("password", "Password Assignment", re.compile(r'(?:password|passwd|pwd)\s*[:=]\s*["\']?([^\s"\']{8,})["\']?', re.IGNORECASE)),

    # Email
    ("email", "Email Address", re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')),

    # Phone numbers (US format)
    ("phone", "Phone Number", re.compile(r'\b(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')),

    # Credit cards (basic: 13-19 digits with common separators)
    ("credit_card", "Credit Card Number", re.compile(r'\b(?:4[0-9]{3}|5[1-5][0-9]{2}|3[47][0-9]{1}|6(?:011|5[0-9]{2}))[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{1,4}\b')),

    # SSN
    ("ssn", "Social Security Number", re.compile(r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b')),

    # Private keys
    ("private_key", "Private Key", re.compile(r'-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----')),

    # Connection strings
    ("connection_string", "Database URL", re.compile(r'(?:postgres|mysql|mongodb|redis)://[^\s]+')),

    # JWT tokens
    ("jwt", "JWT Token", re.compile(r'eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}')),
]


def detect_pii(content: str) -> list[PIIMatch]:
    """Scan content for PII/secret patterns. Returns all matches."""
    matches = []
    for category, name, pattern in _PATTERNS:
        for m in pattern.finditer(content):
            # Determine what to show in redacted version
            redacted_text = f"[{category.upper()}_REDACTED]"
            matches.append(PIIMatch(
                category=category,
                pattern_name=name,
                start=m.start(),
                end=m.end(),
                redacted=redacted_text,
            ))
    return matches


def redact_content(content: str, matches: list[PIIMatch] | None = None) -> str:
    """Replace all PII matches with redacted placeholders."""
    if matches is None:
        matches = detect_pii(content)
    if not matches:
        return content

    # Sort by position descending so replacements don't shift indices
    sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
    result = content
    for m in sorted_matches:
        result = result[:m.start] + m.redacted + result[m.end:]
    return result


def pii_summary(matches: list[PIIMatch]) -> str:
    """Generate a human-readable summary of PII findings."""
    if not matches:
        return "No sensitive data detected."
    categories = {}
    for m in matches:
        categories.setdefault(m.category, []).append(m.pattern_name)
    parts = []
    for cat, names in categories.items():
        unique = set(names)
        parts.append(f"  - {cat}: {', '.join(unique)} ({len(names)} occurrence(s))")
    return "Sensitive data detected:\n" + "\n".join(parts)
