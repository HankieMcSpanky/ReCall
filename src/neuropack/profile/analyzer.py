"""Analyzer that builds a DeveloperProfile from memory content."""
from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timezone

from neuropack.profile.heuristics import (
    detect_docstring_style,
    detect_error_style,
    detect_import_style,
    detect_naming_style,
    detect_type_hint_usage,
    extract_library_mentions,
)
from neuropack.profile.models import DeveloperProfile
from neuropack.types import MemoryRecord

logger = logging.getLogger(__name__)


class DeveloperProfileAnalyzer:
    """Orchestrates analysis of memories to build a statistical developer profile."""

    def analyze(self, memories: list[MemoryRecord]) -> DeveloperProfile:
        """Build a full DeveloperProfile from a list of memories.

        Runs all sub-analyses across memories and aggregates results.
        """
        if not memories:
            return DeveloperProfile(
                last_updated=datetime.now(timezone.utc).isoformat(),
                confidence=0.0,
                evidence_count=0,
            )

        naming = self._analyze_naming(memories)
        architecture = self._analyze_architecture(memories)
        error_handling = self._analyze_error_handling(memories)
        libraries = self._analyze_libraries(memories)
        code_style = self._analyze_code_style(memories)
        review_feedback = self._analyze_review_feedback(memories)
        anti_patterns = self._analyze_anti_patterns(memories)

        # Calculate confidence based on evidence quantity and diversity
        evidence_count = len(memories)
        code_memories = sum(
            1 for m in memories
            if m.memory_type == "code" or "code" in (m.tags or [])
        )
        decision_memories = sum(
            1 for m in memories
            if m.memory_type == "decision" or "decision" in (m.tags or [])
        )
        diversity_score = min(1.0, (code_memories + decision_memories) / max(evidence_count, 1))
        quantity_score = min(1.0, evidence_count / 50.0)
        confidence = round((diversity_score * 0.6 + quantity_score * 0.4), 2)

        return DeveloperProfile(
            naming_conventions=naming,
            architecture_patterns=architecture,
            error_handling=error_handling,
            preferred_libraries=libraries,
            code_style=code_style,
            review_feedback=review_feedback,
            anti_patterns=anti_patterns,
            last_updated=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            evidence_count=evidence_count,
        )

    def _analyze_naming(self, memories: list[MemoryRecord]) -> dict:
        """Aggregate naming convention patterns across all code memories."""
        all_counts: Counter = Counter()
        all_prefixes: Counter = Counter()

        for memory in memories:
            if not self._is_code_related(memory):
                continue
            result = detect_naming_style(memory.content)
            counts = result.get("counts", {})
            for style, count in counts.items():
                all_counts[style] += count
            for prefix in result.get("common_prefixes", []):
                all_prefixes[prefix] += 1

        total = sum(all_counts.values())
        if total == 0:
            return {
                "variable_style": "unknown",
                "class_style": "unknown",
                "file_style": "unknown",
                "common_prefixes": [],
            }

        dominant = all_counts.most_common(1)[0][0]
        return {
            "variable_style": dominant if dominant != "PascalCase" else (
                all_counts.most_common(2)[1][0] if len(all_counts) > 1 else "snake_case"
            ),
            "class_style": "PascalCase" if all_counts.get("PascalCase", 0) > 0 else dominant,
            "file_style": "snake_case" if all_counts.get("snake_case", 0) >= all_counts.get("camelCase", 0) else "camelCase",
            "common_prefixes": [p for p, _ in all_prefixes.most_common(10)],
        }

    def _analyze_architecture(self, memories: list[MemoryRecord]) -> list[str]:
        """Detect architecture patterns mentioned across memories."""
        pattern_keywords = {
            "MVC": [r'\bMVC\b', r'\bmodel.view.controller\b'],
            "microservices": [r'\bmicroservice', r'\bservice.mesh\b', r'\bAPI.gateway\b'],
            "monolith": [r'\bmonolith\b', r'\bmonolithic\b'],
            "event-driven": [r'\bevent.driven\b', r'\bevent.sourcing\b', r'\bpub.?sub\b', r'\bmessage.queue\b'],
            "layered": [r'\blayered\b', r'\bthree.tier\b', r'\bn.tier\b'],
            "hexagonal": [r'\bhexagonal\b', r'\bports.and.adapters\b'],
            "CQRS": [r'\bCQRS\b', r'\bcommand.query\b'],
            "repository pattern": [r'\brepository.pattern\b', r'\bdata.access.layer\b'],
            "dependency injection": [r'\bdependency.injection\b', r'\bDI\b', r'\bIoC\b'],
            "facade pattern": [r'\bfacade\b'],
            "factory pattern": [r'\bfactory\b'],
            "singleton": [r'\bsingleton\b'],
            "observer pattern": [r'\bobserver.pattern\b', r'\bevent.listener\b'],
            "REST API": [r'\bREST\b', r'\bRESTful\b'],
            "GraphQL": [r'\bGraphQL\b'],
            "serverless": [r'\bserverless\b', r'\blambda\b', r'\bcloud.function\b'],
            "clean architecture": [r'\bclean.architecture\b', r'\buse.case\b'],
            "domain-driven design": [r'\bDDD\b', r'\bdomain.driven\b', r'\bbounded.context\b'],
            "TDD": [r'\bTDD\b', r'\btest.driven\b'],
            "CI/CD": [r'\bCI/?CD\b', r'\bcontinuous.integration\b', r'\bcontinuous.delivery\b'],
        }

        all_text = " ".join(m.content for m in memories)
        detected: list[str] = []

        for pattern_name, regexes in pattern_keywords.items():
            count = 0
            for regex in regexes:
                count += len(re.findall(regex, all_text, re.IGNORECASE))
            if count >= 2:
                detected.append(pattern_name)

        return detected

    def _analyze_error_handling(self, memories: list[MemoryRecord]) -> dict:
        """Aggregate error handling patterns across code memories."""
        style_counts: Counter = Counter()
        patterns: list[str] = []
        examples: list[str] = []

        for memory in memories:
            if not self._is_code_related(memory):
                continue
            style = detect_error_style(memory.content)
            if style != "unknown":
                style_counts[style] += 1

            # Extract specific error handling patterns
            try_blocks = re.findall(
                r'(try:\s*\n(?:.*\n)*?\s*except\s+\w+.*?:.*)',
                memory.content,
            )
            for block in try_blocks[:2]:
                snippet = block.strip()[:200]
                if snippet and len(examples) < 10:
                    examples.append(snippet)

            # Look for custom exception classes
            custom_exc = re.findall(r'class\s+(\w+(?:Error|Exception))\b', memory.content)
            for exc in custom_exc:
                if exc not in patterns:
                    patterns.append(f"custom: {exc}")

        dominant_style = "unknown"
        if style_counts:
            dominant_style = style_counts.most_common(1)[0][0]

        # Add pattern descriptions
        if style_counts.get("try/except", 0) > 0:
            patterns.append("try/except blocks")
        if style_counts.get("Result types", 0) > 0:
            patterns.append("Result type returns")
        if style_counts.get("assertions", 0) > 0:
            patterns.append("assertion guards")
        if style_counts.get("custom exceptions", 0) > 0:
            patterns.append("custom exception hierarchy")

        return {
            "style": dominant_style,
            "patterns": list(dict.fromkeys(patterns)),  # deduplicate preserving order
            "examples": examples[:5],
        }

    def _analyze_libraries(self, memories: list[MemoryRecord]) -> dict[str, dict]:
        """Build library preference map from all memories."""
        lib_counter: Counter = Counter()
        lib_contexts: dict[str, list[str]] = {}

        for memory in memories:
            libs = extract_library_mentions(memory.content)
            for lib in libs:
                normalized = lib.lower().replace("-", "_")
                lib_counter[normalized] += 1
                if normalized not in lib_contexts:
                    lib_contexts[normalized] = []
                # Capture context: the tags or type of memory where this lib appears
                context_hint = memory.memory_type or "general"
                if context_hint not in lib_contexts[normalized]:
                    lib_contexts[normalized].append(context_hint)

        # Look for explicit rejections ("instead of X", "avoid X", "not X")
        rejection_pattern = re.compile(
            r'(?:instead\s+of|avoid|not\s+using|replaced|switched\s+from)\s+([\w\-]+)',
            re.IGNORECASE,
        )
        rejected: set[str] = set()
        for memory in memories:
            for match in rejection_pattern.finditer(memory.content):
                rejected.add(match.group(1).lower().replace("-", "_"))

        result: dict[str, dict] = {}
        for lib, freq in lib_counter.most_common(30):
            entry: dict = {
                "frequency": freq,
                "context": lib_contexts.get(lib, []),
            }
            if lib in rejected:
                entry["alternatives_rejected"] = False
            else:
                # Check if any rejected library is an alternative
                entry["alternatives_rejected"] = [r for r in rejected if r != lib][:5]
            result[lib] = entry

        return result

    def _analyze_code_style(self, memories: list[MemoryRecord]) -> dict:
        """Aggregate code style preferences across memories."""
        import_styles: Counter = Counter()
        docstring_styles: Counter = Counter()
        type_hint_levels: Counter = Counter()
        line_lengths: list[int] = []

        for memory in memories:
            if not self._is_code_related(memory):
                continue

            # Import style
            imp_style = detect_import_style(memory.content)
            if imp_style != "none":
                import_styles[imp_style] += 1

            # Docstring style
            doc_style = detect_docstring_style(memory.content)
            if doc_style != "none":
                docstring_styles[doc_style] += 1

            # Type hints
            hint_level = detect_type_hint_usage(memory.content)
            if hint_level != "none":
                type_hint_levels[hint_level] += 1

            # Line length preference (sample max line lengths)
            lines = memory.content.splitlines()
            if lines:
                max_line = max(len(line) for line in lines)
                line_lengths.append(max_line)

        # Determine line length preference
        line_pref = "unknown"
        if line_lengths:
            avg_max = sum(line_lengths) / len(line_lengths)
            if avg_max <= 80:
                line_pref = "80"
            elif avg_max <= 100:
                line_pref = "100"
            elif avg_max <= 120:
                line_pref = "120"
            else:
                line_pref = "no-limit"

        return {
            "line_length_pref": line_pref,
            "import_style": import_styles.most_common(1)[0][0] if import_styles else "unknown",
            "docstring_style": docstring_styles.most_common(1)[0][0] if docstring_styles else "unknown",
            "type_hints_usage": type_hint_levels.most_common(1)[0][0] if type_hint_levels else "unknown",
        }

    def _analyze_review_feedback(self, memories: list[MemoryRecord]) -> list[str]:
        """Extract recurring review themes from feedback-related memories."""
        feedback_keywords = {
            "naming consistency": [r'\bnaming\b', r'\bconsisten(?:t|cy)\b.*\bnam'],
            "missing tests": [r'\bmissing\s+tests?\b', r'\bno\s+tests?\b', r'\btest\s+coverage\b'],
            "error handling gaps": [r'\berror\s+handling\b', r'\bunhandled\b', r'\bmissing\s+except\b'],
            "documentation needed": [r'\bdocument(?:ation)?\b.*\bmissing\b', r'\bneeds?\s+docs?\b'],
            "type safety": [r'\btype\s+(?:safety|hint|annotation)\b', r'\bmissing\s+type\b'],
            "code duplication": [r'\bduplicat(?:e|ion)\b', r'\bDRY\b', r'\brepeat(?:ed|ing)\b'],
            "complexity concerns": [r'\bcomplex(?:ity)?\b', r'\bsimplif(?:y|ied)\b', r'\brefactor\b'],
            "performance issues": [r'\bperformance\b', r'\bslow\b', r'\boptimiz(?:e|ation)\b'],
            "security concerns": [r'\bsecurity\b', r'\bvulnerab(?:le|ility)\b', r'\binjection\b'],
            "API design": [r'\bAPI\s+design\b', r'\binterface\s+design\b', r'\bcontract\b'],
        }

        # Focus on review/feedback tagged memories
        relevant = [
            m for m in memories
            if any(t in (m.tags or []) for t in ("review", "feedback", "code-review", "pr"))
            or m.memory_type in ("observation", "decision")
        ]
        # Fall back to all memories if no specific review memories exist
        if not relevant:
            relevant = memories

        all_text = " ".join(m.content for m in relevant)
        themes: list[str] = []

        for theme, patterns in feedback_keywords.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, all_text, re.IGNORECASE))
            if count >= 2:
                themes.append(theme)

        return themes

    def _analyze_anti_patterns(self, memories: list[MemoryRecord]) -> list[str]:
        """Detect things the developer explicitly avoids."""
        avoidance_phrases = re.compile(
            r'(?:avoid|don\'t\s+use|never\s+use|stopped\s+using|moved\s+away\s+from|'
            r'replaced|deprecated|anti.?pattern|bad\s+practice|shouldn\'t|'
            r'prefer\s+not|dislike|hate|problematic)',
            re.IGNORECASE,
        )

        anti_patterns: list[str] = []

        for memory in memories:
            # Look for explicit avoidance statements
            sentences = re.split(r'[.!?\n]', memory.content)
            for sentence in sentences:
                if avoidance_phrases.search(sentence):
                    cleaned = sentence.strip()
                    if 10 < len(cleaned) < 200:
                        anti_patterns.append(cleaned)

            # Check for memories tagged with negative indicators
            tags = memory.tags or []
            if any(t in tags for t in ("mistake", "anti-pattern", "avoid", "bad-practice")):
                abstract = memory.l3_abstract or memory.content[:100]
                if abstract and abstract not in anti_patterns:
                    anti_patterns.append(abstract.strip())

        # Deduplicate and limit
        seen: set[str] = set()
        unique: list[str] = []
        for ap in anti_patterns:
            normalized = ap.lower()[:50]
            if normalized not in seen:
                seen.add(normalized)
                unique.append(ap)
        return unique[:20]

    @staticmethod
    def _is_code_related(memory: MemoryRecord) -> bool:
        """Check if a memory likely contains code or code discussion."""
        if memory.memory_type == "code":
            return True
        tags = memory.tags or []
        if any(t in tags for t in ("code", "snippet", "implementation", "programming")):
            return True
        # Heuristic: contains code-like patterns
        content = memory.content
        code_indicators = [
            r'def\s+\w+\s*\(', r'class\s+\w+', r'import\s+\w+',
            r'function\s+\w+', r'const\s+\w+\s*=', r'let\s+\w+\s*=',
            r'if\s*\(.*\)\s*\{', r'for\s*\(.*\)\s*\{',
        ]
        return any(re.search(pat, content) for pat in code_indicators)
