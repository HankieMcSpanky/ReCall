from __future__ import annotations

import logging
import os
from collections import defaultdict

from neuropack.watcher.events import ActivityEvent

logger = logging.getLogger(__name__)

# Map file extensions to language names
EXTENSION_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java",
    ".c": "c", ".cpp": "c++", ".h": "c",
    ".jsx": "react", ".tsx": "react typescript",
    ".vue": "vue", ".rb": "ruby",
    ".md": "documentation", ".yaml": "configuration",
    ".json": "configuration", ".toml": "configuration",
}


class QueryDeriver:
    """Convert recent activity events into recall queries."""

    def derive_queries(self, events: list[ActivityEvent]) -> list[str]:
        """Derive 1-5 recall queries from recent activity events.

        Strategy:
        - File events: extract project name, module/component, language
        - Git events: use commit messages, branch names
        - Terminal events: extract tool names, flags, package names
        """
        if not events:
            return []

        queries: list[str] = []

        # Group file events by project/area
        file_queries = self._derive_file_queries(events)
        queries.extend(file_queries)

        # Git event queries
        git_queries = self._derive_git_queries(events)
        queries.extend(git_queries)

        # Terminal event queries
        terminal_queries = self._derive_terminal_queries(events)
        queries.extend(terminal_queries)

        # Deduplicate and limit to 5
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            q = q.strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                unique.append(q)

        return unique[:5]

    def _derive_file_queries(self, events: list[ActivityEvent]) -> list[str]:
        """Derive queries from file events by grouping by project and area."""
        file_events = [e for e in events if e.type in ("file_modified", "file_created")]
        if not file_events:
            return []

        # Group by project (top-level directory)
        projects: dict[str, list[ActivityEvent]] = defaultdict(list)
        for event in file_events:
            project = self._extract_project_name(event.path)
            projects[project].append(event)

        queries: list[str] = []
        for project, proj_events in projects.items():
            # Extract languages and component names
            languages: set[str] = set()
            components: set[str] = set()
            for event in proj_events:
                ext = event.metadata.get("extension", "")
                lang = EXTENSION_LANGUAGES.get(ext, "")
                if lang:
                    languages.add(lang)
                component = self._extract_component(event.path)
                if component:
                    components.add(component)

            # Build query
            parts: list[str] = []
            if languages:
                parts.append(sorted(languages)[0])
            if components:
                parts.append(" ".join(sorted(components)[:2]))
            if project:
                parts.append(project)
            parts.append("patterns")

            query = " ".join(parts)
            if len(query) > 10:
                queries.append(query)

        return queries[:2]

    def _derive_git_queries(self, events: list[ActivityEvent]) -> list[str]:
        """Derive queries from git events."""
        queries: list[str] = []

        for event in events:
            if event.type == "git_commit":
                message = event.metadata.get("message", "")
                if message and len(message) > 5:
                    queries.append(message)

            elif event.type == "git_branch":
                branch = event.metadata.get("branch", "")
                if branch and branch not in ("main", "master", "develop"):
                    # Convert branch name to query: "feature/auth-system" -> "auth system"
                    cleaned = branch.replace("/", " ").replace("-", " ").replace("_", " ")
                    if len(cleaned) > 3:
                        queries.append(cleaned)

            elif event.type == "git_diff":
                changed_files = event.metadata.get("changed_files", [])
                if changed_files:
                    # Extract meaningful names from changed file paths
                    names: set[str] = set()
                    for f in changed_files[:5]:
                        base = os.path.splitext(os.path.basename(f))[0]
                        if base and len(base) > 2:
                            names.add(base.replace("_", " ").replace("-", " "))
                    if names:
                        queries.append(" ".join(sorted(names)[:3]))

        return queries[:2]

    def _derive_terminal_queries(self, events: list[ActivityEvent]) -> list[str]:
        """Derive queries from terminal commands."""
        terminal_events = [e for e in events if e.type == "terminal_command"]
        if not terminal_events:
            return []

        queries: list[str] = []
        for event in terminal_events:
            command = event.metadata.get("command", "")
            tokens = self._extract_meaningful_tokens(command)
            if tokens:
                queries.append(" ".join(tokens))

        return queries[:1]

    def _extract_project_name(self, path: str) -> str:
        """Extract project name from a file path."""
        parts = path.replace("\\", "/").split("/")
        # Look for common project root indicators
        for i, part in enumerate(parts):
            if part in ("src", "lib", "app", "packages", "projects"):
                if i + 1 < len(parts):
                    return parts[i + 1]
            # If we find a directory that likely has a project root
            if i > 0 and part in (
                "node_modules", ".git", "__pycache__", "venv", ".venv"
            ):
                return parts[i - 1]
        # Fallback: use first meaningful directory after root
        meaningful = [p for p in parts if p and not p.startswith(".") and ":" not in p]
        if len(meaningful) >= 2:
            return meaningful[1] if meaningful[0] in ("home", "Users", "dev") else meaningful[0]
        return ""

    def _extract_component(self, path: str) -> str:
        """Extract component/module name from a file path."""
        parts = path.replace("\\", "/").split("/")
        # Look for directory names that indicate components
        skip = {"src", "lib", "app", "test", "tests", "__pycache__", "node_modules"}
        for part in reversed(parts[:-1]):  # Exclude filename
            if part and part not in skip and not part.startswith("."):
                return part.replace("_", " ").replace("-", " ")
        return ""

    def _extract_meaningful_tokens(self, command: str) -> list[str]:
        """Extract meaningful tokens from a terminal command."""
        parts = command.split()
        if not parts:
            return []

        tokens: list[str] = []
        base_cmd = os.path.basename(parts[0])
        tokens.append(base_cmd)

        for part in parts[1:]:
            # Skip flags
            if part.startswith("-"):
                continue
            # Skip common path prefixes
            if part.startswith("/") or part.startswith("./"):
                base = os.path.basename(part)
                if base and len(base) > 2:
                    tokens.append(base)
                continue
            # Keep meaningful tokens
            if len(part) > 2 and not part.startswith("{"):
                tokens.append(part)

        return tokens[:4]
