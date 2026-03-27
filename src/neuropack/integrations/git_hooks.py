"""Git hooks integration for NeuroPack.

Automatically captures commit messages, diffs, and code review context
into NeuroPack memory via post-commit, post-merge, and post-checkout hooks.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

NEUROPACK_HOOK_MARKER = "# --- NeuroPack git-hooks integration ---"
NEUROPACK_HOOK_END = "# --- End NeuroPack ---"

DEFAULT_HOOKS = ["post-commit", "post-merge", "post-checkout"]

_HOOK_TEMPLATE = """\
{marker}
# NeuroPack: auto-capture git context into memory.
# Safe wrapper: if np fails, the git operation still succeeds.
np git capture {hook_type} 2>/dev/null || true
{end_marker}
"""


def _git_hooks_dir(repo_path: str) -> Path:
    """Return the .git/hooks directory for the given repo."""
    repo = Path(repo_path).resolve()
    git_dir = repo / ".git"
    if not git_dir.is_dir():
        raise FileNotFoundError(f"Not a git repository: {repo} (no .git directory)")
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    return hooks_dir


def _neuropack_snippet(hook_type: str) -> str:
    """Return the NeuroPack snippet for a given hook type."""
    return _HOOK_TEMPLATE.format(
        marker=NEUROPACK_HOOK_MARKER,
        end_marker=NEUROPACK_HOOK_END,
        hook_type=hook_type,
    )


def _make_executable(path: Path) -> None:
    """Ensure a file has executable permissions."""
    current = path.stat().st_mode
    path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_hooks(repo_path: str, hooks: list[str] | None = None) -> list[str]:
    """Install NeuroPack git hooks into the given repo.

    Args:
        repo_path: Path to the git repository root.
        hooks: Which hooks to install. Defaults to post-commit, post-merge,
               post-checkout.

    Returns:
        List of installed hook file paths.
    """
    hooks_to_install = hooks or DEFAULT_HOOKS
    hooks_dir = _git_hooks_dir(repo_path)
    installed: list[str] = []

    for hook_type in hooks_to_install:
        hook_path = hooks_dir / hook_type
        snippet = _neuropack_snippet(hook_type)

        if hook_path.exists():
            existing = hook_path.read_text(encoding="utf-8")
            # Already installed — skip
            if NEUROPACK_HOOK_MARKER in existing:
                installed.append(str(hook_path))
                continue
            # Append to existing hook
            content = existing.rstrip("\n") + "\n\n" + snippet
        else:
            # Create new hook script
            content = "#!/bin/sh\n\n" + snippet

        hook_path.write_text(content, encoding="utf-8")
        _make_executable(hook_path)
        installed.append(str(hook_path))

    return installed


def uninstall_hooks(repo_path: str) -> list[str]:
    """Remove NeuroPack lines from all hooks in the given repo.

    Returns:
        List of hook paths that were modified.
    """
    hooks_dir = _git_hooks_dir(repo_path)
    modified: list[str] = []

    for hook_path in hooks_dir.iterdir():
        if not hook_path.is_file():
            continue
        content = hook_path.read_text(encoding="utf-8")
        if NEUROPACK_HOOK_MARKER not in content:
            continue

        # Remove the NeuroPack block
        lines = content.split("\n")
        new_lines: list[str] = []
        inside_block = False
        for line in lines:
            if line.strip() == NEUROPACK_HOOK_MARKER:
                inside_block = True
                continue
            if inside_block and line.strip() == NEUROPACK_HOOK_END:
                inside_block = False
                continue
            if inside_block:
                continue
            new_lines.append(line)

        # Clean up trailing blank lines
        cleaned = "\n".join(new_lines).rstrip("\n") + "\n"

        # If only the shebang remains, remove the file entirely
        stripped = cleaned.strip()
        if stripped in ("#!/bin/sh", "#!/bin/bash", ""):
            hook_path.unlink()
        else:
            hook_path.write_text(cleaned, encoding="utf-8")

        modified.append(str(hook_path))

    return modified


def get_installed_hooks(repo_path: str) -> list[str]:
    """Return list of hook types that have NeuroPack installed."""
    try:
        hooks_dir = _git_hooks_dir(repo_path)
    except FileNotFoundError:
        return []

    installed: list[str] = []
    for hook_path in hooks_dir.iterdir():
        if not hook_path.is_file():
            continue
        content = hook_path.read_text(encoding="utf-8")
        if NEUROPACK_HOOK_MARKER in content:
            installed.append(hook_path.name)
    return sorted(installed)


def _run_git(repo_path: str, *args: str) -> str:
    """Run a git command and return stdout. Returns empty string on failure."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _repo_name(repo_path: str) -> str:
    """Extract repository name from path."""
    return Path(repo_path).resolve().name


def _get_or_create_store(store: object | None = None):
    """Get the provided store or create a fresh one."""
    if store is not None:
        return store
    from neuropack.config import NeuropackConfig
    from neuropack.core.store import MemoryStore

    s = MemoryStore(NeuropackConfig())
    s.initialize()
    return s


def capture_post_commit(repo_path: str, store: object | None = None) -> None:
    """Capture a post-commit event into NeuroPack memory.

    Collects commit message, diff summary, changed files, author, and date,
    then stores a formatted memory.
    """
    short_hash = _run_git(repo_path, "log", "-1", "--format=%h")
    full_message = _run_git(repo_path, "log", "-1", "--format=%B")
    author = _run_git(repo_path, "log", "-1", "--format=%an")
    date = _run_git(repo_path, "log", "-1", "--format=%Y-%m-%d")
    diff_stat = _run_git(repo_path, "diff", "HEAD~1", "--stat")
    changed_files = _run_git(repo_path, "diff", "HEAD~1", "--name-only")

    if not short_hash:
        return

    # Build formatted memory content
    lines = [
        f"Git Commit: {short_hash}",
        f"Author: {author}",
        f"Date: {date}",
        "",
        f"Message: {full_message}",
    ]

    if diff_stat:
        stat_lines = diff_stat.strip().split("\n")
        # The last line is the summary (e.g. "2 files changed, ...")
        file_lines = stat_lines[:-1] if len(stat_lines) > 1 else []
        summary_line = stat_lines[-1] if stat_lines else ""

        if file_lines:
            lines.append("")
            lines.append("Files changed:")
            for fl in file_lines:
                lines.append(f"  {fl.strip()}")

        if summary_line:
            lines.append("")
            lines.append(f"Summary: {summary_line.strip()}")

    content = "\n".join(lines)

    # Build tags
    repo = _repo_name(repo_path)
    tags = ["git-commit", f"repo-{repo}"]
    if changed_files:
        for f in changed_files.split("\n"):
            f = f.strip()
            if f:
                # Add directory-level tags for the most relevant paths
                parts = Path(f).parts
                if len(parts) > 1:
                    tags.append(f"dir-{parts[0]}")

    # Deduplicate tags
    tags = list(dict.fromkeys(tags))

    source = f"git:commit:{short_hash}"

    ms = _get_or_create_store(store)
    try:
        ms.store(content=content, tags=tags, source=source, priority=0.4)
    finally:
        if store is None:
            ms.close()


def capture_post_merge(repo_path: str, store: object | None = None) -> None:
    """Capture a post-merge event into NeuroPack memory.

    Collects merge commit info and the branch that was merged.
    """
    short_hash = _run_git(repo_path, "log", "-1", "--format=%h")
    date = _run_git(repo_path, "log", "-1", "--format=%Y-%m-%d")
    subject = _run_git(repo_path, "log", "-1", "--format=%s")
    current_branch = _run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")

    if not short_hash:
        return

    # Try to extract merged branch from the merge commit subject
    # Typical: "Merge branch 'feature/foo' into main"
    merged_branch = ""
    if subject.startswith("Merge branch '"):
        parts = subject.split("'")
        if len(parts) >= 2:
            merged_branch = parts[1]
    elif subject.startswith("Merge pull request"):
        merged_branch = subject

    # Get diff stats for the merge
    diff_stat_summary = _run_git(repo_path, "diff", "HEAD~1", "--stat")
    stat_summary = ""
    if diff_stat_summary:
        stat_lines = diff_stat_summary.strip().split("\n")
        stat_summary = stat_lines[-1].strip() if stat_lines else ""

    # Count merge commits
    merge_description = f"{merged_branch} -> {current_branch}" if merged_branch else subject

    lines = [
        f"Git Merge: {merge_description}",
        f"Commit: {short_hash}",
        f"Date: {date}",
    ]

    if stat_summary:
        lines.append("")
        lines.append(stat_summary)

    content = "\n".join(lines)

    repo = _repo_name(repo_path)
    tags = ["git-merge", f"repo-{repo}"]
    if merged_branch:
        safe_branch = merged_branch.replace("/", "-")
        tags.append(f"branch-{safe_branch}")

    source = f"git:merge:{short_hash}"

    ms = _get_or_create_store(store)
    try:
        ms.store(content=content, tags=tags, source=source, priority=0.5)
    finally:
        if store is None:
            ms.close()


def capture_post_checkout(repo_path: str, store: object | None = None) -> None:
    """Capture a post-checkout (branch switch) event into NeuroPack memory.

    Stores a lightweight context-switch memory.
    """
    branch = _run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    if not branch:
        return

    date = _run_git(repo_path, "log", "-1", "--format=%Y-%m-%d %H:%M")

    content = f"Switched to branch: {branch}\nDate: {date}"

    repo = _repo_name(repo_path)
    safe_branch = branch.replace("/", "-")
    tags = ["git-checkout", f"repo-{repo}", f"branch-{safe_branch}"]

    source = f"git:checkout:{branch}"

    ms = _get_or_create_store(store)
    try:
        ms.store(content=content, tags=tags, source=source, priority=0.2)
    finally:
        if store is None:
            ms.close()
