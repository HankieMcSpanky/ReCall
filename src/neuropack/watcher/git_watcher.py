from __future__ import annotations

import logging
import os
import subprocess
import threading
from datetime import datetime, timezone
from queue import Queue

from neuropack.watcher.events import ActivityEvent

logger = logging.getLogger(__name__)


class GitWatcher:
    """Poll git repositories for commits, branch switches, and diffs."""

    def __init__(
        self,
        directories: list[str],
        event_queue: Queue,
        poll_interval: int = 10,
    ):
        self._directories = directories
        self._queue = event_queue
        self._poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Track state per repo
        self._last_commit: dict[str, str] = {}
        self._last_branch: dict[str, str] = {}
        self._last_diff_stat: dict[str, str] = {}

    def start(self) -> None:
        """Start the git polling loop in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("GitWatcher started, polling every %ds", self._poll_interval)

    def stop(self) -> None:
        """Stop the git watcher."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        logger.info("GitWatcher stopped")

    def _poll_loop(self) -> None:
        """Main polling loop."""
        # Discover git repos on first run
        repos = self._find_git_repos()
        if not repos:
            logger.info("No git repositories found in watched directories")
            return

        logger.info("Tracking %d git repositories", len(repos))

        while not self._stop_event.is_set():
            for repo_path in repos:
                try:
                    self._poll_repo(repo_path)
                except Exception as e:
                    logger.debug("Error polling git repo %s: %s", repo_path, e)
            self._stop_event.wait(timeout=self._poll_interval)

    def _find_git_repos(self) -> list[str]:
        """Find git repositories in watched directories."""
        repos: list[str] = []
        for directory in self._directories:
            # Check if directory itself is a git repo
            if os.path.isdir(os.path.join(directory, ".git")):
                repos.append(directory)
                continue
            # Check immediate subdirectories
            try:
                for entry in os.scandir(directory):
                    if entry.is_dir() and os.path.isdir(os.path.join(entry.path, ".git")):
                        repos.append(entry.path)
            except OSError:
                pass
        return repos

    def _poll_repo(self, repo_path: str) -> None:
        """Poll a single git repository for changes."""
        # Check for new commits
        commit_hash = self._git_cmd(repo_path, ["git", "log", "-1", "--format=%H"])
        if commit_hash and commit_hash != self._last_commit.get(repo_path):
            if repo_path in self._last_commit:
                # New commit detected
                commit_msg = self._git_cmd(
                    repo_path, ["git", "log", "-1", "--format=%s"]
                ) or ""
                commit_author = self._git_cmd(
                    repo_path, ["git", "log", "-1", "--format=%an"]
                ) or ""
                event = ActivityEvent(
                    type="git_commit",
                    path=repo_path,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "commit_hash": commit_hash,
                        "message": commit_msg,
                        "author": commit_author,
                    },
                )
                self._queue.put(event)
                logger.debug("Git commit: %s in %s", commit_hash[:8], repo_path)
            self._last_commit[repo_path] = commit_hash

        # Check for branch switches
        branch = self._git_cmd(repo_path, ["git", "branch", "--show-current"])
        if branch and branch != self._last_branch.get(repo_path):
            if repo_path in self._last_branch:
                event = ActivityEvent(
                    type="git_branch",
                    path=repo_path,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "branch": branch,
                        "previous_branch": self._last_branch[repo_path],
                    },
                )
                self._queue.put(event)
                logger.debug("Git branch switch: %s in %s", branch, repo_path)
            self._last_branch[repo_path] = branch

        # Check for uncommitted changes
        diff_stat = self._git_cmd(repo_path, ["git", "diff", "--stat"])
        if diff_stat and diff_stat != self._last_diff_stat.get(repo_path):
            if repo_path in self._last_diff_stat:
                # Extract changed file paths from diff stat
                changed_files = []
                for line in diff_stat.strip().split("\n"):
                    line = line.strip()
                    if "|" in line:
                        file_part = line.split("|")[0].strip()
                        if file_part:
                            changed_files.append(file_part)
                if changed_files:
                    event = ActivityEvent(
                        type="git_diff",
                        path=repo_path,
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            "changed_files": changed_files,
                            "diff_stat": diff_stat[:500],
                        },
                    )
                    self._queue.put(event)
                    logger.debug("Git diff: %d files changed in %s", len(changed_files), repo_path)
            self._last_diff_stat[repo_path] = diff_stat

    def _git_cmd(self, repo_path: str, cmd: list[str]) -> str | None:
        """Run a git command and return stdout, or None on failure."""
        try:
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            pass
        return None
