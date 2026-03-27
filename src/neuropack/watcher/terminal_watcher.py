from __future__ import annotations

import logging
import os
import platform
import threading
from datetime import datetime, timezone
from queue import Queue

from neuropack.watcher.events import ActivityEvent

logger = logging.getLogger(__name__)

# Commands to ignore (noise)
NOISE_COMMANDS = {
    "cd", "ls", "dir", "clear", "cls", "exit", "quit", "pwd", "echo",
    "cat", "less", "more", "man", "help", "history", "which", "where",
    "whoami", "date", "time", "true", "false", ":", "test",
}


class TerminalWatcher:
    """Watch shell history file for new commands."""

    def __init__(self, event_queue: Queue, history_file: str = ""):
        self._queue = event_queue
        self._history_file = history_file or self._detect_history_file()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start tail-following the history file."""
        if not self._history_file:
            logger.warning("No shell history file found; terminal watching disabled")
            return

        if not os.path.isfile(self._history_file):
            logger.warning("History file not found: %s", self._history_file)
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._tail_loop, daemon=True)
        self._thread.start()
        logger.info("TerminalWatcher started, following: %s", self._history_file)

    def stop(self) -> None:
        """Stop the terminal watcher."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("TerminalWatcher stopped")

    def _detect_history_file(self) -> str:
        """Auto-detect the shell history file."""
        home = os.path.expanduser("~")

        if platform.system() == "Windows":
            # PowerShell history
            ps_history = os.path.join(
                home, "AppData", "Roaming", "Microsoft", "Windows",
                "PowerShell", "PSReadLine", "ConsoleHost_history.txt",
            )
            if os.path.isfile(ps_history):
                return ps_history

        # zsh history (common on macOS and many Linux)
        zsh_hist = os.path.join(home, ".zsh_history")
        if os.path.isfile(zsh_hist):
            return zsh_hist

        # bash history
        bash_hist = os.path.join(home, ".bash_history")
        if os.path.isfile(bash_hist):
            return bash_hist

        return ""

    def _tail_loop(self) -> None:
        """Tail-follow the history file using seek/tell polling."""
        try:
            with open(self._history_file, "r", encoding="utf-8", errors="replace") as f:
                # Seek to end of file
                f.seek(0, 2)

                while not self._stop_event.is_set():
                    line = f.readline()
                    if line:
                        line = line.strip()
                        if line:
                            self._process_line(line)
                    else:
                        self._stop_event.wait(timeout=2)
        except Exception as e:
            logger.warning("TerminalWatcher error: %s", e)

    def _process_line(self, line: str) -> None:
        """Process a history line, filter noise, emit event."""
        # Handle zsh extended history format: ": timestamp:0;command"
        command = line
        if command.startswith(": ") and ";" in command:
            command = command.split(";", 1)[1]

        # Get the base command (first word)
        parts = command.split()
        if not parts:
            return

        base_cmd = parts[0].strip()
        # Strip path prefixes
        if "/" in base_cmd or "\\" in base_cmd:
            base_cmd = os.path.basename(base_cmd)

        if base_cmd.lower() in NOISE_COMMANDS:
            return

        event = ActivityEvent(
            type="terminal_command",
            path="",
            timestamp=datetime.now(timezone.utc),
            metadata={"command": command},
        )
        self._queue.put(event)
        logger.debug("Terminal command: %s", command[:100])
