from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from queue import Queue

from neuropack.watcher.events import ActivityEvent

logger = logging.getLogger(__name__)

# Code file extensions to watch
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".jsx", ".tsx", ".vue", ".rb", ".md", ".yaml", ".json", ".toml",
}


class FileSystemWatcher:
    """Watch directories for file changes using watchdog (optional dependency)."""

    def __init__(
        self,
        directories: list[str],
        event_queue: Queue,
        extensions: set[str] | None = None,
    ):
        self._directories = directories
        self._queue = event_queue
        self._extensions = extensions or CODE_EXTENSIONS
        self._observer = None
        self._running = False
        self._last_event_times: dict[str, float] = {}
        self._debounce_lock = threading.Lock()

    def start(self) -> None:
        """Start watching directories for file changes."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
        except ImportError:
            logger.warning(
                "watchdog not installed; file system watching disabled. "
                "Install with: pip install neuropack[watcher]"
            )
            return

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    watcher._handle_event(event.src_path, "file_created")

            def on_modified(self, event):
                if not event.is_directory:
                    watcher._handle_event(event.src_path, "file_modified")

        self._observer = Observer()
        handler = _Handler()

        for directory in self._directories:
            try:
                self._observer.schedule(handler, directory, recursive=True)
                logger.info("Watching directory: %s", directory)
            except Exception as e:
                logger.warning("Could not watch directory %s: %s", directory, e)

        self._observer.daemon = True
        self._observer.start()
        self._running = True
        logger.info("FileSystemWatcher started for %d directories", len(self._directories))

    def stop(self) -> None:
        """Stop the file system watcher."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        self._running = False
        logger.info("FileSystemWatcher stopped")

    def _handle_event(self, path: str, event_type: str) -> None:
        """Filter and debounce file events."""
        import os

        # Check extension
        _, ext = os.path.splitext(path)
        if ext.lower() not in self._extensions:
            return

        # Debounce: ignore if same path within 1 second
        now = time.monotonic()
        with self._debounce_lock:
            last = self._last_event_times.get(path, 0.0)
            if now - last < 1.0:
                return
            self._last_event_times[path] = now

        event = ActivityEvent(
            type=event_type,
            path=path,
            timestamp=datetime.now(timezone.utc),
            metadata={"extension": ext.lower()},
        )
        self._queue.put(event)
        logger.debug("File event: %s %s", event_type, path)
