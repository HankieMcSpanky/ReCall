from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from neuropack.watcher.cache import AnticipationCache
from neuropack.watcher.events import ActivityEvent
from neuropack.watcher.fs_watcher import FileSystemWatcher
from neuropack.watcher.git_watcher import GitWatcher
from neuropack.watcher.query_deriver import QueryDeriver
from neuropack.watcher.terminal_watcher import TerminalWatcher

if TYPE_CHECKING:
    from neuropack.config import NeuropackConfig
    from neuropack.core.store import MemoryStore

logger = logging.getLogger(__name__)


class AnticipatoryDaemon:
    """Background daemon that watches developer activity and pre-loads context."""

    def __init__(self, store: MemoryStore, config: NeuropackConfig):
        self._store = store
        self._config = config
        self._event_queue: queue.Queue[ActivityEvent] = queue.Queue()
        self._cache = AnticipationCache(ttl_seconds=config.watcher_cache_ttl)
        self._query_deriver = QueryDeriver()

        # Parse watched directories
        dirs = [
            d.strip() for d in config.watcher_dirs.split(",")
            if d.strip()
        ]
        self._directories = dirs

        # Sub-watchers
        self._fs_watcher = FileSystemWatcher(dirs, self._event_queue)
        self._git_watcher = GitWatcher(
            dirs, self._event_queue, poll_interval=config.watcher_poll_interval,
        )
        self._terminal_watcher = TerminalWatcher(
            self._event_queue, history_file=config.watcher_history_file,
        )

        # Main loop
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

        # Event callback for foreground mode
        self._on_event = None
        self._on_recall = None

    @property
    def is_running(self) -> bool:
        """Check if the daemon is running."""
        return self._running

    @property
    def cache(self) -> AnticipationCache:
        """Access the anticipation cache."""
        return self._cache

    @property
    def directories(self) -> list[str]:
        """Return watched directories."""
        return self._directories

    def start(self) -> None:
        """Launch watchers and main loop in a background thread."""
        if self._running:
            logger.warning("AnticipatoryDaemon is already running")
            return

        self._stop_event.clear()

        # Start sub-watchers
        self._fs_watcher.start()
        self._git_watcher.start()
        self._terminal_watcher.start()

        # Start main processing loop
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        self._running = True
        logger.info(
            "AnticipatoryDaemon started, watching %d directories",
            len(self._directories),
        )

    def stop(self) -> None:
        """Graceful shutdown of all watchers and the main loop."""
        if not self._running:
            return

        self._stop_event.set()

        # Stop sub-watchers
        self._fs_watcher.stop()
        self._git_watcher.stop()
        self._terminal_watcher.stop()

        # Stop main loop
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

        self._running = False
        logger.info("AnticipatoryDaemon stopped")

    def _main_loop(self) -> None:
        """Consume events, debounce, derive queries, run recall, cache results."""
        debounce_seconds = self._config.watcher_debounce_seconds
        pending_events: list[ActivityEvent] = []
        last_process_time = 0.0

        while not self._stop_event.is_set():
            # Drain event queue
            try:
                while True:
                    event = self._event_queue.get_nowait()
                    pending_events.append(event)
                    # Notify foreground callback
                    if self._on_event is not None:
                        try:
                            self._on_event(event)
                        except Exception:
                            pass
            except queue.Empty:
                pass

            # Debounce: process events after quiet period
            now = time.monotonic()
            if pending_events and (now - last_process_time) >= debounce_seconds:
                self._process_events(pending_events)
                pending_events = []
                last_process_time = now

            self._stop_event.wait(timeout=1.0)

        # Process remaining events on shutdown
        if pending_events:
            self._process_events(pending_events)

    def _process_events(self, events: list[ActivityEvent]) -> None:
        """Derive queries from events, run recall, cache results."""
        queries = self._query_deriver.derive_queries(events)
        if not queries:
            return

        logger.debug("Derived %d queries from %d events", len(queries), len(events))

        for query_text in queries:
            try:
                results = self._store.recall(query=query_text, limit=5, min_score=0.1)
                if results:
                    result_dicts = [
                        {
                            "id": r.record.id,
                            "l3_abstract": r.record.l3_abstract,
                            "content_preview": r.record.content[:200],
                            "tags": r.record.tags,
                            "score": round(r.score, 4),
                            "namespace": r.record.namespace,
                            "source": "anticipatory",
                            "query": query_text,
                        }
                        for r in results
                    ]
                    self._cache.put(
                        query_text, result_dicts, datetime.now(timezone.utc),
                    )
                    logger.debug(
                        "Cached %d results for query: %s", len(result_dicts), query_text,
                    )
                    # Notify foreground callback
                    if self._on_recall is not None:
                        try:
                            self._on_recall(query_text, result_dicts)
                        except Exception:
                            pass
            except Exception as e:
                logger.debug("Recall failed for query '%s': %s", query_text, e)
