"""Webhook event system: fire HTTP POST on memory events."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class WebhookEmitter:
    """Fire webhooks on memory events. Non-blocking (uses background threads)."""

    def __init__(self, url: str = "", events: str = "store,delete,consolidate"):
        self._url = url
        self._events = set(e.strip() for e in events.split(",") if e.strip())

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Fire a webhook if the event type is enabled. Non-blocking."""
        if not self.enabled or event not in self._events:
            return

        payload = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }

        # Fire in background thread to not block the caller
        thread = threading.Thread(target=self._send, args=(payload,), daemon=True)
        thread.start()

    def _send(self, payload: dict) -> None:
        """Send the webhook payload. Best-effort, no retries."""
        try:
            import urllib.request

            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except Exception as e:
            logger.warning("Webhook delivery failed: %s", e)
