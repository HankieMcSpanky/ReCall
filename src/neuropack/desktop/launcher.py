"""Desktop app wrapper: pywebview window + background server + system tray."""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)


class NeuropackDesktop:
    """Orchestrates server, webview window, and system tray."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7341, db_path: str | None = None):
        self.host = host
        self.port = port
        self.db_path = db_path
        self._shutdown = threading.Event()

    def _start_server(self) -> None:
        import uvicorn

        from neuropack.api.app import create_app
        from neuropack.config import NeuropackConfig

        kwargs: dict = {"api_port": self.port, "api_host": self.host}
        if self.db_path:
            kwargs["db_path"] = self.db_path
        config = NeuropackConfig(**kwargs)
        app = create_app(config)

        uvi_config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(uvi_config)
        server.run()

    def _wait_for_server(self, timeout: float = 15.0) -> bool:
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + timeout
        url = f"http://{self.host}:{self.port}/health"
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(url, timeout=2)
                return True
            except (urllib.error.URLError, OSError):
                time.sleep(0.3)
        return False

    def _create_icon_image(self):
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([2, 2, 62, 62], fill=(108, 99, 255, 255))
        draw.text((14, 20), "NP", fill=(255, 255, 255, 255))
        return img

    def _start_tray(self) -> None:
        import webbrowser

        from pystray import Icon, Menu, MenuItem

        def open_dashboard(icon, item):
            webbrowser.open(f"http://{self.host}:{self.port}/")

        def quit_app(icon, item):
            icon.stop()
            self._shutdown.set()
            os._exit(0)

        menu = Menu(
            MenuItem("Open Dashboard", open_dashboard, default=True),
            Menu.SEPARATOR,
            MenuItem("Quit NeuroPack", quit_app),
        )

        icon = Icon("NeuroPack", self._create_icon_image(), "NeuroPack Memory Store", menu)
        icon.run()

    def run(self) -> None:
        """Start server, open window, run tray."""
        # Start server
        server_thread = threading.Thread(target=self._start_server, daemon=True)
        server_thread.start()

        if not self._wait_for_server():
            print("Error: NeuroPack server failed to start.", file=sys.stderr)
            sys.exit(1)

        print(f"NeuroPack running at http://{self.host}:{self.port}/")

        # Start tray in background
        tray_thread = threading.Thread(target=self._start_tray, daemon=True)
        tray_thread.start()

        # Open window (blocks until closed)
        try:
            import webview
            webview.create_window(
                "NeuroPack",
                f"http://{self.host}:{self.port}/",
                width=1100,
                height=750,
                min_size=(700, 500),
            )
            webview.start()
        except ImportError:
            import webbrowser
            webbrowser.open(f"http://{self.host}:{self.port}/")
            print("pywebview not installed. Opened in browser instead.")
            print("Press Ctrl+C to quit, or use the system tray icon.")

        # Window closed — wait for tray quit
        self._shutdown.wait()
