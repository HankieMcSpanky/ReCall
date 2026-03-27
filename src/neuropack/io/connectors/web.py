"""Web page connector: extract clean text from URLs or HTML files."""
from __future__ import annotations

from pathlib import Path


def parse_url(url: str) -> list[dict]:
    """Fetch a URL and extract main content as a memory.

    Requires trafilatura: pip install trafilatura

    Returns list of dicts with keys: content, tags, source, priority.
    """
    try:
        import trafilatura
    except ImportError:
        raise ImportError(
            "Web parsing requires trafilatura: pip install trafilatura"
        )

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return []

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text or len(text.strip()) < 10:
        return []

    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else url
    tags = ["web", "imported"]
    if metadata and metadata.date:
        tags.append(f"date:{metadata.date}")

    return [{
        "content": text.strip(),
        "tags": tags,
        "source": f"web:{title[:100]}",
        "priority": 0.5,
    }]


def parse_html(html_string: str, source: str = "html") -> list[dict]:
    """Extract main content from an HTML string.

    Requires trafilatura: pip install trafilatura
    """
    try:
        import trafilatura
    except ImportError:
        raise ImportError(
            "HTML parsing requires trafilatura: pip install trafilatura"
        )

    text = trafilatura.extract(html_string, include_comments=False, include_tables=True)
    if not text or len(text.strip()) < 10:
        return []

    return [{
        "content": text.strip(),
        "tags": ["html", "imported"],
        "source": source,
        "priority": 0.5,
    }]


def parse_html_file(path: str) -> list[dict]:
    """Extract content from a local HTML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HTML file not found: {path}")

    html = p.read_text(encoding="utf-8")
    return parse_html(html, source=f"html:{p.stem}")
