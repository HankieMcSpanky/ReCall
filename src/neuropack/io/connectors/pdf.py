"""PDF connector: extract text from PDF files into memory-ready dicts."""
from __future__ import annotations

from pathlib import Path


def parse_pdf(path: str) -> list[dict]:
    """Extract text from a PDF file, one memory per page.

    Requires pymupdf (PyMuPDF): pip install pymupdf
    Falls back to pdfplumber if pymupdf unavailable.

    Returns list of dicts with keys: content, tags, source, priority.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = _extract_pages(str(p))
    if not pages:
        return []

    filename = p.stem
    memories: list[dict] = []

    for page_num, text in pages:
        text = text.strip()
        if len(text) < 10:
            continue

        memories.append({
            "content": text,
            "tags": ["pdf", "imported", f"page:{page_num}"],
            "source": f"pdf:{filename}",
            "priority": 0.5,
        })

    return memories


def _extract_pages(path: str) -> list[tuple[int, str]]:
    """Try pymupdf first, then pdfplumber."""
    # Try PyMuPDF
    try:
        import pymupdf
        doc = pymupdf.open(path)
        pages = []
        for i, page in enumerate(doc, 1):
            pages.append((i, page.get_text()))
        doc.close()
        return pages
    except ImportError:
        pass

    # Try pdfplumber
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                pages.append((i, text))
        return pages
    except ImportError:
        pass

    raise ImportError(
        "PDF parsing requires pymupdf or pdfplumber: "
        "pip install pymupdf  OR  pip install pdfplumber"
    )
