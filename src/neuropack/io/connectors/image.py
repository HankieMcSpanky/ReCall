"""Image connector: extract descriptions from screenshots and images.

Stores images in ~/.neuropack/images/ and creates text memories
from their content (OCR + optional vision model description).

Supports: PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF

Usage:
    from neuropack.io.connectors.image import process_image, process_image_folder

    # Single image → memory dict
    memory = process_image("/path/to/screenshot.png")
    store.store(**memory)

    # Folder of images → list of memory dicts
    memories = process_image_folder("/path/to/screenshots/")

    # With vision model description (requires LLM)
    memory = process_image("/path/to/screenshot.png", llm_provider=ollama)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
_IMAGE_STORE_DIR = os.path.expanduser("~/.neuropack/images")


def process_image(
    path: str,
    llm_provider: Any = None,
    store_copy: bool = True,
) -> dict:
    """Process a single image into a memory-ready dict.

    1. Copies image to ~/.neuropack/images/ (with timestamp)
    2. Extracts text via OCR (if available)
    3. Generates description via vision model (if llm_provider given)
    4. Returns a dict ready for store.store()

    Args:
        path: Path to image file
        llm_provider: Optional LLM with vision capability for description
        store_copy: If True, copy image to ~/.neuropack/images/
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if p.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Not a supported image format: {p.suffix}")

    # Copy to storage
    stored_path = ""
    if store_copy:
        stored_path = _store_image(p)

    # Extract text content
    parts: list[str] = []

    # Try OCR
    ocr_text = _extract_ocr(str(p))
    if ocr_text:
        parts.append(f"Text in image:\n{ocr_text}")

    # Try vision model description
    if llm_provider:
        description = _describe_with_vision(str(p), llm_provider)
        if description:
            parts.append(f"Description:\n{description}")

    # Fallback: just record the filename and metadata
    if not parts:
        parts.append(f"Screenshot: {p.name}")
        size = p.stat().st_size
        parts.append(f"File size: {size // 1024}KB")

    content = "\n".join(parts)
    if stored_path:
        content += f"\n\nStored at: {stored_path}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return {
        "content": content,
        "tags": ["image", "screenshot", p.suffix.lstrip(".").lower()],
        "source": f"image:{p.name}",
        "priority": 0.4,
        "l3_override": f"Screenshot {p.stem} ({timestamp})",
    }


def process_image_folder(
    folder: str,
    llm_provider: Any = None,
    store_copies: bool = True,
) -> list[dict]:
    """Process all images in a folder. Returns list of memory dicts."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    memories = []
    for f in sorted(folder_path.iterdir()):
        if f.suffix.lower() in _IMAGE_EXTENSIONS:
            try:
                mem = process_image(str(f), llm_provider=llm_provider, store_copy=store_copies)
                memories.append(mem)
            except Exception as e:
                logger.debug("Failed to process %s: %s", f.name, e)

    return memories


def _store_image(source: Path) -> str:
    """Copy image to ~/.neuropack/images/ with timestamp prefix."""
    dest_dir = Path(_IMAGE_STORE_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_name = f"{timestamp}_{source.name}"
    dest_path = dest_dir / dest_name

    shutil.copy2(str(source), str(dest_path))
    return str(dest_path)


def _extract_ocr(image_path: str) -> str:
    """Extract text from image using OCR. Returns empty string if unavailable."""
    # Try pytesseract
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip() if text and text.strip() else ""
    except ImportError:
        pass
    except Exception as e:
        logger.debug("OCR failed: %s", e)

    # Try easyocr
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(image_path)
        text = " ".join(r[1] for r in results)
        return text.strip() if text else ""
    except ImportError:
        pass
    except Exception as e:
        logger.debug("EasyOCR failed: %s", e)

    return ""


def _describe_with_vision(image_path: str, llm_provider: Any) -> str:
    """Use a vision-capable LLM to describe the image content."""
    import base64

    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine MIME type
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        }
        mime = mime_map.get(ext, "image/png")

        # Try OpenAI-compatible vision API
        if hasattr(llm_provider, "chat"):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Describe this screenshot/image concisely. Include:\n"
                        "- What the image shows (UI, code, error, diagram, etc.)\n"
                        "- Any visible text, error messages, or code\n"
                        "- Key details that would help someone recall this later"
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{image_data}"
                    }},
                ],
            }]
            response = llm_provider.chat(messages, max_tokens=300)
            return response.strip() if response else ""

    except Exception as e:
        logger.debug("Vision description failed: %s", e)

    return ""
