"""Load PDF and image files into page images for vision LLM extraction."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

# 200 DPI keeps small digits readable on dense tables and scans without huge payloads.
DEFAULT_PDF_DPI = 200


class InvalidDocumentError(ValueError):
    """Raised when an input file is unsupported, empty, or corrupt/unreadable."""


@dataclass(frozen=True)
class PageImage:
    """In-memory image bytes ready to send to the Gemini API."""

    data: bytes
    mime_type: str
    page_number: int


def load_document_images(
    file_path: str | Path,
    dpi: int = DEFAULT_PDF_DPI,
) -> list[PageImage]:
    """Convert a PDF's pages to PNG images, or pass a JPG/PNG image through."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise InvalidDocumentError(f"Path is not a file: {path}")
    if path.stat().st_size == 0:
        raise InvalidDocumentError(f"File is empty: {path.name}")

    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return _load_pdf_pages(path, dpi=dpi)
    if suffix in IMAGE_EXTENSIONS:
        return [_load_image_file(path)]

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise InvalidDocumentError(
        f"Unsupported file type '{suffix or '(none)'}'. Supported types: {supported}"
    )


def _load_pdf_pages(path: Path, dpi: int) -> list[PageImage]:
    """Render every page of a PDF to PNG bytes in printed page order."""
    pages: list[PageImage] = []
    # Read bytes first so PyMuPDF never holds an OS file lock on Windows if opening fails.
    pdf_bytes = path.read_bytes()
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise InvalidDocumentError("PDF has no pages.")
            for idx, page in enumerate(doc, start=1):
                pix = page.get_pixmap(dpi=dpi)
                pages.append(
                    PageImage(
                        data=pix.tobytes("png"),
                        mime_type="image/png",
                        page_number=idx,
                    )
                )
    except InvalidDocumentError:
        raise
    except Exception as exc:
        raise InvalidDocumentError("Could not read file as a valid PDF.") from exc
    return pages


def _load_image_file(path: Path) -> PageImage:
    """Read a JPG/PNG image, applying EXIF orientation so phone photos are upright."""
    image_bytes = path.read_bytes()
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            transposed = ImageOps.exif_transpose(img)
            # Convert unusual color modes (e.g. CMYK, RGBA for JPEG) cleanly.
            suffix = path.suffix.lower()
            buf = io.BytesIO()
            if suffix in {".jpg", ".jpeg"}:
                if transposed.mode != "RGB":
                    transposed = transposed.convert("RGB")
                transposed.save(buf, format="JPEG", quality=95)
                mime_type = "image/jpeg"
            else:
                transposed.save(buf, format="PNG")
                mime_type = "image/png"
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidDocumentError("Could not read file as a valid image.") from exc

    return PageImage(
        data=buf.getvalue(),
        mime_type=mime_type,
        page_number=1,
    )
