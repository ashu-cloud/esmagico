"""Generate reproducible degraded copies of the 10 labelled documents.

Creates five degradation variants from a fixed random seed in a git-ignored
folder (`degraded_samples/` by default):
  - blur             : Gaussian blur (radius 2.5)
  - rotation         : Small random rotation (+/- 4 to 8 degrees, white fill)
  - jpeg_compression : Heavy JPEG compression (quality 10)
  - low_resolution   : Downscaled to 35% width/height (~70 DPI equivalent)
  - partial_crop     : Trims 6% to 10% off 1-2 seeded edges

Usage:
    python generate_degraded.py [--seed 42] [--output-dir degraded_samples]
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageFilter

from invoice_extractor.config import REPO_ROOT, get_labelled_dir
from invoice_extractor.loader import (
    DEFAULT_PDF_DPI,
    SUPPORTED_EXTENSIONS,
    load_document_images,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "degraded_samples"
DEFAULT_SEED = 42

DEGRADATION_TYPES = (
    "blur",
    "rotation",
    "jpeg_compression",
    "low_resolution",
    "partial_crop",
)


def _apply_blur(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    radius = 2.5
    return img.filter(ImageFilter.GaussianBlur(radius=radius)), {"radius": radius}


def _apply_rotation(
    img: Image.Image, rng: random.Random
) -> tuple[Image.Image, dict[str, Any]]:
    magnitude = round(rng.uniform(4.0, 8.0), 2)
    angle = magnitude if rng.random() < 0.5 else -magnitude
    rotated = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )
    return rotated, {"angle_degrees": angle}


def _apply_jpeg_compression(
    img: Image.Image, rng: random.Random
) -> tuple[Image.Image, dict[str, Any]]:
    quality = 10
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling="4:2:0")
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")
    compressed.load()
    return compressed, {"jpeg_quality": quality}


def _apply_low_resolution(
    img: Image.Image, rng: random.Random
) -> tuple[Image.Image, dict[str, Any]]:
    scale = 0.35
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    downscaled = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return downscaled, {
        "scale": scale,
        "original_size": [img.width, img.height],
        "new_size": [new_w, new_h],
    }


def _apply_partial_crop(
    img: Image.Image, rng: random.Random
) -> tuple[Image.Image, dict[str, Any]]:
    """Trim 6%-10% off two seeded edges (one horizontal, one vertical)."""
    w, h = img.width, img.height
    h_edge = rng.choice(["left", "right"])
    v_edge = rng.choice(["top", "bottom"])
    h_frac = round(rng.uniform(0.06, 0.10), 3)
    v_frac = round(rng.uniform(0.06, 0.10), 3)

    left = int(round(w * h_frac)) if h_edge == "left" else 0
    right = int(round(w * (1.0 - h_frac))) if h_edge == "right" else w
    top = int(round(h * v_frac)) if v_edge == "top" else 0
    bottom = int(round(h * (1.0 - v_frac))) if v_edge == "bottom" else h

    cropped = img.crop((left, top, right, bottom))
    return cropped, {
        "horizontal_edge": h_edge,
        "horizontal_frac": h_frac,
        "vertical_edge": v_edge,
        "vertical_frac": v_frac,
        "box": [left, top, right, bottom],
    }


_TRANSFORMS = {
    "blur": _apply_blur,
    "rotation": _apply_rotation,
    "jpeg_compression": _apply_jpeg_compression,
    "low_resolution": _apply_low_resolution,
    "partial_crop": _apply_partial_crop,
}


def _save_degraded_document(
    images: list[Image.Image],
    out_dir: Path,
    doc_id: str,
    degradation: str,
) -> Path:
    """Save 1-page degraded images as JPG/PNG, or multi-page images as a PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    use_jpeg = degradation == "jpeg_compression"

    if len(images) == 1:
        if use_jpeg:
            out_path = out_dir / f"{doc_id}.jpg"
            images[0].convert("RGB").save(out_path, format="JPEG", quality=10, subsampling="4:2:0")
        else:
            out_path = out_dir / f"{doc_id}.png"
            images[0].convert("RGB").save(out_path, format="PNG")
        return out_path

    # Multi-page document (e.g., L04.pdf): pack degraded page images into a PDF
    # so extract_invoice() processes all pages together through the standard loader.
    out_path = out_dir / f"{doc_id}.pdf"
    pdf = pymupdf.open()
    try:
        for img in images:
            rgb = img.convert("RGB")
            buf = io.BytesIO()
            if use_jpeg:
                rgb.save(buf, format="JPEG", quality=10, subsampling="4:2:0")
            else:
                rgb.save(buf, format="PNG")
            # Scale pixel dimensions back to PDF points (72 pt/in at DEFAULT_PDF_DPI)
            # so loader.py re-renders each page at the exact degraded pixel dimensions.
            pt_scale = 72.0 / DEFAULT_PDF_DPI
            page = pdf.new_page(width=rgb.width * pt_scale, height=rgb.height * pt_scale)
            page.insert_image(page.rect, stream=buf.getvalue())
        pdf.save(out_path)
    finally:
        pdf.close()
    return out_path


def generate_degraded_dataset(
    *,
    sample_pack: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Create all 5 degraded variants of L01..L10 reproducibly from `seed`."""
    labelled_dir = get_labelled_dir(sample_pack)
    if not labelled_dir.exists():
        raise FileNotFoundError(f"Labelled directory not found: {labelled_dir}")

    doc_files = sorted(
        p
        for p in labelled_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not doc_files:
        raise FileNotFoundError(f"No supported documents found in {labelled_dir}")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "seed": seed,
        "source_labelled_dir": str(labelled_dir),
        "degradations": {},
    }

    for deg in DEGRADATION_TYPES:
        rng = random.Random(f"{seed}:{deg}")
        deg_dir = out_root / deg
        deg_entries: dict[str, Any] = {}
        transform = _TRANSFORMS[deg]

        for doc_path in doc_files:
            doc_id = doc_path.stem
            pages = load_document_images(doc_path)
            degraded_pages: list[Image.Image] = []
            page_params: list[dict[str, Any]] = []

            for page in pages:
                with Image.open(io.BytesIO(page.data)) as pil_img:
                    rgb = pil_img.convert("RGB")
                degraded_img, params = transform(rgb, rng)
                degraded_pages.append(degraded_img)
                page_params.append(params)

            saved_path = _save_degraded_document(degraded_pages, deg_dir, doc_id, deg)
            deg_entries[doc_id] = {
                "source": doc_path.name,
                "output": str(saved_path.relative_to(out_root)),
                "pages": len(degraded_pages),
                "params": page_params,
            }

        manifest["degradations"][deg] = deg_entries

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducibly generate degraded copies of the 10 labelled documents."
    )
    parser.add_argument(
        "--sample-pack",
        default=None,
        help="Path to the sample pack folder (defaults to SAMPLE_PACK_DIR / 'Sample Pack').",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Git-ignored output directory for degraded files (default: degraded_samples/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Fixed RNG seed for reproducible degradations (default: {DEFAULT_SEED}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path = generate_degraded_dataset(
            sample_pack=args.sample_pack,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated reproducible degraded dataset (seed={args.seed}).")
    print(f"Manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
