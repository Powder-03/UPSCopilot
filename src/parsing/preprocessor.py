"""Document Preprocessor: extracts, normalizes, and renders PDF pages to images using PyMuPDF and Pillow."""
import gc
import logging
from collections.abc import Generator
from pathlib import Path

import pymupdf
from PIL import Image

from src.config import settings

logger = logging.getLogger(__name__)


class PDFPreprocessor:
    """Renders PDF pages into optimized, standardized images for multimodal vision processing."""

    def __init__(self, dpi: int | None = None):
        self.dpi = dpi or settings.parsing_dpi
        # Standard zoom matrix: 72 DPI is base in PDF, so zoom = dpi / 72
        self.zoom = self.dpi / 72.0
        self.matrix = pymupdf.Matrix(self.zoom, self.zoom)

    def get_page_count(self, pdf_path: str | Path) -> int:
        """Returns the total number of pages in the PDF."""
        with pymupdf.open(str(pdf_path)) as doc:
            return len(doc)

    def render_page_processed(
        self,
        pdf_path: str | Path,
        page_num: int,
        quality: int = 80,
        darkness_threshold: float = 0.005,
        max_dimension: int = 1600,
        doc: pymupdf.Document | None = None,
    ) -> tuple[bytes, bool]:
        """
        Single-pass rendering: extracts page, checks visual blankness directly on C pixmap samples,
        and compresses directly to JPEG bytes in C with zero redundant intermediate bitmap allocations.
        Returns: (jpeg_bytes, is_blank).
        """
        close_doc = False
        active_doc = doc
        if active_doc is None:
            active_doc = pymupdf.open(str(pdf_path))
            close_doc = True
        try:
            if page_num < 1 or page_num > len(active_doc):
                raise ValueError(f"Page number {page_num} is out of bounds (1 to {len(active_doc)}).")
            page = active_doc[page_num - 1]
            pix = page.get_pixmap(matrix=self.matrix, alpha=False)

            # Fast in-memory blank check directly on C pixmap samples (subsampling every 4th pixel)
            samples = pix.samples
            total_pixels = pix.width * pix.height
            dark_pixels = 0
            step = 12  # 4 pixels * 3 channels (RGB)
            for i in range(0, len(samples), step):
                if (samples[i] + samples[i + 1] + samples[i + 2]) < 660:  # average < 220
                    dark_pixels += 1
            sampled_total = max(1, total_pixels // 4)
            is_blank = (dark_pixels / float(sampled_total)) < darkness_threshold

            # Direct C-level JPEG compression (avoids PIL Image and BytesIO)
            img_bytes = pix.tobytes("jpeg", jpg_quality=quality)
            del pix
            gc.collect()
            return img_bytes, is_blank
        finally:
            if close_doc and active_doc is not None:
                active_doc.close()

    def render_page_to_image(self, pdf_path: str | Path, page_num: int) -> Image.Image:
        """
        Renders a single 1-indexed page of a PDF to a PIL RGB Image.
        """
        with pymupdf.open(str(pdf_path)) as doc:
            if page_num < 1 or page_num > len(doc):
                raise ValueError(f"Page number {page_num} is out of bounds (1 to {len(doc)}).")
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=self.matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            del pix
            return img

    def render_page_to_bytes(
        self,
        pdf_path: str | Path,
        page_num: int,
        img_format: str = "JPEG",
        max_dimension: int = 1600,
        quality: int = 80,
    ) -> bytes:
        """
        Renders a 1-indexed page to image bytes using single-pass C-level JPEG compression.
        """
        img_bytes, _ = self.render_page_processed(
            pdf_path=pdf_path,
            page_num=page_num,
            quality=quality,
            max_dimension=max_dimension,
        )
        return img_bytes

    def is_page_visually_blank(self, img: Image.Image, darkness_threshold: float = 0.005) -> bool:
        """
        Lightweight visual heuristic: checks if a rendered page has virtually zero ink/writing.
        Returns True if the fraction of non-white / dark pixels is below the threshold.
        """
        grayscale = img.convert("L")
        # Invert so black ink has high values
        # Count pixels that are noticeably darker than white paper background (pixel value < 220)
        width, height = grayscale.size
        total_pixels = width * height
        # Fast histogram of 256 grayscale values: sum counts for values < 220 (noticeably darker than white paper)
        hist = grayscale.histogram()
        dark_pixels = sum(hist[:220])
        dark_ratio = dark_pixels / float(total_pixels)
        return dark_ratio < darkness_threshold

    def iter_rendered_pages(
        self,
        pdf_path: str | Path,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> Generator[tuple[int, Image.Image, bool], None, None]:
        """
        Yields (page_num, PIL_image, is_blank) for each page in the specified range.
        """
        total = self.get_page_count(pdf_path)
        last_page = min(end_page or total, total)
        for p in range(start_page, last_page + 1):
            img = self.render_page_to_image(pdf_path, p)
            blank = self.is_page_visually_blank(img)
            yield p, img, blank
