"""Master Document Parsing Pipeline: converts any scanned UPSC answer booklet PDF into clean, evaluated JSON."""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.config import settings
from src.models.parsing import ParsedDocument, ParsedQuestion
from src.parsing.preprocessor import PDFPreprocessor
from src.parsing.segmenter import QCABSegmenter
from src.parsing.vision_client import BaseVisionClient, get_vision_client

logger = logging.getLogger(__name__)


def _clean_question_text(raw_text: str) -> str:
    """Strips residual Devanagari script, question numbering prefixes, and trailing marks/word limit metadata."""
    if not raw_text:
        return ""
    text = raw_text.strip()

    # If text has multiple lines, filter out lines that are exclusively or primarily Devanagari
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    english_lines: list[str] = []
    for line in lines:
        devanagari_count = len(re.findall(r"[\u0900-\u097F]", line))
        latin_count = len(re.findall(r"[A-Za-z]", line))
        if latin_count >= 5 and latin_count >= devanagari_count:
            cleaned_line = re.sub(r"[\u0900-\u097F]+", " ", line)
            cleaned_line = " ".join(cleaned_line.split())
            if cleaned_line:
                english_lines.append(cleaned_line)
        elif latin_count >= 15:
            cleaned_line = re.sub(r"[\u0900-\u097F]+", " ", line)
            cleaned_line = " ".join(cleaned_line.split())
            if cleaned_line:
                english_lines.append(cleaned_line)

    if english_lines:
        text = " ".join(english_lines)
    else:
        # Fallback if single line had both or no lines passed filter: strip Devanagari if latin exists
        if re.search(r"[A-Za-z]{4,}", text):
            text = re.sub(r"[\u0900-\u097F]+", " ", text)
            text = " ".join(text.split())

    # Strip leading question numbering (e.g. "Q1.", "Q.1:", "Question 1:", "1.", "1 -", "1)")
    text = re.sub(
        r"^(?:(?:Question|Q)\.?\s*\d+[\s.:\-\)]*|\d+[\s.:\-\)]+)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Strip trailing marks/word limits like "(10 Marks, 150 words)", "(15 marks)", "(150 Words)", "(10M)", "(15M)"
    text = re.sub(
        r"\s*\(\s*(?:\d+\s*(?:Marks?|marks?|M|m)?(?:\s*[,/]\s*)?)?(?:\d+\s*(?:words?|Words?)?)?\s*\)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Strip standalone trailing marks / word counts
    text = re.sub(r"\s+\d+\s*(?:Marks?|marks?)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\d+\s*(?:words?|Words?)\s*$", "", text, flags=re.IGNORECASE)

    return text.strip()


class DocumentParsingPipeline:
    """End-to-end ingestion pipeline for parsing scanned handwritten UPSC QCAB PDFs."""

    def __init__(
        self,
        vision_model_id: str | None = None,
        vision_client: BaseVisionClient | None = None,
        dpi: int | None = None,
        max_workers: int | None = None,
    ):
        self.preprocessor = PDFPreprocessor(dpi=dpi)
        self.vision_client = vision_client or get_vision_client(model_id=vision_model_id)
        self.segmenter = QCABSegmenter()
        self.max_workers = max_workers or settings.parsing_max_workers


    def _process_question_slice(
        self,
        pdf_path: str | Path,
        q_slice: dict[str, Any],
        master_question: str | None = None,
    ) -> ParsedQuestion:
        """Processes a single question's pages: checks for blank sheets, invokes vision OCR if ink is present."""
        q_num = q_slice["q_num"]
        max_marks = q_slice["max_marks"]
        pages = q_slice["pages"]

        logger.info(f"Processing Q{q_num:02d} (Pages {pages})...")

        # Check if all pages in this slice are visually blank
        all_blank = True
        page_images_bytes: list[bytes] = []

        for p in pages:
            img = self.preprocessor.render_page_to_image(pdf_path, p)
            is_blank = self.preprocessor.is_page_visually_blank(img)
            if not is_blank:
                all_blank = False
            # Render bytes for vision model (compact JPEG under 200 KB per page)
            img_bytes = self.preprocessor.render_page_to_bytes(
                pdf_path, p, img_format="JPEG", max_dimension=1600, quality=80
            )
            page_images_bytes.append(img_bytes)

        # Fast path for visually blank / unattempted sheets (saves API tokens)
        if all_blank:
            logger.info(f"Q{q_num:02d} detected as completely blank/unattempted. Skipping vision API.")
            return ParsedQuestion(
                q_num=q_num,
                max_marks=max_marks,
                question=master_question or f"Question {q_num}",
                candidate_answer="",
                page_numbers=pages,
                is_blank=True,
                word_count=0,
            )

        # Invoke Bedrock Vision to transcribe handwriting and diagrams
        try:
            extracted = self.vision_client.extract_answer_from_page_images(
                image_bytes_list=page_images_bytes,
                image_format="jpeg",
                fallback_question=master_question,
                expected_q_num=q_num,
            )

            resolved_q_num = int(extracted.get("q_num", q_num))
            raw_q = extracted.get("question") or master_question or f"Question {q_num}"
            resolved_question = master_question if master_question else (_clean_question_text(raw_q) or f"Question {q_num}")
            candidate_answer = extracted.get("candidate_answer", "")
            diagrams = extracted.get("diagrams_found", [])
            is_blank = bool(extracted.get("is_blank", not candidate_answer.strip()))

            return ParsedQuestion(
                q_num=resolved_q_num,
                max_marks=float(extracted.get("max_marks", max_marks)),
                question=resolved_question,
                candidate_answer=candidate_answer,
                page_numbers=pages,
                diagrams=diagrams,
                is_blank=is_blank,
                word_count=len(candidate_answer.split()),
            )
        except Exception as e:
            logger.error(f"Error transcribing Q{q_num:02d}: {e}")
            # Record transcription error rather than fraudulently marking the page as an unattempted blank
            return ParsedQuestion(
                q_num=q_num,
                max_marks=max_marks,
                question=master_question or f"Question {q_num}",
                candidate_answer="",
                page_numbers=pages,
                is_blank=False,
                word_count=0,
                error=f"Transcription failed: {e}",
            )

    def parse_pdf(
        self,
        pdf_path: str | Path,
        master_questions: list[dict[str, Any]] | None = None,
        limit_questions: int | None = None,
        max_pages: int | None = None,
        start_page: int | None = None,
    ) -> ParsedDocument:
        """
        Parses any scanned UPSC PDF into a complete ParsedDocument containing all questions.
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        total_pages = self.preprocessor.get_page_count(pdf_file)
        if max_pages:
            total_pages = min(total_pages, max_pages)

        logger.info(f"Starting ingestion pipeline for '{pdf_file.name}' ({total_pages} total pages)...")

        slices = self.segmenter.calculate_default_qcab_page_slices(total_pages, start_page=start_page)
        if limit_questions:
            slices = slices[:limit_questions]

        # Build master lookup if provided
        master_map: dict[int, str] = {}
        if master_questions:
            for item in master_questions:
                q_n = int(item.get("q_num", 0))
                if q_n:
                    master_map[q_n] = item.get("question", "")

        extracted_questions: list[ParsedQuestion] = []

        # Execute extraction across parallel worker threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_slice = {
                executor.submit(
                    self._process_question_slice,
                    pdf_file,
                    s,
                    master_map.get(s["q_num"]),
                ): s
                for s in slices
            }

            for future in as_completed(future_to_slice):
                try:
                    q_res = future.result()
                    extracted_questions.append(q_res)
                except Exception as exc:
                    logger.error(f"Worker generated an exception: {exc}")

        # Reconcile, de-duplicate, and sort canonically 1 to 20
        final_questions = self.segmenter.reconcile_and_sort_questions(
            extracted_questions=extracted_questions,
            master_questions=master_questions,
        )

        return ParsedDocument(
            source_file=str(pdf_file.resolve()),
            total_pages=total_pages,
            questions=final_questions,
            metadata={
                "parsed_questions_count": len(final_questions),
                "model_used": self.vision_client.model_id,
            },
        )

    def parse_pdf_to_json_file(
        self,
        pdf_path: str | Path,
        output_json_path: str | Path,
        master_questions_path: str | Path | None = None,
        limit_questions: int | None = None,
        max_pages: int | None = None,
        start_page: int | None = None,
    ) -> Path:
        """
        Convenience method: parses a PDF and writes the output JSON file ready for the evaluation engine.
        """
        master_questions = None
        if master_questions_path:
            with open(master_questions_path, encoding="utf-8") as f:
                master_questions = json.load(f)

        parsed_doc = self.parse_pdf(
            pdf_path=pdf_path,
            master_questions=master_questions,
            limit_questions=limit_questions,
            max_pages=max_pages,
            start_page=start_page,
        )

        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        evaluation_data = parsed_doc.to_evaluation_list()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Parsed JSON successfully saved to: {out_path} ({len(evaluation_data)} questions)")
        return out_path
