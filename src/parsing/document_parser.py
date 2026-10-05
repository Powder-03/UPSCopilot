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
from src.providers.vision import BaseVisionClient, get_vision_client
from src.utils.tracing import traceable

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


def _normalize_page_result(page: dict[str, Any]) -> dict[str, Any]:
    """
    Normalizes a single-page OCR result to a standard multi-section format.
    Supports both enhanced schema (continuation_answer + questions list)
    and legacy schema (single question per page).
    """
    p_num = page.get("page_num", 0)
    is_cover = bool(page.get("is_cover_or_rubric", False))
    is_blank = bool(page.get("is_blank", False))
    page_error = page.get("error")

    # If already using enhanced schema with "questions" list
    if "questions" in page and isinstance(page["questions"], list):
        questions = []
        for q_item in page["questions"]:
            if not isinstance(q_item, dict):
                continue
            q_num = q_item.get("q_num")
            try:
                q_num = int(q_num) if q_num is not None else None
            except (ValueError, TypeError):
                q_num = None
            max_marks = q_item.get("max_marks")
            try:
                max_marks = float(max_marks) if max_marks is not None else None
            except (ValueError, TypeError):
                max_marks = None
            q_text = q_item.get("question") or ""
            ans_text = (q_item.get("answer") or q_item.get("candidate_answer") or "").strip()
            diagrams = q_item.get("diagrams") or []
            questions.append({
                "q_num": q_num,
                "max_marks": max_marks,
                "question": q_text,
                "answer": ans_text,
                "diagrams": diagrams,
            })

        cont_ans = (page.get("continuation_answer") or "").strip()
        cont_diagrams = page.get("continuation_diagrams") or []

        return {
            "page_num": p_num,
            "is_cover_or_rubric": is_cover,
            "is_blank": is_blank,
            "continuation_answer": cont_ans,
            "continuation_diagrams": cont_diagrams,
            "questions": questions,
            "error": page_error,
        }

    # Otherwise normalize legacy schema
    has_header = bool(page.get("has_question_header", False))
    raw_q_num = page.get("q_num")
    try:
        q_num = int(raw_q_num) if raw_q_num is not None else None
    except (ValueError, TypeError):
        q_num = None

    ans_text = (page.get("candidate_answer") or page.get("answer") or "").strip()
    diagrams = page.get("diagrams") or []
    q_text = page.get("question") or ""
    max_marks = page.get("max_marks")
    try:
        max_marks = float(max_marks) if max_marks is not None else None
    except (ValueError, TypeError):
        max_marks = None

    questions = []
    cont_ans = ""
    cont_diagrams = []

    if has_header or q_num is not None:
        questions.append({
            "q_num": q_num,
            "max_marks": max_marks,
            "question": q_text,
            "answer": ans_text,
            "diagrams": diagrams,
        })
    else:
        cont_ans = ans_text
        cont_diagrams = diagrams

    return {
        "page_num": p_num,
        "is_cover_or_rubric": is_cover,
        "is_blank": is_blank,
        "continuation_answer": cont_ans,
        "continuation_diagrams": cont_diagrams,
        "questions": questions,
        "error": page_error,
    }


class DocumentParsingPipeline:
    """End-to-end ingestion pipeline for parsing scanned handwritten UPSC QCAB PDFs and mock tests."""

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

    @traceable(name="Process_Question_Pages", run_type="chain")
    def _process_question_slice(
        self,
        pdf_path: str | Path,
        q_slice: dict[str, Any],
        master_question: str | None = None,
    ) -> ParsedQuestion:
        """Processes a single question's pages: checks for blank sheets, invokes vision OCR if ink is present."""
        q_num = q_slice["q_num"]
        max_marks = q_slice["max_marks"]
        pages = q_slice.get("pages", [])

        if not pages:
            return ParsedQuestion(
                q_num=q_num,
                max_marks=max_marks,
                question=master_question or f"Question {q_num}",
                candidate_answer="",
                page_numbers=[],
                is_blank=True,
                word_count=0,
            )

        logger.info("Processing Q%02d (Pages %s)...", q_num, pages)

        try:
            # Check if all pages in this slice are visually blank
            all_blank = True
            page_images_bytes: list[bytes] = []

            for p in pages:
                # Single-pass C rendering & blank detection (avoids double-rendering and memory bloat)
                img_bytes, is_blank = self.preprocessor.render_page_processed(
                    pdf_path, p, quality=80
                )
                if not is_blank:
                    all_blank = False
                page_images_bytes.append(img_bytes)

            # Fast path for visually blank / unattempted sheets ONLY if master question is already known
            if all_blank and master_question:
                logger.info("Q%02d detected as completely blank/unattempted. Skipping vision API.", q_num)
                return ParsedQuestion(
                    q_num=q_num,
                    max_marks=max_marks,
                    question=master_question,
                    candidate_answer="",
                    page_numbers=pages,
                    is_blank=True,
                    word_count=0,
                )

            # Invoke Multimodal Vision to transcribe handwriting and diagrams
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
            logger.error("Error transcribing Q%02d: %s", q_num, e)
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

    @traceable(name="Process_Single_Page", run_type="chain")
    def _parse_single_page_worker(
        self,
        pdf_path: str | Path,
        page_num: int,
    ) -> dict[str, Any]:
        """Processes a single page: checks for visual blankness, invokes vision OCR if ink is present."""
        try:
            img_bytes, is_blank = self.preprocessor.render_page_processed(
                pdf_path, page_num, quality=80
            )
            if is_blank:
                return {
                    "page_num": page_num,
                    "has_question_header": False,
                    "is_cover_or_rubric": False,
                    "is_blank": True,
                    "q_num": None,
                    "max_marks": None,
                    "question": "",
                    "candidate_answer": "",
                    "diagrams": [],
                    "continuation_answer": "",
                    "continuation_diagrams": [],
                    "questions": [],
                }

            result = self.vision_client.parse_single_page(
                image_bytes=img_bytes,
                page_num=page_num,
                image_format="jpeg",
            )
            return result
        except Exception as e:
            logger.error("Error transcribing Page %d: %s", page_num, e)
            return {
                "page_num": page_num,
                "has_question_header": False,
                "is_cover_or_rubric": False,
                "is_blank": False,
                "q_num": None,
                "max_marks": None,
                "question": "",
                "candidate_answer": "",
                "diagrams": [],
                "continuation_answer": "",
                "continuation_diagrams": [],
                "questions": [],
                "error": str(e),
            }

    def _stitch_pages_into_questions(
        self,
        page_results: list[dict[str, Any]],
        master_questions: list[dict[str, Any]] | None = None,
        total_expected_questions: int | None = None,
    ) -> list[ParsedQuestion]:
        """
        Dynamically stitches single-page OCR results into complete multi-page ParsedQuestions.
        Never makes fixed-page assumptions (e.g. Q1 is on pages 3-4, Q2 on pages 5-6).
        Automatically skips cover pages and instructions, recognizes new question headers anywhere,
        and concatenates continuation pages and split answers without data loss.
        """
        master_map: dict[int, dict[str, Any]] = {}
        if master_questions:
            for item in master_questions:
                q_n = int(item.get("q_num", 0))
                if q_n:
                    master_map[q_n] = item

        normalized_pages = [_normalize_page_result(p) for p in page_results]
        sorted_pages = sorted(normalized_pages, key=lambda x: x.get("page_num", 0))

        questions_by_num: dict[int, ParsedQuestion] = {}
        ordered_q_nums: list[int] = []
        active_q_num: int | None = None
        q_counter = 0

        for page in sorted_pages:
            p_num = page["page_num"]
            is_cover = page["is_cover_or_rubric"]
            is_blank = page["is_blank"]
            cont_ans = page["continuation_answer"]
            cont_diag = page["continuation_diagrams"]
            q_list = page["questions"]
            page_error = page["error"]

            has_candidate_content = bool(cont_ans or cont_diag or q_list)

            # Skip cover / rubric sheets ONLY if there is no candidate writing
            if (is_cover or is_blank) and not has_candidate_content:
                logger.info("Skipping blank or cover/rubric on Page %d", p_num)
                continue

            # 1. Process continuation answer (top of page text continuing previous question)
            if cont_ans or cont_diag:
                if active_q_num is not None and active_q_num in questions_by_num:
                    target_q = questions_by_num[active_q_num]
                    if cont_ans:
                        if target_q.candidate_answer:
                            target_q.candidate_answer = f"{target_q.candidate_answer}\n\n{cont_ans}".strip()
                        else:
                            target_q.candidate_answer = cont_ans.strip()
                    if p_num not in target_q.page_numbers:
                        target_q.page_numbers.append(p_num)
                    for d in cont_diag:
                        if d not in target_q.diagrams:
                            target_q.diagrams.append(d)
                else:
                    # No active question was previously opened (e.g. Page 1 starts with handwriting directly)
                    q_counter += 1
                    active_q_num = q_counter
                    master_info = master_map.get(active_q_num, {})
                    resolved_q_text = master_info.get("question") or f"Question {active_q_num}"
                    max_marks = float(master_info.get("max_marks", 10.0 if active_q_num <= 10 else 15.0))

                    new_q = ParsedQuestion(
                        q_num=active_q_num,
                        max_marks=max_marks,
                        question=resolved_q_text,
                        candidate_answer=cont_ans,
                        page_numbers=[p_num],
                        diagrams=list(cont_diag),
                        is_blank=False,
                        word_count=len(cont_ans.split()),
                        error=page_error,
                    )
                    questions_by_num[active_q_num] = new_q
                    ordered_q_nums.append(active_q_num)

            # 2. Process new questions starting on this page
            for q_data in q_list:
                raw_q_num = q_data.get("q_num")
                if raw_q_num is not None:
                    resolved_q_num = int(raw_q_num)
                    q_counter = max(q_counter, resolved_q_num)
                else:
                    q_counter += 1
                    resolved_q_num = q_counter

                master_info = master_map.get(resolved_q_num, {})
                raw_q_text = q_data.get("question")
                if master_info.get("question"):
                    resolved_q_text = master_info["question"]
                else:
                    resolved_q_text = _clean_question_text(raw_q_text) or f"Question {resolved_q_num}"

                if master_info.get("max_marks"):
                    max_marks = float(master_info["max_marks"])
                else:
                    max_marks = float(q_data.get("max_marks") or (10.0 if resolved_q_num <= 10 else 15.0))

                ans_text = q_data.get("answer") or ""
                diagrams = q_data.get("diagrams") or []

                if resolved_q_num in questions_by_num:
                    # Merge if question already exists (e.g. candidate split answer across pages)
                    existing = questions_by_num[resolved_q_num]
                    if ans_text and ans_text not in existing.candidate_answer:
                        if existing.candidate_answer:
                            existing.candidate_answer = f"{existing.candidate_answer}\n\n{ans_text}".strip()
                        else:
                            existing.candidate_answer = ans_text.strip()
                    if p_num not in existing.page_numbers:
                        existing.page_numbers.append(p_num)
                    for d in diagrams:
                        if d not in existing.diagrams:
                            existing.diagrams.append(d)
                else:
                    new_q = ParsedQuestion(
                        q_num=resolved_q_num,
                        max_marks=max_marks,
                        question=resolved_q_text,
                        candidate_answer=ans_text,
                        page_numbers=[p_num],
                        diagrams=list(diagrams),
                        is_blank=False,
                        word_count=len(ans_text.split()),
                        error=page_error,
                    )
                    questions_by_num[resolved_q_num] = new_q
                    ordered_q_nums.append(resolved_q_num)

                active_q_num = resolved_q_num

        # Recalculate word counts and blank states
        for q in questions_by_num.values():
            q.candidate_answer = q.candidate_answer.strip()
            q.word_count = len(q.candidate_answer.split())
            if not q.candidate_answer and not q.diagrams:
                q.is_blank = True

        extracted_list = [questions_by_num[qn] for qn in ordered_q_nums]

        # Reconcile, deduplicate, and sort canonically
        final_questions = self.segmenter.reconcile_and_sort_questions(
            extracted_questions=extracted_list,
            master_questions=master_questions,
            total_expected_questions=total_expected_questions,
        )
        return final_questions

    @traceable(name="DocumentParsingPipeline.parse_pdf", run_type="chain")
    def parse_pdf(
        self,
        pdf_path: str | Path,
        master_questions: list[dict[str, Any]] | None = None,
        master_questions_path: str | Path | None = None,
        limit_questions: int | None = None,
        max_pages: int | None = None,
        start_page: int | None = None,
        mode: str = "dynamic",
        total_expected_questions: int | None = None,
    ) -> ParsedDocument:
        """
        Parses any scanned UPSC PDF into a complete ParsedDocument containing all questions.

        Args:
            pdf_path: Path to the PDF file.
            master_questions: Optional canonical list of questions to reconcile against.
            master_questions_path: Optional path to JSON file with canonical questions.
            limit_questions: Optional limit on number of questions to parse.
            max_pages: Optional maximum number of pages to inspect.
            start_page: Optional start page (1-indexed, default: auto-detects from page 1).
            mode: "dynamic" (default, parses page-by-page concurrently and stitches dynamically without
                  any fixed page count assumptions), or "slice" (legacy fixed-offset mathematical slicing).
            total_expected_questions: Optional expected total questions (e.g. 10 for sectional, 20 for FLT).
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if master_questions_path and master_questions is None:
            with open(master_questions_path, encoding="utf-8") as f:
                master_questions = json.load(f)

        total_pages = self.preprocessor.get_page_count(pdf_file)
        if max_pages:
            total_pages = min(total_pages, max_pages)

        logger.info(
            "Starting ingestion pipeline for '%s' (%d total pages, mode=%s)...",
            pdf_file.name,
            total_pages,
            mode,
        )

        import contextvars

        if mode == "dynamic":
            first_page = start_page or 1
            pages_to_parse = list(range(first_page, total_pages + 1))
            page_results: list[dict[str, Any]] = []

            # Dynamic concurrent page-by-page OCR
            workers = min(self.max_workers, 15)
            logger.info("Executing dynamic page OCR across %d workers for %d pages...", workers, len(pages_to_parse))

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_page = {
                    executor.submit(
                        contextvars.copy_context().run,
                        self._parse_single_page_worker,
                        pdf_file,
                        p,
                    ): p
                    for p in pages_to_parse
                }

                for future in as_completed(future_to_page):
                    try:
                        p_res = future.result()
                        page_results.append(p_res)
                    except Exception as exc:
                        p_num = future_to_page[future]
                        logger.error("Page %d worker generated an exception: %s", p_num, exc)
                        page_results.append({
                            "page_num": p_num,
                            "has_question_header": False,
                            "is_cover_or_rubric": False,
                            "is_blank": False,
                            "q_num": None,
                            "max_marks": None,
                            "question": "",
                            "candidate_answer": "",
                            "diagrams": [],
                            "continuation_answer": "",
                            "continuation_diagrams": [],
                            "questions": [],
                            "error": str(exc),
                        })

            final_questions = self._stitch_pages_into_questions(
                page_results=page_results,
                master_questions=master_questions,
                total_expected_questions=total_expected_questions,
            )
            if limit_questions:
                final_questions = final_questions[:limit_questions]

            return ParsedDocument(
                source_file=str(pdf_file.resolve()),
                total_pages=total_pages,
                questions=final_questions,
                metadata={
                    "parsed_questions_count": len(final_questions),
                    "model_used": self.vision_client.model_id,
                    "mode": "dynamic",
                },
            )

        # Legacy fixed slice fallback
        slices = self.segmenter.calculate_default_qcab_page_slices(total_pages, start_page=start_page)
        if limit_questions:
            slices = slices[:limit_questions]

        master_map: dict[int, str] = {}
        if master_questions:
            for item in master_questions:
                q_n = int(item.get("q_num", 0))
                if q_n:
                    master_map[q_n] = item.get("question", "")

        extracted_questions: list[ParsedQuestion] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_slice = {
                executor.submit(
                    contextvars.copy_context().run,
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
                "mode": "slice",
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
        mode: str = "dynamic",
        total_expected_questions: int | None = None,
    ) -> Path:
        """
        Convenience method: parses a PDF and writes the output JSON file ready for the evaluation engine.
        """
        parsed_doc = self.parse_pdf(
            pdf_path=pdf_path,
            master_questions_path=master_questions_path,
            limit_questions=limit_questions,
            max_pages=max_pages,
            start_page=start_page,
            mode=mode,
            total_expected_questions=total_expected_questions,
        )

        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        evaluation_data = parsed_doc.to_evaluation_list()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Parsed JSON successfully saved to: {out_path} ({len(evaluation_data)} questions)")
        return out_path

