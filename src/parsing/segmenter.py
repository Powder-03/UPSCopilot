"""QCAB Layout Segmenter: maps pages to question boundaries, handles missing questions and canonical re-sorting."""
import logging
from typing import Any

from src.models.parsing import ParsedQuestion

logger = logging.getLogger(__name__)


class QCABSegmenter:
    """Manages question boundary mapping, missing question reconciliation, and canonical 1-to-20 sorting."""

    @staticmethod
    def calculate_default_qcab_page_slices(
        total_pages: int,
        start_page: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Calculates standard UPSC QCAB page boundaries:
        - Optional front matter (cover, rubric sheet): Pages 1 to start_page - 1
        - Questions 1 to 10 (10 Marks): 2 pages each
        - Questions 11 to 20 (15 Marks): 3 pages each
        Total expected pages: 1-2 (front matter) + 10*2 (20) + 10*3 (30) = 51 to 56 pages.
        """
        slices: list[dict[str, Any]] = []
        if start_page is not None:
            current_page = start_page
        elif total_pages >= 55:
            # 55-56 page FLT mock tests typically have Cover (Page 1) + Rubric/Feedback sheet (Page 2)
            current_page = 3
        elif total_pages > 50:
            current_page = 2
        else:
            current_page = 1

        # Q1 to Q10: 10 Markers (2 pages each)
        for q in range(1, 11):
            end_page = min(current_page + 1, total_pages)
            slices.append({
                "q_num": q,
                "max_marks": 10.0,
                "pages": list(range(current_page, end_page + 1)),
            })
            current_page = end_page + 1
            if current_page > total_pages:
                break

        # Q11 to Q20: 15 Markers (3 pages each)
        for q in range(11, 21):
            if current_page > total_pages:
                break
            end_page = min(current_page + 2, total_pages)
            slices.append({
                "q_num": q,
                "max_marks": 15.0,
                "pages": list(range(current_page, end_page + 1)),
            })
            current_page = end_page + 1

        return slices

    @staticmethod
    def reconcile_and_sort_questions(
        extracted_questions: list[ParsedQuestion],
        master_questions: list[dict[str, Any]] | None = None,
    ) -> list[ParsedQuestion]:
        """
        Reconciles extracted questions:
        1. De-duplicates and sorts canonically by q_num (1 to 20).
        2. Detects any missing / skipped questions.
        3. Injects blank/unattempted placeholders from the Master Question Paper if available.
        """
        # Map by q_num, keeping the entry with the highest word count if duplicate scans exist
        q_map: dict[int, ParsedQuestion] = {}
        for q in extracted_questions:
            if q.q_num not in q_map or q.word_count > q_map[q.q_num].word_count:
                q_map[q.q_num] = q

        # If a master question paper is provided, ensure all 20 questions exist
        if master_questions:
            for idx, master in enumerate(master_questions, start=1):
                q_num = int(master.get("q_num", idx))
                if q_num not in q_map:
                    logger.info(f"Question {q_num} missing from scan. Injecting unattempted placeholder.")
                    q_map[q_num] = ParsedQuestion(
                        q_num=q_num,
                        max_marks=float(master.get("max_marks", 10.0 if q_num <= 10 else 15.0)),
                        question=master.get("question", f"Question {q_num}"),
                        candidate_answer="",
                        page_numbers=[],
                        is_blank=True,
                        word_count=0,
                    )
                elif not q_map[q_num].question or len(q_map[q_num].question) < 10:
                    # Enrich question text from master if OCR got truncated
                    q_map[q_num].question = master.get("question", q_map[q_num].question)

        # Return canonically sorted list
        return sorted(q_map.values(), key=lambda q: q.q_num)
