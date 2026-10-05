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
        elif total_pages >= 52:
            # 52-56 page FLT mock tests / UPSC booklets typically have Cover (Page 1) + Rubric/Instructions (Page 2)
            current_page = 3
        elif total_pages == 51:
            # 1 cover page (Page 1) + 50 question pages
            current_page = 2
        else:
            current_page = 1

        # Q1 to Q10: 10 Markers (2 pages each)
        for q in range(1, 11):
            if current_page > total_pages:
                slices.append({
                    "q_num": q,
                    "max_marks": 10.0,
                    "pages": [],
                })
                continue
            end_page = min(current_page + 1, total_pages)
            slices.append({
                "q_num": q,
                "max_marks": 10.0,
                "pages": list(range(current_page, end_page + 1)),
            })
            current_page = end_page + 1

        # Q11 to Q20: 15 Markers (3 pages each)
        for q in range(11, 21):
            if current_page > total_pages:
                slices.append({
                    "q_num": q,
                    "max_marks": 15.0,
                    "pages": [],
                })
                continue
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
        total_expected_questions: int | None = None,
    ) -> list[ParsedQuestion]:
        """
        Reconciles extracted questions:
        1. De-duplicates and merges multi-page answers for the same q_num.
        2. Sorts canonically by q_num (1 to N).
        3. Reconciles against master question paper (if provided) or detects skipped questions in sequence.
        4. Injects blank/unattempted placeholders ONLY for genuinely skipped questions (supports sectional tests).
        """
        # Map by q_num, cleanly merging multi-page / multi-section answers
        q_map: dict[int, ParsedQuestion] = {}
        for q in extracted_questions:
            if q.q_num not in q_map:
                q_map[q.q_num] = q
            else:
                existing = q_map[q.q_num]
                # Merge answer text if new content is present
                if q.candidate_answer and q.candidate_answer not in existing.candidate_answer:
                    if existing.candidate_answer:
                        existing.candidate_answer = f"{existing.candidate_answer}\n\n{q.candidate_answer}".strip()
                    else:
                        existing.candidate_answer = q.candidate_answer.strip()
                    existing.word_count = len(existing.candidate_answer.split())
                    existing.is_blank = existing.word_count == 0 and not existing.diagrams
                # Merge page numbers
                for p in q.page_numbers:
                    if p not in existing.page_numbers:
                        existing.page_numbers.append(p)
                existing.page_numbers.sort()
                # Merge diagrams
                for d in q.diagrams:
                    if d not in existing.diagrams:
                        existing.diagrams.append(d)
                # Keep richer question prompt if existing was generic
                if (not existing.question or len(existing.question) < 15) and q.question:
                    existing.question = q.question

        # If a master question paper is provided, reconcile against master
        if master_questions:
            for idx, master in enumerate(master_questions, start=1):
                q_num = int(master.get("q_num", idx))
                if q_num not in q_map:
                    logger.info("Question %d missing from extraction. Injecting placeholder.", q_num)
                    q_map[q_num] = ParsedQuestion(
                        q_num=q_num,
                        max_marks=float(master.get("max_marks", 10.0 if q_num <= 10 else 15.0)),
                        question=master.get("question", f"Question {q_num}"),
                        candidate_answer="",
                        page_numbers=[],
                        is_blank=True,
                        word_count=0,
                    )
                else:
                    if not q_map[q_num].question or len(q_map[q_num].question) < 10:
                        q_map[q_num].question = master.get("question", q_map[q_num].question)
                    if "max_marks" in master:
                        q_map[q_num].max_marks = float(master["max_marks"])
        else:
            # Full booklet scan or mock test without master:
            if extracted_questions:
                max_q = max((q.q_num for q in extracted_questions), default=0)
                if total_expected_questions is not None:
                    target_count = total_expected_questions
                elif max_q >= 18:
                    # Standard 20-question Full-Length Test (FLT)
                    target_count = 20
                else:
                    # Sectional or mini-mock test: target is the maximum question attempted
                    target_count = max(max_q, 1)

                for q_num in range(1, target_count + 1):
                    if q_num not in q_map:
                        logger.info("Question %d missing from extraction. Injecting placeholder.", q_num)
                        q_map[q_num] = ParsedQuestion(
                            q_num=q_num,
                            max_marks=10.0 if q_num <= 10 else 15.0,
                            question=f"Question {q_num}",
                            candidate_answer="",
                            page_numbers=[],
                            is_blank=True,
                            word_count=0,
                        )

        # Return canonically sorted list
        return sorted(q_map.values(), key=lambda q: q.q_num)
