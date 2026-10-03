"""Unified End-to-End Evaluation Pipeline: connects PDF parsing, KB-grounded evaluation, and student scorecard distillation."""
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.evaluation.engine import UPSCEvaluationEngine
from src.evaluation.model_factory import get_eval_llm
from src.models.api import (
    JobStatus,
    OverallFeedback,
    StudentEvaluationReport,
    StudentQuestionEvaluation,
    StudentSummary,
)
from src.models.evaluation import EvaluationResult
from src.models.exceptions import DocumentParsingError
from src.models.parsing import ParsedDocument, ParsedQuestion
from src.parsing.pipeline import DocumentParsingPipeline
from src.utils.tracing import traceable

logger = logging.getLogger(__name__)


class UnifiedEvaluationPipeline:
    """Orchestrates end-to-end PDF parsing, evaluation against knowledge base, and student scorecard distillation."""

    def __init__(
        self,
        vision_model_id: str | None = None,
        eval_model_id: str | None = None,
        workers: int = 4,
    ):
        self.parser = DocumentParsingPipeline(
            vision_model_id=vision_model_id,
            max_workers=workers,
        )
        eval_llm = get_eval_llm(model_id=eval_model_id) if eval_model_id else None
        self.engine = UPSCEvaluationEngine(eval_llm=eval_llm)
        self.workers = workers

    @traceable(name="UnifiedEvaluationPipeline.run_pipeline", run_type="chain")
    def run_pipeline(
        self,
        pdf_path: str | Path,
        start_page: int | None = None,
        max_pages: int | None = None,
        master_questions_path: str | Path | None = None,
        progress_callback: Callable[[JobStatus, int, str], None] | None = None,
    ) -> StudentEvaluationReport:
        """
        Executes end-to-end PDF processing:
        1. Parse pages & transcribe handwriting using Bedrock vision.
        2. Evaluate each question against the UPSC KB using 2-call CoT + G-Eval.
        3. Distill into clean student scorecard (strictly zero performance bands).
        """
        def update_progress(status: JobStatus, pct: int, step: str) -> None:
            if progress_callback:
                progress_callback(status, pct, step)

        # ---------------------------------------------------------
        # STAGE 1: Document Parsing & Vision Transcription
        # ---------------------------------------------------------
        update_progress(JobStatus.PARSING, 10, "Slicing document pages and checking for blank sheets...")
        parsed_doc: ParsedDocument = self.parser.parse_pdf(
            pdf_path=pdf_path,
            start_page=start_page,
            max_pages=max_pages,
            master_questions_path=master_questions_path,
        )

        total_questions = len(parsed_doc.questions)
        if total_questions == 0:
            raise ValueError(f"No questions could be extracted from {pdf_path}.")

        logger.info("Successfully parsed %d questions from %s.", total_questions, pdf_path)
        update_progress(JobStatus.EVALUATING, 30, f"Beginning evaluation for {total_questions} questions...")

        # ---------------------------------------------------------
        # STAGE 2: Multi-Question Evaluation
        # ---------------------------------------------------------
        eval_pairs: list[tuple[ParsedQuestion, EvaluationResult]] = []

        # If running multi-worker, evaluate questions in parallel; otherwise sequential
        if self.workers > 1 and total_questions > 1:
            eval_pairs = self._evaluate_concurrently(parsed_doc.questions, update_progress)
        else:
            eval_pairs = self._evaluate_sequentially(parsed_doc.questions, update_progress)

        # Sort back into canonical question order (1 to N)
        eval_pairs.sort(key=lambda pair: pair[0].q_num)

        # ---------------------------------------------------------
        # STAGE 3: Distillation into Student Scorecard (Zero Performance Bands)
        # ---------------------------------------------------------
        update_progress(JobStatus.DISTILLING, 95, "Distilling actionable strengths and improvements...")
        report = self.distill_results(
            document_name=Path(pdf_path).name,
            evaluations=eval_pairs,
        )

        update_progress(JobStatus.COMPLETED, 100, "Evaluation complete! Student scorecard ready.")
        return report

    def _evaluate_sequentially(
        self,
        questions: list[ParsedQuestion],
        progress_callback: Callable[[JobStatus, int, str], None],
    ) -> list[tuple[ParsedQuestion, EvaluationResult]]:
        """Evaluates questions sequentially while reporting granular progress."""
        results: list[tuple[ParsedQuestion, EvaluationResult]] = []
        total = len(questions)

        for idx, q in enumerate(questions, 1):
            if q.error:
                raise DocumentParsingError(f"Cannot evaluate Q{q.q_num:02d}: {q.error}")
            progress_callback(
                JobStatus.EVALUATING,
                int(30 + 60 * (idx - 1) / total),
                f"Evaluating Question {q.q_num} of {total}...",
            )
            eval_res = self.engine.evaluate_answer(
                question=q.question,
                candidate_answer=q.candidate_answer,
                max_marks=q.max_marks,
            )
            results.append((q, eval_res))

        return results

    def _evaluate_concurrently(
        self,
        questions: list[ParsedQuestion],
        progress_callback: Callable[[JobStatus, int, str], None],
    ) -> list[tuple[ParsedQuestion, EvaluationResult]]:
        """Evaluates questions in parallel worker threads."""
        for q in questions:
            if q.error:
                raise DocumentParsingError(f"Cannot evaluate Q{q.q_num:02d}: {q.error}")

        results: list[tuple[ParsedQuestion, EvaluationResult]] = []
        total = len(questions)
        completed = 0

        with ThreadPoolExecutor(max_workers=min(self.workers, total)) as executor:
            future_to_q = {
                executor.submit(
                    self.engine.evaluate_answer,
                    q.question,
                    q.candidate_answer,
                    q.max_marks,
                ): q
                for q in questions
            }

            for future in as_completed(future_to_q):
                q = future_to_q[future]
                eval_res = future.result()
                results.append((q, eval_res))
                completed += 1
                progress_callback(
                    JobStatus.EVALUATING,
                    int(30 + 60 * completed / total),
                    f"Completed Question {q.q_num} ({completed}/{total})...",
                )

        return results

    @staticmethod
    @traceable(name="Distill_Student_Scorecard", run_type="chain")
    def distill_results(
        document_name: str,
        evaluations: list[tuple[ParsedQuestion, EvaluationResult]],
    ) -> StudentEvaluationReport:
        """
        Distills raw EvaluationResults into the clean, student-facing StudentEvaluationReport.
        Strictly purges all performance bands, G-Eval logprobs, and internal CoT reasoning trails.
        """
        student_questions: list[StudentQuestionEvaluation] = []
        all_strengths: list[str] = []
        all_improvements: list[str] = []

        for q, res in evaluations:
            score = round(res.total_score, 2)
            pct = round((score / q.max_marks) * 100, 1) if q.max_marks > 0 else 0.0

            # Pros: extract clean strengths
            pros = [s.strip() for s in res.strengths if s and s.strip()]

            # What to do better: combine topper action plan and critical weaknesses without duplicates
            what_to_do_better: list[str] = []
            seen: set[str] = set()

            for item in (res.topper_action_plan or []) + (res.weaknesses or []):
                cleaned = item.strip()
                if cleaned and cleaned.lower() not in seen:
                    what_to_do_better.append(cleaned)
                    seen.add(cleaned.lower())

            # Fallback if question was unattempted or empty
            if q.is_blank or not q.candidate_answer.strip():
                score = 0.0
                pct = 0.0
                pros = []
                what_to_do_better = [
                    "Attempt all questions under timed exam conditions to capture essential step marks.",
                    "Structure points with explicit subheadings, relevant Constitutional Articles, and Committee reports.",
                ]

            student_q = StudentQuestionEvaluation(
                q_num=q.q_num,
                max_marks=float(q.max_marks),
                score=score,
                percentage=pct,
                question=q.question,
                pros=pros,
                what_to_do_better=what_to_do_better,
            )
            student_questions.append(student_q)

            if pros:
                all_strengths.extend(pros)
            if what_to_do_better:
                all_improvements.extend(what_to_do_better)

        # Compute paper-level summary totals
        total_score = round(sum(sq.score for sq in student_questions), 2)
        total_max_marks = round(sum(sq.max_marks for sq in student_questions), 2)
        overall_pct = (
            round((total_score / total_max_marks) * 100, 1) if total_max_marks > 0 else 0.0
        )

        # Synthesize top 3 overall strengths and top 3 priority areas to improve
        key_strengths = _pick_top_distinct(all_strengths, limit=3)
        if not key_strengths:
            key_strengths = ["Structured attempt adhering to standard UPSC QCAB formatting."]

        top_areas_to_improve = _pick_top_distinct(all_improvements, limit=3)
        if not top_areas_to_improve:
            top_areas_to_improve = [
                "Ground recommendations in specific government commissions (2nd ARC, Punchhi Commission).",
                "Ensure balanced coverage across constitutional, administrative, and practical dimensions.",
            ]

        summary = StudentSummary(
            total_score=total_score,
            max_marks=total_max_marks,
            percentage=overall_pct,
            overall_feedback=OverallFeedback(
                key_strengths=key_strengths,
                top_areas_to_improve=top_areas_to_improve,
            ),
        )

        return StudentEvaluationReport(
            status="success",
            document=document_name,
            summary=summary,
            questions=student_questions,
        )


def _pick_top_distinct(items: list[str], limit: int = 3) -> list[str]:
    """Selects up to `limit` distinct, high-quality bullet points."""
    distinct: list[str] = []
    seen: set[str] = set()

    for item in items:
        cleaned = item.strip()
        # Filter out trivial or unattempted notes for overall paper feedback
        if not cleaned or "unattempted" in cleaned.lower() or len(cleaned) < 15:
            continue
        key = cleaned.lower()[:40]
        if key not in seen:
            distinct.append(cleaned)
            seen.add(key)
        if len(distinct) >= limit:
            break

    return distinct
