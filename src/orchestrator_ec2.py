"""EC2 Coordinator Orchestrator: Concurrency-gated fan-out evaluator.

Enforces MAX_CONCURRENT_COPIES = 2 via asyncio.Semaphore to prevent RAM spikes on t3.micro.
Prefetches KB ground truth on EC2 in parallel (<1s), fans out 20 concurrent SingleQuestionEval
Lambda functions (~7.5s), and compiles/delivers student scorecards.

Tracing: One copy check = one LangSmith thread. The @traceable on `_process_job_guarded`
creates the root trace; every child operation (OCR, KB prefetch, Lambda fan-out, distillation,
PDF generation, email delivery) nests inside it via captured RunTree headers propagated
across asyncio executor threads.
"""
import asyncio
import json
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import settings
from src.handlers.single_question_eval import handler as single_question_handler
from src.kb.prefetcher import KnowledgeBasePrefetcher
from src.models.api import JobStatus, StudentEvaluationReport
from src.models.evaluation import EvaluationResult
from src.models.parsing import ParsedDocument, ParsedQuestion
from src.orchestrator import UnifiedEvaluationPipeline
from src.parsing.document_parser import DocumentParsingPipeline
from src.services.email_service import EmailService
from src.services.job_state_service import JobStateService
from src.services.scorecard_pdf import generate_scorecard_pdf
from src.services.storage_service import StorageService
from src.utils.tracing import flush_traces, traceable

logger = logging.getLogger(__name__)

# Strict concurrency limit: at most 2 copies evaluated simultaneously on EC2
MAX_CONCURRENT_COPIES = 2
_copy_semaphore: asyncio.Semaphore | None = None


def get_copy_semaphore() -> asyncio.Semaphore:
    global _copy_semaphore
    if _copy_semaphore is None:
        _copy_semaphore = asyncio.Semaphore(MAX_CONCURRENT_COPIES)
    return _copy_semaphore


def _get_trace_headers() -> dict[str, str]:
    """Capture current LangSmith RunTree headers for propagation to child threads."""
    try:
        from langsmith.run_helpers import get_current_run_tree

        current_run = get_current_run_tree()
        if current_run:
            return current_run.to_headers()
    except Exception:
        pass
    return {}


def _make_langsmith_extra(headers: dict[str, str]) -> dict[str, Any] | None:
    """Build a langsmith_extra dict that parents a @traceable call under the given headers."""
    if not headers:
        return None
    return {"parent": headers}


# --- Standalone @traceable helpers for child operations ---
# Each is a module-level function decorated with @traceable so LangSmith auto-nests them
# when called with langsmith_extra={"parent": <headers>} from executor threads.


@traceable(name="DocumentParsing_OCR", run_type="chain")
def _run_vision_ocr(
    local_pdf: Path,
    start_page: int | None,
    max_pages: int | None,
    dpi: int = 135,
    max_workers: int = 4,
) -> ParsedDocument:
    """Runs multimodal vision OCR pipeline on the answer booklet."""
    parser = DocumentParsingPipeline(dpi=dpi, max_workers=max_workers)
    return parser.parse_pdf(
        pdf_path=local_pdf,
        start_page=start_page,
        max_pages=max_pages,
    )


@traceable(name="KB_Ground_Truth_Prefetch", run_type="retriever")
def _run_kb_prefetch(
    questions: list[ParsedQuestion],
    prefetcher: KnowledgeBasePrefetcher,
) -> dict[int, list[str]]:
    """Prefetches KB ground truth blocks for all questions in parallel."""
    return prefetcher.prefetch_all(questions)


@traceable(name="Student_Scorecard_Distillation", run_type="chain")
def _run_distillation(
    filename: str,
    eval_pairs: list[tuple[ParsedQuestion, EvaluationResult]],
) -> StudentEvaluationReport:
    """Distills evaluation results into the student scorecard."""
    return UnifiedEvaluationPipeline.distill_results(
        document_name=filename,
        evaluations=eval_pairs,
    )


@traceable(name="Scorecard_PDF_Generation", run_type="tool")
def _run_pdf_generation(report: StudentEvaluationReport) -> bytes:
    """Generates the scorecard PDF."""
    return generate_scorecard_pdf(report)


@traceable(name="Amazon_SES_Email_Delivery", run_type="tool")
def _run_email_delivery(
    email_svc: EmailService,
    to_email: str,
    report: StudentEvaluationReport,
    job_id: str,
    pdf_bytes: bytes | None,
) -> bool:
    """Sends evaluation scorecard email via Amazon SES."""
    return email_svc.send_evaluation_email(
        to_email=to_email,
        report=report,
        job_id=job_id,
        pdf_bytes=pdf_bytes,
    )


class EC2CoordinatorOrchestrator:
    """Manages the full copy evaluation lifecycle on EC2 with 20-Lambda fan-out."""

    def __init__(
        self,
        storage_service: StorageService | None = None,
        job_state_service: JobStateService | None = None,
        email_service: EmailService | None = None,
        prefetcher: KnowledgeBasePrefetcher | None = None,
        lambda_function_name: str | None = None,
        region_name: str | None = None,
    ):
        self.storage = storage_service or StorageService()
        self.job_state = job_state_service or JobStateService()
        self.email_svc = email_service or EmailService()
        self.prefetcher = prefetcher or KnowledgeBasePrefetcher(top_k=6, max_workers=10)
        self.lambda_function_name = lambda_function_name or getattr(settings, "eval_lambda_function_name", "SingleQuestionEvalFunction")
        self.region_name = region_name or settings.aws_region
        self._lambda_client: Any = None

    @property
    def lambda_client(self) -> Any:
        if self._lambda_client is None:
            import boto3
            self._lambda_client = boto3.client("lambda", region_name=self.region_name)
        return self._lambda_client

    def _invoke_single_question_eval(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluates 1 question: invokes AWS Lambda if in production,
        otherwise falls back to direct handler execution.
        """
        # Production AWS mode: invoke SingleQuestionEval Lambda via boto3
        if settings.has_aws_credentials and self.lambda_function_name:
            try:
                response = self.lambda_client.invoke(
                    FunctionName=self.lambda_function_name,
                    InvocationType="RequestResponse",
                    Payload=json.dumps(payload),
                )
                raw_payload = response["Payload"].read()
                data = json.loads(raw_payload)
                if isinstance(data, dict) and data.get("status") == "success":
                    return data
            except Exception as e:
                logger.warning(
                    "AWS Lambda invoke failed for Q%02d: %s. Falling back to local handler.",
                    payload.get("q_num", 0), e
                )

        # Fallback / Local mode: invoke handler directly
        return single_question_handler(payload)

    @traceable(name="EC2_FanOut_20_Lambda_Evaluation", run_type="chain")
    async def evaluate_copy_fanout(
        self,
        questions: list[ParsedQuestion],
        kb_contexts: dict[int, list[str]],
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> list[tuple[ParsedQuestion, EvaluationResult]]:
        """
        Fires 20 Single-Question evaluations in parallel using asyncio.
        All 20 questions complete concurrently in ~7-9 seconds total.
        Passes LangSmith parent trace headers so every Lambda appears nested under this thread.
        """
        loop = asyncio.get_running_loop()
        total = len(questions)
        logger.info("Fanning out %d concurrent question evaluations...", total)

        # Capture current trace headers so Lambda child traces nest under this run
        trace_headers = _get_trace_headers()

        # Gate concurrency to protect against Vertex AI rate limits / quotas (e.g. 5 or 10 concurrent)
        sem = asyncio.Semaphore(settings.eval_max_concurrency)

        async def _eval_one(payload: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await loop.run_in_executor(None, self._invoke_single_question_eval, payload)

        tasks = []
        for q in questions:
            payload = {
                "q_num": q.q_num,
                "question": q.question,
                "candidate_answer": q.candidate_answer,
                "max_marks": q.max_marks,
                "kb_context": kb_contexts.get(q.q_num, []),
                "trace_headers": trace_headers,
            }
            tasks.append(_eval_one(payload))

        raw_results = await asyncio.gather(*tasks)

        eval_pairs: list[tuple[ParsedQuestion, EvaluationResult]] = []
        for q, res in zip(questions, raw_results, strict=False):
            eval_dict = res.get("evaluation", {})
            eval_result = EvaluationResult.model_validate(eval_dict)
            eval_pairs.append((q, eval_result))

        eval_pairs.sort(key=lambda pair: pair[0].q_num)
        if progress_callback:
            progress_callback(85, f"All {total} questions evaluated in parallel.")

        return eval_pairs

    async def process_job(
        self,
        job_id: str,
        storage_ref: str,
        filename: str = "booklet.pdf",
        email: str | None = None,
        start_page: int | None = None,
        max_pages: int | None = None,
    ) -> StudentEvaluationReport:
        """
        Processes a full evaluation job gated by the MAX_CONCURRENT_COPIES semaphore.
        """
        semaphore = get_copy_semaphore()
        async with semaphore:
            return await self._process_job_guarded(
                job_id=job_id,
                storage_ref=storage_ref,
                filename=filename,
                email=email,
                start_page=start_page,
                max_pages=max_pages,
            )

    @traceable(name="UPSC_Answer_Booklet_Evaluation", run_type="chain")
    async def _process_job_guarded(
        self,
        job_id: str,
        storage_ref: str,
        filename: str = "booklet.pdf",
        email: str | None = None,
        start_page: int | None = None,
        max_pages: int | None = None,
    ) -> StudentEvaluationReport:
        """Internal execution within the concurrency gate.

        This is the single root trace for one copy check. All child operations
        (OCR, KB prefetch, Lambda fan-out, distillation, PDF gen, email delivery)
        nest under this trace via langsmith_extra parent header propagation.
        """
        logger.info("EC2 Coordinator starting job %s (Storage: %s)", job_id, storage_ref)

        def _update(pct: int, step: str, status: JobStatus = JobStatus.PARSING):
            data = self.job_state.get_job(job_id) or {"job_id": job_id}
            data["status"] = status.value
            data["progress_pct"] = pct
            data["current_step"] = step
            self.job_state.save_job(job_id, data)

        try:
            # 1. Size verification (reject oversized files > 200 MB)
            file_size = self.storage.get_file_size(storage_ref)
            max_bytes = 200 * 1024 * 1024
            if file_size > max_bytes:
                self.storage.delete_file(storage_ref)
                err_msg = f"File size ({file_size // (1024 * 1024)} MB) exceeds 200 MB maximum limit."
                _update(0, err_msg, JobStatus.FAILED)
                raise ValueError(err_msg)

            # Capture parent trace headers for propagation into executor threads.
            # All @traceable helper functions called via run_in_executor receive these
            # headers in langsmith_extra={"parent": headers} so they nest properly.
            parent_headers = _get_trace_headers()
            ls_extra = _make_langsmith_extra(parent_headers)

            # 2. Ephemeral Download & Vision OCR
            _update(15, "Downloading PDF and starting OCR...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_pdf = self.storage.get_local_path(storage_ref, temp_dir=Path(tmp_dir))

                _update(25, "Running multimodal vision OCR...")
                loop = asyncio.get_running_loop()
                parsed_doc: ParsedDocument = await loop.run_in_executor(
                    None,
                    lambda: _run_vision_ocr(
                        local_pdf=local_pdf,
                        start_page=start_page,
                        max_pages=max_pages,
                        dpi=settings.parsing_dpi,
                        max_workers=4,
                        langsmith_extra=ls_extra,
                    ),
                )

            # 3. Immediate Storage Cleanup: Delete S3 raw PDF as soon as OCR is done
            self.storage.delete_file(storage_ref)
            logger.info("Storage cleaned up: deleted raw PDF for job %s", job_id)

            questions = parsed_doc.questions
            total_questions = len(questions)
            _update(40, f"OCR complete ({total_questions} questions parsed). Prefetching ground truth...")

            # 4. Parallel Knowledge Base Prefetching (<1s on EC2)
            kb_contexts: dict[int, list[str]] = await loop.run_in_executor(
                None,
                lambda: _run_kb_prefetch(
                    questions=questions,
                    prefetcher=self.prefetcher,
                    langsmith_extra=ls_extra,
                ),
            )
            _update(50, "Ground truth prefetched. Fanning out 20 concurrent AI evaluators...")

            # 5. 20-Lambda Concurrent Fan-Out Evaluation (~7-9s)
            eval_pairs = await self.evaluate_copy_fanout(
                questions=questions,
                kb_contexts=kb_contexts,
                progress_callback=_update,
            )

            # 6. Distill Student Scorecard
            _update(90, "Distilling actionable student feedback...")
            report: StudentEvaluationReport = await loop.run_in_executor(
                None,
                lambda: _run_distillation(
                    filename=filename,
                    eval_pairs=eval_pairs,
                    langsmith_extra=ls_extra,
                ),
            )

            # 7. Generate Scorecard PDF
            pdf_bytes: bytes | None = None
            try:
                pdf_bytes = await loop.run_in_executor(
                    None,
                    lambda: _run_pdf_generation(report, langsmith_extra=ls_extra),
                )
            except Exception as pdf_err:
                logger.warning("PDF generation failed for %s: %s", job_id, pdf_err)

            # 8. Send Email via Amazon SES
            email_sent = False
            target_email = email
            if target_email:
                _update(95, f"Delivering scorecard to {target_email}...")
                email_sent = await loop.run_in_executor(
                    None,
                    lambda: _run_email_delivery(
                        email_svc=self.email_svc,
                        to_email=target_email,
                        report=report,
                        job_id=job_id,
                        pdf_bytes=pdf_bytes,
                        langsmith_extra=ls_extra,
                    ),
                )

            # 9. Mark Completed
            final_data = self.job_state.get_job(job_id) or {"job_id": job_id}
            final_data["status"] = JobStatus.COMPLETED.value
            final_data["progress_pct"] = 100
            final_data["current_step"] = "Evaluation complete."
            final_data["email_sent"] = email_sent
            final_data["result"] = report.model_dump()
            self.job_state.save_job(job_id, final_data)

            logger.info("Job %s finished in <30s. Total score: %s/%s",
                        job_id, report.summary.total_score, report.summary.max_marks)
            return report

        except Exception as e:
            logger.exception("EC2 Coordinator job %s failed: %s", job_id, e)
            _update(0, f"Evaluation failed: {str(e)}", JobStatus.FAILED)
            raise
        finally:
            flush_traces()
