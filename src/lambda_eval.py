"""AWS Lambda Evaluation Worker: Reads parsed document from DynamoDB, runs UPSC evaluation, generates PDF scorecard, sends email.

Triggered by the EvalQueue SQS. Reads parsed document JSON from DynamoDB (stored by the Vision
Lambda), runs the 2-call CoT + G-Eval evaluation engine against the KB, generates a scorecard
PDF using PyMuPDF, and dispatches the results via Amazon SES with the PDF attached.
"""
import datetime
import json
import logging
from typing import Any

from src.evaluation.engine import UPSCEvaluationEngine
from src.models.api import JobStatus
from src.models.evaluation import EvaluationResult
from src.models.parsing import ParsedDocument, ParsedQuestion
from src.pipeline import UnifiedEvaluationPipeline
from src.services.email_service import EmailService
from src.services.job_state_service import JobStateService
from src.services.scorecard_pdf import generate_scorecard_pdf
from src.utils.tracing import flush_traces, traceable

logger = logging.getLogger(__name__)


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """
    AWS Lambda entrypoint triggered by EvalQueue SQS.
    For each record: reads parsed document from DynamoDB, evaluates, generates PDF, sends email.
    """
    logger.info("Eval Lambda invoked with event keys: %s", list(event.keys()))

    state = JobStateService()

    try:
        if "Records" not in event:
            return _process_single(event, state)

        records = event["Records"]
        logger.info("Processing %d SQS eval record(s)", len(records))
        batch_item_failures = []

        for record in records:
            message_id = record.get("messageId", "unknown")
            try:
                body = record.get("body", "{}")
                payload = json.loads(body) if isinstance(body, str) else body
                _process_single(payload, state)
            except Exception as e:
                logger.exception("Eval worker failed for SQS message %s: %s", message_id, e)
                batch_item_failures.append({"itemIdentifier": message_id})

        return {"batchItemFailures": batch_item_failures}
    finally:
        flush_traces()


@traceable(name="LambdaEvalWorker.process_job", run_type="chain")
def _process_single(
    payload: dict[str, Any],
    state: JobStateService,
) -> dict[str, Any]:
    """Processes a single evaluation job."""
    job_id = payload["job_id"]
    email = payload.get("email")

    logger.info("Eval worker starting job %s", job_id)

    # Load job state from DynamoDB (contains parsed_document from Vision Lambda)
    job_data = state.get_job(job_id)
    if not job_data:
        raise RuntimeError(f"Job {job_id} not found in state store.")

    parsed_data = job_data.get("parsed_document")
    if not parsed_data:
        raise RuntimeError(f"Job {job_id} has no parsed_document in state. Vision stage may have failed.")

    # Reconstruct ParsedDocument from stored JSON
    parsed_doc = ParsedDocument.model_validate(parsed_data)
    questions = parsed_doc.questions
    total_questions = len(questions)
    doc_name = getattr(parsed_doc, "source_file", None) or getattr(parsed_doc, "source_filename", None) or f"job_{job_id}"

    logger.info("Loaded parsed document for %s: %d questions", job_id, total_questions)

    # Initialize evaluation engine
    _update_state(state, job_id, JobStatus.EVALUATING, 40, f"Evaluating {total_questions} questions against KB...")

    engine = UPSCEvaluationEngine()

    # Evaluate each question
    eval_pairs: list[tuple[ParsedQuestion, EvaluationResult]] = []
    for idx, q in enumerate(questions, 1):
        pct = int(40 + 45 * idx / total_questions)
        _update_state(state, job_id, JobStatus.EVALUATING, pct, f"Evaluating Q{q.q_num} ({idx}/{total_questions})...")

        if q.error and not (q.candidate_answer and q.candidate_answer.strip()):
            logger.warning("Q%02d had transcription error: %s. Evaluating as unreadable.", q.q_num, q.error)
            result = engine.evaluate_answer(
                question=q.question or f"Question {q.q_num}",
                candidate_answer="",
                max_marks=q.max_marks,
            )
        else:
            result = engine.evaluate_answer(
                question=q.question,
                candidate_answer=q.candidate_answer or "",
                max_marks=q.max_marks,
            )
        eval_pairs.append((q, result))

    eval_pairs.sort(key=lambda pair: pair[0].q_num)

    # Distill into student scorecard
    _update_state(state, job_id, JobStatus.DISTILLING, 90, "Distilling actionable feedback...")
    report = UnifiedEvaluationPipeline.distill_results(
        document_name=doc_name,
        evaluations=eval_pairs,
    )

    # Generate PDF scorecard
    pdf_bytes: bytes | None = None
    try:
        pdf_bytes = generate_scorecard_pdf(report)
        logger.info("Generated scorecard PDF for %s: %d bytes", job_id, len(pdf_bytes))
    except Exception as pdf_err:
        logger.warning("PDF generation failed for %s: %s", job_id, pdf_err)

    # Send email with PDF attachment
    email_sent = False
    target_email = email or job_data.get("email")
    if target_email:
        _update_state(state, job_id, JobStatus.SENDING_EMAIL, 95, f"Sending scorecard to {target_email}...")
        email_svc = EmailService()
        email_sent = email_svc.send_evaluation_email(
            to_email=target_email,
            report=report,
            job_id=job_id,
            pdf_bytes=pdf_bytes,
        )

    # Mark completed
    final_data = state.get_job(job_id) or {"job_id": job_id}
    final_data["status"] = JobStatus.COMPLETED.value
    final_data["progress_pct"] = 100
    final_data["current_step"] = "Evaluation complete."
    final_data["email_sent"] = email_sent
    final_data["result"] = report.model_dump()
    # Remove large parsed_document blob to keep DynamoDB item lean
    final_data.pop("parsed_document", None)
    final_data["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    state.save_job(job_id, final_data)

    logger.info("Job %s completed. Score: %s/%s. Email sent: %s",
                job_id, report.summary.total_score, report.summary.max_marks, email_sent)

    return {"status": "success", "job_id": job_id, "email_sent": email_sent}


def _update_state(
    state: JobStateService,
    job_id: str,
    status: JobStatus,
    pct: int,
    step: str,
) -> None:
    """Helper to update job state in DynamoDB."""
    data = state.get_job(job_id) or {"job_id": job_id}
    data["status"] = status.value
    data["progress_pct"] = pct
    data["current_step"] = step
    data["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    state.save_job(job_id, data)
