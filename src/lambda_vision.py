"""AWS Lambda Vision Worker: Parses PDF pages via multimodal Vision OCR, stores parsed JSON in DynamoDB, and dispatches eval stage.

Triggered by the VisionQueue SQS. Downloads the PDF from S3, runs the document parsing pipeline
(page slicing, handwriting transcription, question segmentation), then stores the parsed
document JSON into DynamoDB and pushes a message to the EvalQueue for the next stage.
"""
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from src.models.api import JobStatus
from src.parsing.pipeline import DocumentParsingPipeline
from src.services.job_state_service import JobStateService
from src.services.queue_service import QueueService
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """
    AWS Lambda entrypoint triggered by VisionQueue SQS.
    For each record: downloads PDF from S3, runs vision OCR, stores parsed JSON in DynamoDB,
    and pushes eval job to EvalQueue.
    """
    logger.info("Vision Lambda invoked with event keys: %s", list(event.keys()))

    storage = StorageService()
    state = JobStateService()
    queue = QueueService()

    if "Records" not in event:
        # Direct invocation fallback
        return _process_single(event, storage, state, queue)

    records = event["Records"]
    logger.info("Processing %d SQS vision record(s)", len(records))
    batch_item_failures = []

    for record in records:
        message_id = record.get("messageId", "unknown")
        try:
            body = record.get("body", "{}")
            payload = json.loads(body) if isinstance(body, str) else body
            _process_single(payload, storage, state, queue)
        except Exception as e:
            logger.exception("Vision worker failed for SQS message %s: %s", message_id, e)
            batch_item_failures.append({"itemIdentifier": message_id})

    return {"batchItemFailures": batch_item_failures}


def _process_single(
    payload: dict[str, Any],
    storage: StorageService,
    state: JobStateService,
    queue: QueueService,
) -> dict[str, Any]:
    """Processes a single vision OCR job."""
    job_id = payload["job_id"]
    pdf_ref = payload["pdf_path"]
    email = payload.get("email")
    start_page = payload.get("start_page")
    max_pages = payload.get("max_pages")

    logger.info("Vision worker starting job %s (PDF: %s)", job_id, pdf_ref)

    # Update state: PARSING
    _update_state(state, job_id, JobStatus.PARSING, 10, "Downloading PDF and starting Vision OCR...")

    # Download PDF from S3 to /tmp
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_pdf = storage.get_local_path(pdf_ref, temp_dir=Path(tmp_dir))
        logger.info("Downloaded PDF for %s to %s", job_id, local_pdf)

        # Run vision parsing pipeline
        _update_state(state, job_id, JobStatus.PARSING, 20, "Running multimodal vision OCR on answer pages...")

        parser = DocumentParsingPipeline(
            max_workers=2,  # Lambda has limited vCPU
        )
        parsed_doc = parser.parse_pdf(
            pdf_path=local_pdf,
            start_page=start_page,
            max_pages=max_pages,
        )

    total_questions = len(parsed_doc.questions)
    logger.info("Vision OCR complete for %s: %d questions parsed.", job_id, total_questions)

    # Store parsed document JSON into DynamoDB job state
    parsed_data = parsed_doc.model_dump()
    _update_state(
        state, job_id, JobStatus.EVALUATING, 35,
        f"Vision OCR complete. {total_questions} questions parsed. Dispatching evaluation...",
        extra={"parsed_document": parsed_data},
    )

    # Dispatch to EvalQueue
    queue.dispatch_eval(job_id=job_id, email=email)
    logger.info("Dispatched eval job for %s to EvalQueue.", job_id)

    return {"status": "success", "job_id": job_id, "questions_parsed": total_questions}


def _update_state(
    state: JobStateService,
    job_id: str,
    status: JobStatus,
    pct: int,
    step: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Helper to update job state in DynamoDB."""
    import datetime

    data = state.get_job(job_id) or {"job_id": job_id}
    data["status"] = status.value
    data["progress_pct"] = pct
    data["current_step"] = step
    data["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    if extra:
        data.update(extra)
    state.save_job(job_id, data)
