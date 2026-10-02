"""AWS Lambda worker entrypoint for background UPSC answer copy evaluation triggered by SQS."""
import json
import logging
from typing import Any

from src.services.job_manager import job_manager

logger = logging.getLogger(__name__)


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """
    AWS Lambda entrypoint triggered by Amazon SQS or direct test payload.
    Processes evaluation jobs, running the full OCR, retrieval, and 2-call LLM pipeline.
    """
    logger.info("Lambda worker invoked with event keys: %s", list(event.keys()))

    # Case 1: SQS Event Source Mapping
    if "Records" in event:
        records = event["Records"]
        logger.info("Processing %d SQS record(s)", len(records))
        batch_item_failures = []

        for record in records:
            message_id = record.get("messageId", "unknown")
            try:
                body = record.get("body", "{}")
                payload = json.loads(body) if isinstance(body, str) else body

                job_id = payload["job_id"]
                pdf_path = payload["pdf_path"]
                start_page = payload.get("start_page")
                max_pages = payload.get("max_pages")

                logger.info("Starting processing for SQS job %s (message %s)", job_id, message_id)
                job_manager._run_job_worker(
                    job_id=job_id,
                    pdf_path=pdf_path,
                    start_page=start_page,
                    max_pages=max_pages,
                )
                logger.info("Completed processing for SQS job %s", job_id)

            except Exception as e:
                logger.exception("Failed processing SQS message %s: %s", message_id, e)
                # Report failure for SQS partial batch retry
                batch_item_failures.append({"itemIdentifier": message_id})

        return {"batchItemFailures": batch_item_failures}

    # Case 2: Direct Invocation (e.g., CLI test or Step Functions)
    job_id = event.get("job_id")
    pdf_path = event.get("pdf_path")
    if job_id and pdf_path:
        start_page = event.get("start_page")
        max_pages = event.get("max_pages")
        logger.info("Direct invocation for job %s", job_id)
        job_manager._run_job_worker(
            job_id=job_id,
            pdf_path=pdf_path,
            start_page=start_page,
            max_pages=max_pages,
        )
        return {"status": "success", "job_id": job_id}

    logger.warning("Unrecognized event format: %s", event)
    return {"status": "ignored", "reason": "No valid Records or job_id payload found"}
