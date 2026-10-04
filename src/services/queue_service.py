"""Pluggable job queue service supporting Amazon SQS and local thread fallback.

Supports the decoupled two-stage architecture with separate Vision and Evaluation queues.
"""
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


class QueueService:
    """Manages job dispatching, abstracting Amazon SQS vs local background thread."""

    def __init__(
        self,
        vision_queue_url: str | None = None,
        eval_queue_url: str | None = None,
        region_name: str | None = None,
        queue_url: str | None = None,
    ):
        if queue_url is not None and vision_queue_url is None:
            vision_queue_url = queue_url
        self.vision_queue_url = vision_queue_url if vision_queue_url is not None else settings.vision_queue_url
        self.eval_queue_url = eval_queue_url if eval_queue_url is not None else settings.eval_queue_url
        self.region_name = region_name or settings.aws_region
        self._sqs_client: Any = None

    @property
    def queue_url(self) -> str | None:
        """Alias for vision_queue_url for backward compatibility."""
        return self.vision_queue_url

    @property
    def is_sqs_enabled(self) -> bool:
        """Returns True if at least one SQS queue URL is configured."""
        return bool((self.vision_queue_url and self.vision_queue_url.strip())
                     or (self.eval_queue_url and self.eval_queue_url.strip()))

    @property
    def sqs_client(self) -> Any:
        """Lazy instantiation of boto3 SQS client."""
        if self._sqs_client is None:
            import boto3

            self._sqs_client = boto3.client("sqs", region_name=self.region_name)
        return self._sqs_client

    def dispatch(
        self,
        job_id: str,
        pdf_path: str | Path,
        email: str | None = None,
        start_page: int | None = None,
        max_pages: int | None = None,
        local_thread_runner: Callable[..., None] | None = None,
    ) -> str:
        """Alias for dispatch_vision for backward compatibility."""
        return self.dispatch_vision(
            job_id=job_id,
            pdf_path=pdf_path,
            email=email,
            start_page=start_page,
            max_pages=max_pages,
            local_thread_runner=local_thread_runner,
        )

    def dispatch_vision(
        self,
        job_id: str,
        pdf_path: str | Path,
        email: str | None = None,
        start_page: int | None = None,
        max_pages: int | None = None,
        local_thread_runner: Callable[..., None] | None = None,
    ) -> str:
        """
        Dispatches a Vision OCR job.
        - If VisionQueue SQS is configured, pushes message to SQS and returns 'sqs'.
        - Otherwise, invokes local_thread_runner in a daemon thread and returns 'thread'.
        """
        payload = {
            "job_id": job_id,
            "pdf_path": str(pdf_path),
            "email": email,
            "start_page": start_page,
            "max_pages": max_pages,
        }

        target_queue = self.vision_queue_url or self.eval_queue_url or getattr(settings, "queue_url", None)
        if target_queue and target_queue.strip():
            logger.info("Dispatching job %s to SQS: %s", job_id, target_queue)
            self.sqs_client.send_message(
                QueueUrl=target_queue,
                MessageBody=json.dumps(payload),
            )
            return "sqs"

        if local_thread_runner:
            import threading

            thread = threading.Thread(
                target=local_thread_runner,
                args=(job_id, Path(str(pdf_path)), start_page, max_pages),
                daemon=True,
                name=f"Worker-{job_id}",
            )
            thread.start()
            logger.info("Dispatched background thread %s for %s", thread.name, job_id)
            return "thread"

        raise RuntimeError(f"Cannot dispatch job {job_id}: SQS disabled and no local runner.")

    def dispatch_eval(
        self,
        job_id: str,
        email: str | None = None,
    ) -> str:
        """
        Dispatches an evaluation job (assumes parsed data is in DynamoDB).
        Only used in the decoupled SQS mode.
        """
        if not (self.eval_queue_url and self.eval_queue_url.strip()):
            logger.debug("EvalQueue not configured; skipping SQS dispatch for %s.", job_id)
            return "local"

        payload = {
            "job_id": job_id,
            "email": email,
        }
        logger.info("Dispatching eval job %s to SQS: %s", job_id, self.eval_queue_url)
        self.sqs_client.send_message(
            QueueUrl=self.eval_queue_url,
            MessageBody=json.dumps(payload),
        )
        return "sqs"
