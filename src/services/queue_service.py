"""Pluggable job queue service supporting Amazon SQS and local thread fallback."""
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
        queue_url: str | None = None,
        region_name: str | None = None,
    ):
        self.queue_url = queue_url if queue_url is not None else settings.sqs_queue_url
        self.region_name = region_name or settings.aws_region
        self._sqs_client: Any = None

    @property
    def is_sqs_enabled(self) -> bool:
        """Returns True if Amazon SQS queue URL is configured."""
        return bool(self.queue_url and self.queue_url.strip())

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
        start_page: int | None = None,
        max_pages: int | None = None,
        local_thread_runner: Callable[[str, Path, int | None, int | None], None] | None = None,
    ) -> str:
        """
        Dispatches a job for evaluation.
        - If SQS is enabled, pushes a JSON message to SQS and returns 'sqs'.
        - Otherwise, invokes local_thread_runner in a daemon thread and returns 'thread'.
        """
        pdf_path_str = str(pdf_path)

        if self.is_sqs_enabled:
            payload = {
                "job_id": job_id,
                "pdf_path": pdf_path_str,
                "start_page": start_page,
                "max_pages": max_pages,
            }
            logger.info("Dispatching job %s to SQS queue %s", job_id, self.queue_url)
            self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(payload),
            )
            return "sqs"

        if local_thread_runner:
            import threading

            thread = threading.Thread(
                target=local_thread_runner,
                args=(job_id, Path(pdf_path_str), start_page, max_pages),
                daemon=True,
                name=f"Worker-{job_id}",
            )
            thread.start()
            logger.info("Dispatched background thread %s for %s", thread.name, job_id)
            return "thread"

        raise RuntimeError(f"Cannot dispatch job {job_id}: SQS is disabled and no local runner provided.")
