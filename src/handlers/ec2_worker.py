"""EC2 Worker Daemon: Continuous SQS consumer with concurrency gating and 20-Lambda fan-out.

Polls the Evaluation/Job SQS queue, processes UPSC answer copies using EC2CoordinatorOrchestrator
with MAX_CONCURRENT_COPIES = 2 guardrail, and delivers evaluated scorecards.
"""
import asyncio
import json
import logging
import signal
import sys
from typing import Any

from src.config import settings
from src.orchestrator_ec2 import EC2CoordinatorOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ec2_worker")


class EC2WorkerDaemon:
    """Long-polling SQS worker executing evaluation jobs on EC2."""

    def __init__(
        self,
        queue_url: str | None = None,
        region_name: str | None = None,
        orchestrator: EC2CoordinatorOrchestrator | None = None,
    ):
        self.queue_url = queue_url or settings.eval_queue_url or settings.queue_url
        self.region_name = region_name or settings.aws_region
        self.orchestrator = orchestrator or EC2CoordinatorOrchestrator()
        self._running = False
        self._sqs: Any = None

    @property
    def sqs(self) -> Any:
        if self._sqs is None:
            import boto3
            self._sqs = boto3.client("sqs", region_name=self.region_name)
        return self._sqs

    async def run(self) -> None:
        """Main consumer loop."""
        if not self.queue_url:
            logger.error("No SQS queue URL configured (EVAL_QUEUE_URL or QUEUE_URL). Exiting.")
            sys.exit(1)

        self._running = True
        logger.info("EC2 Worker started. Polling SQS: %s (Region: %s)", self.queue_url, self.region_name)

        while self._running:
            try:
                # Long poll SQS (WaitTimeSeconds=20)
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.sqs.receive_message(
                        QueueUrl=self.queue_url,
                        MaxNumberOfMessages=1,
                        WaitTimeSeconds=20,
                        VisibilityTimeout=300,
                    ),
                )

                messages = response.get("Messages", [])
                if not messages:
                    continue

                for msg in messages:
                    if not self._running:
                        break
                    await self._process_single_message(msg)

            except asyncio.CancelledError:
                logger.info("Worker loop cancelled.")
                break
            except Exception as e:
                logger.exception("Unexpected error in SQS poll loop: %s", e)
                await asyncio.sleep(5)

        logger.info("EC2 Worker stopped gracefully.")

    async def _process_single_message(self, msg: dict[str, Any]) -> None:
        """Processes one SQS message with error isolation."""
        receipt_handle = msg.get("ReceiptHandle")
        body_raw = msg.get("Body", "{}")

        try:
            body = json.loads(body_raw)
        except Exception:
            logger.error("Failed to parse SQS message JSON: %s", body_raw)
            return

        job_id = body.get("job_id")
        storage_ref = body.get("pdf_path")
        email = body.get("email")
        start_page = body.get("start_page")
        max_pages = body.get("max_pages")

        if not job_id or not storage_ref:
            logger.error("Malformed message missing job_id or pdf_path: %s", body)
            return

        logger.info("Processing job %s from SQS (Storage: %s, Email: %s)", job_id, storage_ref, email)

        try:
            await self.orchestrator.process_job(
                job_id=job_id,
                storage_ref=storage_ref,
                filename="booklet.pdf",
                email=email,
                start_page=start_page,
                max_pages=max_pages,
            )

            # Successfully evaluated -> delete message from SQS
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.sqs.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle,
                ),
            )
            logger.info("Successfully completed and deleted SQS message for job %s", job_id)

        except Exception as e:
            logger.exception("Failed to process job %s: %s", job_id, e)
            # Message will become visible again after VisibilityTimeout or go to DLQ

    def stop(self) -> None:
        """Signals the worker loop to stop."""
        logger.info("Stopping EC2 Worker...")
        self._running = False


def main() -> None:
    daemon = EC2WorkerDaemon()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler(signum: int, frame: Any) -> None:
        logger.info("Received signal %s; shutting down...", signum)
        daemon.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        loop.run_until_complete(daemon.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
