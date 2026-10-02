"""Thread-safe background job manager and state coordinator for asynchronous UPSC evaluation."""
import datetime
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from src.config import settings
from src.models.api import (
    JobStatus,
    JobStatusResponse,
    JobSubmitResponse,
    StudentEvaluationReport,
)
from src.pipeline import UnifiedEvaluationPipeline
from src.services.email_service import EmailService
from src.services.job_state_service import JobStateService
from src.services.queue_service import QueueService
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class JobManager:
    """Coordinates job lifecycle, file storage, execution in worker threads/SQS, and email delivery."""

    def __init__(
        self,
        upload_dir: str | Path | None = None,
        job_dir: str | Path | None = None,
        pipeline: UnifiedEvaluationPipeline | None = None,
        email_service: EmailService | None = None,
        storage_service: StorageService | None = None,
        job_state_service: JobStateService | None = None,
        queue_service: QueueService | None = None,
    ):
        self.upload_dir = Path(upload_dir or settings.upload_dir)
        self.job_dir = Path(job_dir or settings.job_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.job_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline = pipeline
        self.email_service = email_service or EmailService()
        self.storage_service = storage_service or StorageService(local_dir=self.upload_dir)
        self.job_state_service = job_state_service or JobStateService(local_dir=self.job_dir)
        self.queue_service = queue_service or QueueService()

        # In-memory fast cache with thread lock
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @property
    def pipeline(self) -> UnifiedEvaluationPipeline:
        """Lazily initialize UnifiedEvaluationPipeline on first use."""
        if self._pipeline is None:
            self._pipeline = UnifiedEvaluationPipeline()
        return self._pipeline

    def create_job(
        self,
        file_bytes: bytes,
        filename: str,
        email: str | None = None,
        start_page: int | None = None,
        max_pages: int | None = None,
    ) -> tuple[JobSubmitResponse, Path]:
        """
        Creates a new asynchronous job, saves the uploaded PDF (S3 or local), and initializes state.
        Returns the immediate HTTP submit response and the resolved local PDF path.
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        storage_ref = self.storage_service.save_file(file_bytes=file_bytes, filename=filename, job_id=job_id)
        local_pdf_path = self.storage_service.get_local_path(storage_ref, temp_dir=self.upload_dir)

        now = datetime.datetime.now(datetime.UTC).isoformat()
        job_data: dict[str, Any] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "progress_pct": 0,
            "current_step": "Job queued for processing",
            "email": email.strip() if email else None,
            "email_sent": False,
            "error": None,
            "result": None,
            "pdf_path": storage_ref,
            "filename": filename,
            "start_page": start_page,
            "max_pages": max_pages,
            "created_at": now,
            "updated_at": now,
        }

        with self._lock:
            self._jobs[job_id] = job_data
            self._persist_job_to_disk(job_id)

        submit_msg = (
            f"Evaluation queued! A full scorecard will be emailed to {email} upon completion."
            if email
            else "Evaluation queued! Poll check_status_url for progress and final scorecard."
        )

        response = JobSubmitResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            check_status_url=f"/api/v1/jobs/{job_id}",
            email=job_data["email"],
            message=submit_msg,
        )
        return response, local_pdf_path

    def get_job(self, job_id: str) -> JobStatusResponse | None:
        """Retrieves current job status from in-memory cache, DynamoDB, or local disk."""
        with self._lock:
            job_data = self._jobs.get(job_id)

        if not job_data:
            job_data = self.job_state_service.get_job(job_id)
            if job_data:
                with self._lock:
                    self._jobs[job_id] = job_data
            else:
                return None

        result_obj = None
        if job_data.get("result"):
            result_obj = StudentEvaluationReport.model_validate(job_data["result"])

        return JobStatusResponse(
            job_id=job_data["job_id"],
            status=JobStatus(job_data["status"]),
            progress_pct=job_data.get("progress_pct", 0),
            current_step=job_data.get("current_step", ""),
            email=job_data.get("email"),
            email_sent=job_data.get("email_sent", False),
            error=job_data.get("error"),
            result=result_obj,
        )

    def update_progress(
        self,
        job_id: str,
        status: JobStatus,
        progress_pct: int,
        current_step: str,
    ) -> None:
        """Updates in-memory and persisted progress."""
        with self._lock:
            if job_id not in self._jobs:
                fetched = self.job_state_service.get_job(job_id)
                if fetched:
                    self._jobs[job_id] = fetched

            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status.value
                self._jobs[job_id]["progress_pct"] = progress_pct
                self._jobs[job_id]["current_step"] = current_step
                self._jobs[job_id]["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
                self._persist_job_to_disk(job_id)

    def dispatch_job(
        self,
        job_id: str,
        pdf_path: Path | str,
        start_page: int | None = None,
        max_pages: int | None = None,
    ) -> str:
        """
        Dispatches background evaluation for the given job.
        Uses SQS if configured, or falls back to background worker thread.
        """
        with self._lock:
            # Prefer the canonical storage_ref (s3:// URI if available) stored in job_data
            storage_ref = self._jobs.get(job_id, {}).get("pdf_path", str(pdf_path))

        return self.queue_service.dispatch(
            job_id=job_id,
            pdf_path=storage_ref,
            start_page=start_page,
            max_pages=max_pages,
            local_thread_runner=self._run_job_worker,
        )

    def _run_job_worker(
        self,
        job_id: str,
        pdf_path: Path | str,
        start_page: int | None,
        max_pages: int | None,
    ) -> None:
        """Worker executing the pipeline and email dispatch."""
        logger.info("Worker started executing job %s (Target: %s)", job_id, pdf_path)
        try:
            resolved_pdf = self.storage_service.get_local_path(str(pdf_path))

            def callback(status: JobStatus, pct: int, step: str) -> None:
                self.update_progress(job_id, status, pct, step)

            # Run evaluation pipeline
            report = self.pipeline.run_pipeline(
                pdf_path=resolved_pdf,
                start_page=start_page,
                max_pages=max_pages,
                progress_callback=callback,
            )

            # Email Delivery
            email_sent = False
            with self._lock:
                target_email = self._jobs.get(job_id, {}).get("email")

            if target_email:
                self.update_progress(job_id, JobStatus.SENDING_EMAIL, 98, f"Sending scorecard email to {target_email}...")
                email_sent = self.email_service.send_evaluation_email(
                    to_email=target_email,
                    report=report,
                    job_id=job_id,
                )

            # Mark completed
            with self._lock:
                if job_id not in self._jobs:
                    fetched = self.job_state_service.get_job(job_id)
                    if fetched:
                        self._jobs[job_id] = fetched

                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = JobStatus.COMPLETED.value
                    self._jobs[job_id]["progress_pct"] = 100
                    self._jobs[job_id]["current_step"] = "Evaluation complete."
                    self._jobs[job_id]["email_sent"] = email_sent
                    self._jobs[job_id]["result"] = report.model_dump()
                    self._jobs[job_id]["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
                    self._persist_job_to_disk(job_id)

            logger.info("Job %s completed successfully. Email sent: %s", job_id, email_sent)

        except Exception as e:
            logger.exception("Job %s failed with exception: %s", job_id, e)
            with self._lock:
                if job_id not in self._jobs:
                    fetched = self.job_state_service.get_job(job_id)
                    if fetched:
                        self._jobs[job_id] = fetched

                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = JobStatus.FAILED.value
                    self._jobs[job_id]["error"] = str(e)
                    self._jobs[job_id]["current_step"] = "Evaluation failed."
                    self._jobs[job_id]["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
                    self._persist_job_to_disk(job_id)

    def _persist_job_to_disk(self, job_id: str) -> None:
        """Internal helper to save job state to DynamoDB and/or local JSON file."""
        if job_id in self._jobs:
            try:
                self.job_state_service.save_job(job_id, self._jobs[job_id])
            except Exception as e:
                logger.warning("Failed to persist job %s: %s", job_id, e)


# Global singleton job manager
job_manager = JobManager()
