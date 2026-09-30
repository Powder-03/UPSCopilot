"""Thread-safe background job manager and state coordinator for asynchronous UPSC evaluation."""
import datetime
import json
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

logger = logging.getLogger(__name__)


class JobManager:
    """Coordinates job lifecycle, file storage, execution in worker threads, and email delivery."""

    def __init__(
        self,
        upload_dir: str | Path | None = None,
        job_dir: str | Path | None = None,
        pipeline: UnifiedEvaluationPipeline | None = None,
        email_service: EmailService | None = None,
    ):
        self.upload_dir = Path(upload_dir or settings.upload_dir)
        self.job_dir = Path(job_dir or settings.job_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.job_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline = pipeline
        self.email_service = email_service or EmailService()

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
        Creates a new asynchronous job, saves the uploaded PDF, and initializes state.
        Returns the immediate HTTP submit response and the saved PDF path.
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        pdf_path = self.upload_dir / f"{job_id}_{filename}"
        pdf_path.write_bytes(file_bytes)

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
            "pdf_path": str(pdf_path),
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
        return response, pdf_path

    def get_job(self, job_id: str) -> JobStatusResponse | None:
        """Retrieves current job status from in-memory cache or disk."""
        with self._lock:
            job_data = self._jobs.get(job_id)

        if not job_data:
            job_file = self.job_dir / f"{job_id}.json"
            if job_file.exists():
                try:
                    with open(job_file, encoding="utf-8") as f:
                        job_data = json.load(f)
                    with self._lock:
                        self._jobs[job_id] = job_data
                except Exception as e:
                    logger.warning("Failed to load job %s from disk: %s", job_id, e)
                    return None
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
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status.value
                self._jobs[job_id]["progress_pct"] = progress_pct
                self._jobs[job_id]["current_step"] = current_step
                self._jobs[job_id]["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
                self._persist_job_to_disk(job_id)

    def dispatch_job(
        self,
        job_id: str,
        pdf_path: Path,
        start_page: int | None = None,
        max_pages: int | None = None,
    ) -> None:
        """Spawns background evaluation thread for the given job."""
        thread = threading.Thread(
            target=self._run_job_worker,
            args=(job_id, pdf_path, start_page, max_pages),
            daemon=True,
            name=f"Worker-{job_id}",
        )
        thread.start()
        logger.info("Dispatched background thread %s for %s", thread.name, job_id)

    def _run_job_worker(
        self,
        job_id: str,
        pdf_path: Path,
        start_page: int | None,
        max_pages: int | None,
    ) -> None:
        """Worker thread executing the pipeline and email dispatch."""
        logger.info("Worker started executing job %s (PDF: %s)", job_id, pdf_path)
        try:
            def callback(status: JobStatus, pct: int, step: str) -> None:
                self.update_progress(job_id, status, pct, step)

            # Run evaluation pipeline
            report = self.pipeline.run_pipeline(
                pdf_path=pdf_path,
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
                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = JobStatus.FAILED.value
                    self._jobs[job_id]["error"] = str(e)
                    self._jobs[job_id]["current_step"] = "Evaluation failed."
                    self._jobs[job_id]["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
                    self._persist_job_to_disk(job_id)

    def _persist_job_to_disk(self, job_id: str) -> None:
        """Internal helper to save job state to JSON file."""
        try:
            job_file = self.job_dir / f"{job_id}.json"
            with open(job_file, "w", encoding="utf-8") as f:
                json.dump(self._jobs[job_id], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to persist job %s to disk: %s", job_id, e)


# Global singleton job manager
job_manager = JobManager()
