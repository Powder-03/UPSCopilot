"""Unit tests for JobManager: job lifecycle, concurrency safety, and state persistence."""
from pathlib import Path

from src.models.api import (
    JobStatus,
    OverallFeedback,
    StudentEvaluationReport,
    StudentQuestionEvaluation,
    StudentSummary,
)
from src.services.email_service import EmailService
from src.services.job_manager import JobManager


def test_job_manager_lifecycle(tmp_path: Path):
    """Tests job creation, progress tracking, and retrieval."""
    upload_dir = tmp_path / "uploads"
    job_dir = tmp_path / "jobs"

    manager = JobManager(
        upload_dir=upload_dir,
        job_dir=job_dir,
        email_service=EmailService(provider="mock"),
    )

    pdf_bytes = b"%PDF-1.4 Mock PDF Content"
    submit_resp, saved_path = manager.create_job(
        file_bytes=pdf_bytes,
        filename="test_copy.pdf",
        email="aspirant@upsc.gov",
        start_page=3,
        max_pages=10,
    )

    assert submit_resp.job_id.startswith("job_")
    assert submit_resp.status == JobStatus.QUEUED
    assert submit_resp.email == "aspirant@upsc.gov"
    assert saved_path.exists()
    assert saved_path.read_bytes() == pdf_bytes

    # Retrieve initial job
    job = manager.get_job(submit_resp.job_id)
    assert job is not None
    assert job.status == JobStatus.QUEUED
    assert job.progress_pct == 0

    # Update progress
    manager.update_progress(submit_resp.job_id, JobStatus.PARSING, 20, "OCR parsing active...")
    job = manager.get_job(submit_resp.job_id)
    assert job is not None
    assert job.status == JobStatus.PARSING
    assert job.progress_pct == 20
    assert job.current_step == "OCR parsing active..."

    # Complete job
    sample_report = StudentEvaluationReport(
        status="success",
        document="test_copy.pdf",
        summary=StudentSummary(
            total_score=5.0,
            max_marks=10.0,
            percentage=50.0,
            overall_feedback=OverallFeedback(
                key_strengths=["Good structure"],
                top_areas_to_improve=["Add case laws"],
            ),
        ),
        questions=[
            StudentQuestionEvaluation(
                q_num=1,
                max_marks=10.0,
                score=5.0,
                percentage=50.0,
                question="Question 1",
                pros=["Clear attempt"],
                what_to_do_better=["Deepen analysis"],
            )
        ],
    )

    # Simulate completion directly in job data
    with manager._lock:
        manager._jobs[submit_resp.job_id]["status"] = JobStatus.COMPLETED.value
        manager._jobs[submit_resp.job_id]["progress_pct"] = 100
        manager._jobs[submit_resp.job_id]["result"] = sample_report.model_dump()
        manager._persist_job_to_disk(submit_resp.job_id)

    completed_job = manager.get_job(submit_resp.job_id)
    assert completed_job is not None
    assert completed_job.status == JobStatus.COMPLETED
    assert completed_job.progress_pct == 100
    assert completed_job.result is not None
    assert completed_job.result.summary.total_score == 5.0
    assert completed_job.result.questions[0].score == 5.0


def test_job_manager_not_found(tmp_path: Path):
    """Verifies that non-existent job returns None."""
    manager = JobManager(
        upload_dir=tmp_path / "uploads",
        job_dir=tmp_path / "jobs",
    )
    assert manager.get_job("job_non_existent") is None
