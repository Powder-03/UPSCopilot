"""Unit tests for API schemas and student scorecard models, ensuring zero performance bands."""
import json

import pytest
from pydantic import ValidationError
from src.models.api import (
    JobStatus,
    JobStatusResponse,
    JobSubmitResponse,
    OverallFeedback,
    StudentEvaluationReport,
    StudentQuestionEvaluation,
    StudentSummary,
)


def test_student_question_evaluation_schema():
    """Verifies that StudentQuestionEvaluation serializes cleanly and rejects performance band fields."""
    q_eval = StudentQuestionEvaluation(
        q_num=1,
        max_marks=10.0,
        score=5.5,
        percentage=55.0,
        question="Critically examine constitutional morality.",
        pros=["Accurate case citations (Navtej Johar, Shayara Bano).", "Clear structured subheadings."],
        what_to_do_better=["Cite Dr. B.R. Ambedkar's Constituent Assembly speech.", "Include counter-view on judicial overreach."],
    )

    data = q_eval.model_dump()
    assert data["q_num"] == 1
    assert data["score"] == 5.5
    assert data["percentage"] == 55.0
    assert len(data["pros"]) == 2
    assert len(data["what_to_do_better"]) == 2

    # Strictly assert that no performance band keys exist
    forbidden_keys = {"performance_band", "overall_band", "performance", "trajectory_verdict"}
    assert not any(key in data for key in forbidden_keys)

    # Extra fields should be forbidden
    with pytest.raises(ValidationError):
        StudentQuestionEvaluation(
            q_num=1,
            max_marks=10.0,
            score=5.5,
            percentage=55.0,
            question="Test",
            performance_band="Good",  # Forbidden!
        )


def test_student_evaluation_report_schema():
    """Verifies the complete student scorecard report schema."""
    report = StudentEvaluationReport(
        status="success",
        document="GS-II.pdf",
        summary=StudentSummary(
            total_score=108.5,
            max_marks=250.0,
            percentage=43.4,
            overall_feedback=OverallFeedback(
                key_strengths=["Clean flowcharts on 15-markers.", "Article 323A citation."],
                top_areas_to_improve=["Cite 2nd ARC recommendations.", "Cover economic dimensions."],
            ),
        ),
        questions=[
            StudentQuestionEvaluation(
                q_num=1,
                max_marks=10.0,
                score=5.5,
                percentage=55.0,
                question="Question 1 text",
                pros=["Good point 1"],
                what_to_do_better=["Improvement 1"],
            )
        ],
    )

    raw_json = report.model_dump_json()
    parsed = json.loads(raw_json)

    assert parsed["status"] == "success"
    assert parsed["document"] == "GS-II.pdf"
    assert parsed["summary"]["total_score"] == 108.5
    assert parsed["summary"]["max_marks"] == 250.0
    assert parsed["summary"]["percentage"] == 43.4
    assert len(parsed["questions"]) == 1

    # Verify absence of performance band anywhere in the serialized JSON
    assert "performance_band" not in raw_json
    assert "overall_band" not in raw_json
    assert "trajectory_verdict" not in raw_json


def test_job_response_schemas():
    """Verifies JobSubmitResponse and JobStatusResponse schemas."""
    submit = JobSubmitResponse(
        job_id="job_123",
        status=JobStatus.QUEUED,
        check_status_url="/api/v1/jobs/job_123",
        email="test@example.com",
        message="Queued for evaluation",
    )
    assert submit.job_id == "job_123"
    assert submit.status == JobStatus.QUEUED

    status_resp = JobStatusResponse(
        job_id="job_123",
        status=JobStatus.PARSING,
        progress_pct=25,
        current_step="Parsing pages...",
        email="test@example.com",
        email_sent=False,
    )
    assert status_resp.progress_pct == 25
    assert status_resp.status == JobStatus.PARSING
