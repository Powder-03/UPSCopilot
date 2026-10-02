"""Unit tests for EmailService: HTML scorecard generation and mock dispatch."""
from pathlib import Path

from src.models.api import (
    OverallFeedback,
    StudentEvaluationReport,
    StudentQuestionEvaluation,
    StudentSummary,
)
from src.services.email_service import EmailService


def _create_sample_report() -> StudentEvaluationReport:
    """Helper to build a sample student report."""
    return StudentEvaluationReport(
        status="success",
        document="Sample_GS2.pdf",
        summary=StudentSummary(
            total_score=12.5,
            max_marks=20.0,
            percentage=62.5,
            overall_feedback=OverallFeedback(
                key_strengths=["Exceptional case law citations", "Multi-archetype diagrams"],
                top_areas_to_improve=["Deepen policy implications", "Add punchy conclusions"],
            ),
        ),
        questions=[
            StudentQuestionEvaluation(
                q_num=1,
                max_marks=10.0,
                score=6.5,
                percentage=65.0,
                question="Explain the significance of judicial review in India.",
                pros=["Cited Kesavananda Bharati and Minerva Mills", "Covered Basic Structure Doctrine"],
                what_to_do_better=["Add distinction between procedural vs substantive review"],
            ),
            StudentQuestionEvaluation(
                q_num=2,
                max_marks=10.0,
                score=6.0,
                percentage=60.0,
                question="Discuss the powers of the Election Commission under Article 324.",
                pros=["Covered Mohinder Singh Gill case", "Mentioned Model Code of Conduct"],
                what_to_do_better=["Mention Dinesh Goswami Committee recommendations"],
            ),
        ],
    )


def test_email_html_rendering():
    """Verifies that the generated HTML contains all essential elements and zero performance bands."""
    report = _create_sample_report()
    service = EmailService(provider="mock")

    html_content = service.render_scorecard_html(report)

    # Core elements present
    assert "Sample_GS2.pdf" in html_content
    assert "12.5" in html_content
    assert "20" in html_content
    assert "62.5%" in html_content
    assert "Question 1" in html_content
    assert "Question 2" in html_content
    assert "Kesavananda Bharati" in html_content
    assert "Dinesh Goswami" in html_content

    # Zero performance bands
    assert "performance_band" not in html_content
    assert "overall_band" not in html_content
    assert "Needs Foundation" not in html_content


def test_mock_email_dispatch_with_pdf(tmp_path: Path):
    """Verifies that mock email dispatch saves both HTML preview and PDF attachment."""
    report = _create_sample_report()
    service = EmailService(provider="mock")
    mock_pdf = b"%PDF-1.4 mock pdf data"

    success = service.send_evaluation_email(
        to_email="aspirant@test.com",
        report=report,
        job_id="test_job_preview",
        pdf_bytes=mock_pdf,
    )
    assert success is True


def test_build_mime_message_with_pdf():
    """Verifies that _build_mime_message creates a multipart/mixed message with PDF attachment."""
    report = _create_sample_report()
    service = EmailService(provider="ses", from_email="evaluator@upscopilot.com")
    mock_pdf = b"%PDF-1.4 fake binary pdf content"

    msg = service._build_mime_message(
        to_email="student@example.com",
        subject="Your Scorecard",
        html_body=service.render_scorecard_html(report),
        pdf_bytes=mock_pdf,
    )

    assert msg.get_content_type() == "multipart/mixed"
    assert msg["To"] == "student@example.com"
    assert msg["From"] == "evaluator@upscopilot.com"
    assert msg["Subject"] == "Your Scorecard"

    parts = list(msg.walk())
    payloads = [p.get_content_type() for p in parts]
    assert "text/html" in payloads
    assert "application/pdf" in payloads
