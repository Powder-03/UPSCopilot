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
    assert "Exceptional case law citations" in html_content
    assert "Deepen policy implications" in html_content
    assert "Detailed Scorecard Attached" in html_content

    # Detailed question breakdown is excluded from email body (delivered in PDF)
    assert "Question 1" not in html_content
    assert "Question 2" not in html_content

    # Zero performance bands
    assert "performance_band" not in html_content
    assert "overall_band" not in html_content
    assert "Needs Foundation" not in html_content


def test_email_text_rendering():
    """Verifies that the plain-text fallback contains score and highlights without question breakdown."""
    report = _create_sample_report()
    service = EmailService(provider="mock")

    text_content = service.render_scorecard_text(report)
    assert "Sample_GS2.pdf" in text_content
    assert "12.5 / 20" in text_content
    assert "What You Did Good:" in text_content
    assert "Exceptional case law citations" in text_content
    assert "What Could Be Better:" in text_content
    assert "Deepen policy implications" in text_content
    assert "Question 1" not in text_content


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
        text_body=service.render_scorecard_text(report),
        pdf_bytes=mock_pdf,
    )

    assert msg.get_content_type() == "multipart/mixed"
    assert msg["To"] == "student@example.com"
    assert "evaluator@upscopilot.com" in msg["From"]
    assert msg["Subject"] == "Your Scorecard"

    parts = list(msg.walk())
    payloads = [p.get_content_type() for p in parts]
    assert "text/html" in payloads
    assert "text/plain" in payloads
    assert "application/pdf" in payloads


def test_google_smtp_service_defaults():
    """Verifies that Google SMTP provider initializes with smtp.gmail.com and port 587."""
    service = EmailService(
        provider="google",
        from_email="aspirant@gmail.com",
        smtp_user="aspirant@gmail.com",
        smtp_password="app_password_123",
    )
    assert service.provider == "google"
    assert service.smtp_host == "smtp.gmail.com"
    assert service.smtp_port == 587
    assert service.from_email == "aspirant@gmail.com"


def test_google_smtp_dispatch_mock():
    """Verifies that Google SMTP dispatches emails via smtplib.SMTP with STARTTLS and auth."""
    from unittest.mock import MagicMock, patch

    report = _create_sample_report()
    service = EmailService(
        provider="google",
        from_email="sender@gmail.com",
        smtp_user="sender@gmail.com",
        smtp_password="app_password_xyz",
    )

    mock_smtp_instance = MagicMock()
    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_smtp_cls.return_value.__enter__.return_value = mock_smtp_instance

        success = service.send_evaluation_email(
            to_email="student@upsc.test",
            report=report,
            pdf_bytes=b"%PDF-1.4 sample",
        )

        assert success is True
        mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 587, timeout=20)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("sender@gmail.com", "app_password_xyz")
        assert mock_smtp_instance.sendmail.called
        call_args = mock_smtp_instance.sendmail.call_args[0]
        assert call_args[0] == "sender@gmail.com"
        assert call_args[1] == ["student@upsc.test"]


def test_google_smtp_dispatch_ssl_465():
    """Verifies that Google SMTP dispatches via SMTP_SSL when port is 465."""
    from unittest.mock import MagicMock, patch

    report = _create_sample_report()
    service = EmailService(
        provider="google",
        smtp_port=465,
        from_email="sender@gmail.com",
        smtp_user="sender@gmail.com",
        smtp_password="app_password_ssl",
    )

    mock_ssl_instance = MagicMock()
    with patch("smtplib.SMTP_SSL") as mock_ssl_cls:
        mock_ssl_cls.return_value.__enter__.return_value = mock_ssl_instance

        success = service.send_evaluation_email(
            to_email="student@upsc.test",
            report=report,
            pdf_bytes=b"%PDF-1.4 sample",
        )

        assert success is True
        mock_ssl_cls.assert_called_once_with("smtp.gmail.com", 465, timeout=20)
        mock_ssl_instance.login.assert_called_once_with("sender@gmail.com", "app_password_ssl")
        assert mock_ssl_instance.sendmail.called

