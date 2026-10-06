"""Live test script for sending an evaluation email via Google SMTP with attached PDF scorecard."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings
from src.models.api import (
    OverallFeedback,
    StudentEvaluationReport,
    StudentQuestionEvaluation,
    StudentSummary,
)
from src.services.email_service import EmailService
from src.services.scorecard_pdf import generate_scorecard_pdf


def run_live_test():
    print("=== Testing Google SMTP Email Dispatch ===")
    print(f"Provider: {settings.email_provider}")
    print(f"SMTP Host: {settings.smtp_host}:{settings.smtp_port}")
    print(f"SMTP User: {settings.smtp_user}")
    print(f"From Name: {settings.mail_from_name}")

    if not settings.smtp_user or not settings.smtp_password:
        print("ERROR: SMTP_USER or SMTP_PASSWORD is not configured in .env.")
        sys.exit(1)

    # 1. Build a representative student evaluation report
    report = StudentEvaluationReport(
        status="success",
        document="UPSC_Mains_GS2_Mock_Copy.pdf",
        summary=StudentSummary(
            total_score=48.5,
            max_marks=100.0,
            percentage=48.5,
            overall_feedback=OverallFeedback(
                key_strengths=[
                    "Strong constitutional foundation with precise Article 324 and Article 21 citations.",
                    "Structured multi-stakeholder governance analysis with balanced diagrams.",
                ],
                top_areas_to_improve=[
                    "Deepen committee references (e.g., 2nd ARC, Punchhi Commission) in conclusions.",
                    "Sharpen direct answering of directives like 'Critically Analyze' in 15-markers.",
                ],
            ),
        ),
        questions=[
            StudentQuestionEvaluation(
                q_num=1,
                max_marks=10.0,
                score=5.0,
                percentage=50.0,
                question="Examine the scope of Judicial Review in light of the Basic Structure Doctrine.",
                pros=["Accurately cited Kesavananda Bharati (1973) and Minerva Mills (1980)."],
                what_to_do_better=["Distinguish between procedural review and substantive due process."],
            ),
            StudentQuestionEvaluation(
                q_num=2,
                max_marks=10.0,
                score=4.8,
                percentage=48.0,
                question="Discuss the constitutional mandate of the Election Commission of India under Article 324.",
                pros=["Mentioned Mohinder Singh Gill case and Model Code of Conduct enforcement."],
                what_to_do_better=["Incorporate Law Commission 255th Report recommendations on appointment transparency."],
            ),
        ],
    )

    # 2. Generate authentic high-resolution PDF scorecard
    print("Generating scorecard PDF bytes via PyMuPDF Story...")
    pdf_bytes = generate_scorecard_pdf(report)
    print(f"Generated PDF Scorecard: {len(pdf_bytes)} bytes.")

    # 3. Instantiate EmailService and send email
    service = EmailService()
    recipient = settings.smtp_user  # Send to the user's gmail address
    print(f"Sending evaluation email to {recipient}...")

    success = service.send_evaluation_email(
        to_email=recipient,
        report=report,
        job_id="live_test_001",
        pdf_bytes=pdf_bytes,
    )

    if success:
        print(f"\nSUCCESS: Live evaluation email successfully sent to {recipient} via Google SMTP!")
    else:
        print("\nFAILURE: Could not deliver email via Google SMTP. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    run_live_test()
