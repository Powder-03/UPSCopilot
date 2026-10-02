"""Unit tests for Scorecard PDF generator using PyMuPDF Story engine."""
from pathlib import Path

import pymupdf
from src.models.api import (
    OverallFeedback,
    StudentEvaluationReport,
    StudentQuestionEvaluation,
    StudentSummary,
)
from src.services.scorecard_pdf import generate_scorecard_pdf, save_scorecard_pdf


def _build_sample_report() -> StudentEvaluationReport:
    """Helper to build a sample student evaluation report for PDF testing."""
    return StudentEvaluationReport(
        status="success",
        document="GS2_Topper_Copy.pdf",
        summary=StudentSummary(
            total_score=108.5,
            max_marks=250.0,
            percentage=43.4,
            overall_feedback=OverallFeedback(
                key_strengths=[
                    "Strong grounding in constitutional articles (Art 324, 280, 21).",
                    "Effective multi-archetype presentation with tabular comparisons.",
                ],
                top_areas_to_improve=[
                    "Deepen second-order governance implications.",
                    "Ensure consistent Way Forward in 15-marker questions.",
                ],
            ),
        ),
        questions=[
            StudentQuestionEvaluation(
                q_num=1,
                max_marks=10.0,
                score=4.5,
                percentage=45.0,
                question="Examine the scope of judicial review under the Indian Constitution.",
                pros=["Cited Kesavananda Bharati", "Clear distinction on procedural review"],
                what_to_do_better=["Mention NJAC judgment context"],
            ),
            StudentQuestionEvaluation(
                q_num=2,
                max_marks=15.0,
                score=6.8,
                percentage=45.3,
                question="Analyze the role of the Finance Commission in fiscal federalism.",
                pros=["Covered 15th FC horizontal devolution criteria", "Addressed cess & surcharges issue"],
                what_to_do_better=["Incorporate FRBM target compliance"],
            ),
        ],
    )


def test_generate_scorecard_pdf_structure():
    """Verifies generate_scorecard_pdf produces valid multi-page PDF bytes with expected content."""
    report = _build_sample_report()
    pdf_bytes = generate_scorecard_pdf(report)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")

    # Open with PyMuPDF to inspect structure and text
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert doc.page_count >= 1

    all_text = ""
    for page in doc:
        all_text += page.get_text()

    assert "UPSC Civil Services Mains" in all_text
    assert "GS2_Topper_Copy.pdf" in all_text
    assert "108.5" in all_text
    assert "250" in all_text
    assert "Key Strengths" in all_text
    assert "Priority Areas to Improve" in all_text
    assert "Q1." in all_text
    assert "Q2." in all_text
    assert "Kesavananda Bharati" in all_text
    doc.close()


def test_save_scorecard_pdf_to_disk(tmp_path: Path):
    """Verifies save_scorecard_pdf correctly saves PDF to disk."""
    report = _build_sample_report()
    out_file = tmp_path / "subfolder" / "scorecard_test.pdf"

    saved_path = save_scorecard_pdf(report, out_file)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 1000

    doc = pymupdf.open(str(saved_path))
    assert doc.page_count >= 1
    doc.close()
