"""Scorecard PDF generator using PyMuPDF Story engine.

Renders a polished, multi-page UPSC Mains Evaluation Scorecard PDF from
a StudentEvaluationReport. Uses pymupdf.Story for HTML-to-PDF conversion
with zero external dependencies beyond PyMuPDF (already installed).
"""
import html
import logging
from datetime import UTC, datetime
from pathlib import Path

import pymupdf
from src.models.api import StudentEvaluationReport

logger = logging.getLogger(__name__)

# A4 dimensions in points (72 pt/inch)
A4_WIDTH = 595.28
A4_HEIGHT = 841.89
MARGIN = 50


def _score_color_hex(pct: float) -> str:
    """Returns a hex color string based on score percentage."""
    if pct >= 50:
        return "#15803d"
    if pct >= 35:
        return "#b45309"
    return "#dc2626"


def generate_scorecard_pdf(report: StudentEvaluationReport) -> bytes:
    """
    Generates a complete UPSC Evaluation Scorecard PDF from a StudentEvaluationReport.
    Returns raw PDF bytes suitable for email attachment or disk write.
    """
    doc_name = html.escape(report.document)
    total_score = report.summary.total_score
    max_marks = report.summary.max_marks
    pct = report.summary.percentage
    generated_at = datetime.now(UTC).strftime("%d %B %Y, %H:%M UTC")

    # Build strengths items
    strengths_items = "".join(
        f"<li>{html.escape(s)}</li>" for s in report.summary.overall_feedback.key_strengths
    )
    if not strengths_items:
        strengths_items = "<li>No specific strengths recorded.</li>"

    # Build improvements items
    improvements_items = "".join(
        f"<li>{html.escape(i)}</li>" for i in report.summary.overall_feedback.top_areas_to_improve
    )
    if not improvements_items:
        improvements_items = "<li>No priority improvements recorded.</li>"

    # Build question cards
    question_cards = ""
    for q in report.questions:
        q_text = html.escape(q.question[:200]) if q.question else "—"
        color = _score_color_hex(q.percentage)

        pros_html = ""
        if q.pros:
            pros_items = "".join(f"<li>{html.escape(p)}</li>" for p in q.pros)
            pros_html = f"""
            <div style="margin-top:8px;">
                <b style="color:#15803d;">Strengths:</b>
                <ul style="margin:4px 0 8px 16px;">{pros_items}</ul>
            </div>"""

        better_html = ""
        if q.what_to_do_better:
            better_items = "".join(f"<li>{html.escape(b)}</li>" for b in q.what_to_do_better)
            better_html = f"""
            <div style="margin-top:6px;">
                <b style="color:#b45309;">What To Improve:</b>
                <ul style="margin:4px 0 8px 16px;">{better_items}</ul>
            </div>"""

        question_cards += f"""
        <div style="border:1px solid #d1d5db; border-radius:6px; padding:12px; margin-bottom:14px; background-color:#fafafa;">
            <p style="margin:0 0 6px 0;">
                <b>Q{q.q_num}.</b> <i style="color:#4b5563;">{q_text}</i>
            </p>
            <p style="margin:0 0 4px 0;">
                <b style="color:{color};">Score: {q.score:g} / {q.max_marks:g}</b>
                <span style="color:#6b7280;"> ({q.percentage:.1f}%)</span>
            </p>
            {pros_html}
            {better_html}
        </div>
        """

    # Assemble full HTML
    full_html = f"""
    <html>
    <body style="font-family: Helvetica, Arial, sans-serif; font-size:11px; color:#1f2937; line-height:1.5;">

        <!-- Header -->
        <div style="background-color:#1e3a8a; color:white; padding:20px 24px; border-radius:8px; text-align:center; margin-bottom:20px;">
            <h1 style="margin:0; font-size:20px;">UPSC Civil Services Mains</h1>
            <h2 style="margin:4px 0 0 0; font-size:14px; font-weight:normal; opacity:0.9;">Evaluation Scorecard</h2>
        </div>

        <!-- Meta Info -->
        <table style="width:100%; margin-bottom:18px; font-size:11px;">
            <tr>
                <td><b>Document:</b> {doc_name}</td>
                <td style="text-align:right;"><b>Generated:</b> {generated_at}</td>
            </tr>
        </table>

        <!-- Score Summary -->
        <div style="border:2px solid #1e3a8a; border-radius:8px; padding:16px; text-align:center; margin-bottom:20px;">
            <p style="font-size:10px; text-transform:uppercase; letter-spacing:1px; color:#6b7280; margin:0 0 4px 0;">Overall Score</p>
            <p style="font-size:28px; font-weight:bold; color:#1e3a8a; margin:0 0 4px 0;">
                {total_score:g} <span style="font-size:16px; color:#9ca3af;">/ {max_marks:g}</span>
            </p>
            <p style="font-size:13px; font-weight:bold; color:#374151; margin:0;">Percentage: {pct:.1f}%</p>
        </div>

        <!-- Executive Summary -->
        <div style="border:1px solid #d1d5db; border-radius:6px; padding:14px; margin-bottom:20px;">
            <h3 style="margin:0 0 8px 0; font-size:13px; color:#111827;">Executive Summary</h3>
            <div style="margin-bottom:10px;">
                <b style="color:#15803d; font-size:11px;">Key Strengths:</b>
                <ul style="margin:4px 0 10px 16px;">{strengths_items}</ul>
            </div>
            <div>
                <b style="color:#b45309; font-size:11px;">Priority Areas to Improve:</b>
                <ul style="margin:4px 0 0 16px;">{improvements_items}</ul>
            </div>
        </div>

        <!-- Question Breakdown -->
        <h3 style="font-size:14px; color:#111827; margin:0 0 12px 0;">Question-by-Question Breakdown</h3>
        {question_cards}

        <!-- Footer -->
        <div style="margin-top:20px; text-align:center; font-size:9px; color:#9ca3af; border-top:1px solid #e5e7eb; padding-top:10px;">
            Evaluated by UPSCopilot Evaluation Engine &bull; Calibrated against authentic UPSC CSE Mains standards.
        </div>

    </body>
    </html>
    """

    # Render HTML to PDF using PyMuPDF Story engine
    mediabox = pymupdf.Rect(0, 0, A4_WIDTH, A4_HEIGHT)
    content_rect = pymupdf.Rect(MARGIN, MARGIN, A4_WIDTH - MARGIN, A4_HEIGHT - MARGIN)

    story = pymupdf.Story(html=full_html)
    writer = None
    pdf_path_tmp = None

    try:
        # PyMuPDF Story.write requires a DocumentWriter
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path_tmp = tmp.name

        writer = pymupdf.DocumentWriter(pdf_path_tmp)

        more = True
        while more:
            dev = writer.begin_page(mediabox)
            more, _ = story.place(content_rect)
            story.draw(dev)
            writer.end_page()

        writer.close()

        # Read back bytes
        pdf_bytes = Path(pdf_path_tmp).read_bytes()
        logger.info("Generated scorecard PDF: %d bytes, document: %s", len(pdf_bytes), doc_name)
        return pdf_bytes

    finally:
        # Clean up temp file
        if pdf_path_tmp:
            try:
                Path(pdf_path_tmp).unlink(missing_ok=True)
            except Exception:
                pass


def save_scorecard_pdf(report: StudentEvaluationReport, output_path: str | Path) -> Path:
    """
    Convenience wrapper: generates scorecard PDF and writes to disk.
    Returns the output Path.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = generate_scorecard_pdf(report)
    output.write_bytes(pdf_bytes)
    logger.info("Saved scorecard PDF to %s (%d bytes)", output, len(pdf_bytes))
    return output
