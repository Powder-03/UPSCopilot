"""Email delivery service for UPSC evaluation scorecards.

Supports Amazon SES (boto3), standard SMTP, and mock/console preview for local testing.
Generates responsive, student-centric HTML scorecards with zero performance bands.
"""
import html
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from src.config import settings
from src.models.api import StudentEvaluationReport

logger = logging.getLogger(__name__)


class EmailService:
    """Renders and dispatches evaluation scorecard emails to students."""

    def __init__(
        self,
        provider: str | None = None,
        from_email: str | None = None,
    ):
        self.provider = provider or settings.email_provider
        self.from_email = from_email or settings.ses_from_email

    def render_scorecard_html(self, report: StudentEvaluationReport) -> str:
        """Builds a responsive, elegant HTML scorecard email."""
        doc_name = html.escape(report.document)
        total_score = report.summary.total_score
        max_marks = report.summary.max_marks
        pct = report.summary.percentage

        # Strengths items
        strengths_html = "".join(
            f"<li style='margin-bottom: 6px;'>{html.escape(s)}</li>"
            for s in report.summary.overall_feedback.key_strengths
        )
        if not strengths_html:
            strengths_html = "<li>No specific strengths recorded.</li>"

        # Improvements items
        improvements_html = "".join(
            f"<li style='margin-bottom: 6px;'>{html.escape(i)}</li>"
            for i in report.summary.overall_feedback.top_areas_to_improve
        )
        if not improvements_html:
            improvements_html = "<li>No priority improvements recorded.</li>"

        # Questions cards
        questions_cards_html = ""
        for q in report.questions:
            q_text = html.escape(q.question)
            pros_items = "".join(
                f"<li style='margin-bottom: 4px;'>{html.escape(p)}</li>" for p in q.pros
            )
            pros_block = (
                f"<div style='margin-top: 10px;'>"
                f"<strong style='color: #15803d; font-size: 13px;'>✓ Strengths / Pros:</strong>"
                f"<ul style='margin: 4px 0 10px 20px; padding: 0; color: #1f2937; font-size: 13px;'>{pros_items}</ul>"
                f"</div>"
                if q.pros
                else ""
            )

            better_items = "".join(
                f"<li style='margin-bottom: 4px;'>{html.escape(b)}</li>" for b in q.what_to_do_better
            )
            better_block = (
                f"<div style='margin-top: 8px;'>"
                f"<strong style='color: #b45309; font-size: 13px;'>▲ What to Do Better:</strong>"
                f"<ul style='margin: 4px 0 10px 20px; padding: 0; color: #1f2937; font-size: 13px;'>{better_items}</ul>"
                f"</div>"
                if q.what_to_do_better
                else ""
            )

            score_color = "#15803d" if q.percentage >= 50 else ("#b45309" if q.percentage >= 35 else "#dc2626")

            questions_cards_html += f"""
            <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div style="font-weight: 700; color: #111827; font-size: 15px;">Question {q.q_num}</div>
                    <div style="font-weight: 700; color: {score_color}; font-size: 15px;">
                        {q.score:g} / {q.max_marks:g} <span style="font-size: 12px; font-weight: normal; color: #6b7280;">({q.percentage:.1f}%)</span>
                    </div>
                </div>
                <div style="font-size: 13px; color: #4b5563; font-style: italic; margin-bottom: 12px; line-height: 1.4;">
                    "{q_text}"
                </div>
                {pros_block}
                {better_block}
            </div>
            """

        html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UPSC Evaluation Report</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f3f4f6; margin: 0; padding: 24px 16px; color: #111827;">
    <div style="max-width: 640px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; border: 1px solid #e5e7eb; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
        
        <!-- Header Banner -->
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: #ffffff; padding: 28px 24px; text-align: center;">
            <h1 style="margin: 0; font-size: 22px; font-weight: 800; letter-spacing: -0.5px;">UPSC Mains Evaluation Report</h1>
            <p style="margin: 6px 0 0 0; font-size: 14px; opacity: 0.9;">Document: {doc_name}</p>
        </div>

        <!-- Score Summary Card -->
        <div style="padding: 24px; background-color: #f9fafb; border-bottom: 1px solid #e5e7eb;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #6b7280; font-weight: 600;">Overall Score</div>
                <div style="font-size: 38px; font-weight: 800; color: #1e3a8a; margin: 4px 0;">
                    {total_score:g} <span style="font-size: 20px; color: #9ca3af; font-weight: 500;">/ {max_marks:g}</span>
                </div>
                <div style="font-size: 14px; font-weight: 600; color: #374151;">Overall Percentage: {pct:.1f}%</div>
            </div>

            <!-- Strengths and Weaknesses Grid -->
            <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px;">
                <div style="margin-bottom: 14px;">
                    <div style="font-weight: 700; color: #15803d; font-size: 13px; margin-bottom: 6px;">🎯 Key Strengths:</div>
                    <ul style="margin: 0 0 0 18px; padding: 0; font-size: 13px; color: #374151;">{strengths_html}</ul>
                </div>
                <div>
                    <div style="font-weight: 700; color: #b45309; font-size: 13px; margin-bottom: 6px;">🚀 Priority Areas to Improve:</div>
                    <ul style="margin: 0 0 0 18px; padding: 0; font-size: 13px; color: #374151;">{improvements_html}</ul>
                </div>
            </div>
        </div>

        <!-- Question-by-Question Section -->
        <div style="padding: 24px;">
            <h2 style="margin: 0 0 16px 0; font-size: 16px; font-weight: 700; color: #111827;">Question Breakdown</h2>
            {questions_cards_html}
        </div>

        <!-- Footer -->
        <div style="background-color: #f9fafb; padding: 16px 24px; text-align: center; border-top: 1px solid #e5e7eb; font-size: 12px; color: #9ca3af;">
            Evaluated by UPSC Evaluator Engine • Calibrated against UPSC Civil Services Mains standard.
        </div>
    </div>
</body>
</html>
"""
        return html_body

    def send_evaluation_email(
        self,
        to_email: str,
        report: StudentEvaluationReport,
        job_id: str | None = None,
    ) -> bool:
        """
        Dispatches the evaluation report email to the student.
        Returns True if successful, False otherwise.
        """
        subject = f"[UPSCopilot] Your Evaluation Report - {report.document} (Score: {report.summary.total_score:g}/{report.summary.max_marks:g})"
        html_content = self.render_scorecard_html(report)

        if self.provider == "mock" or not self.from_email:
            logger.info("Mock EmailService: Dispatched email to %s (Subject: %s)", to_email, subject)
            # Write a local preview artifact for localhost review
            if job_id:
                try:
                    preview_dir = Path(settings.job_dir)
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    preview_path = preview_dir / f"email_preview_{job_id}.html"
                    preview_path.write_text(html_content, encoding="utf-8")
                    logger.info("Saved local email preview to %s", preview_path)
                except Exception as e:
                    logger.warning("Failed to save email preview: %s", e)
            return True

        if self.provider == "ses":
            return self._send_via_ses(to_email, subject, html_content)
        elif self.provider == "smtp":
            return self._send_via_smtp(to_email, subject, html_content)
        else:
            logger.warning("Unknown email provider '%s'; falling back to mock logging.", self.provider)
            return True

    def _send_via_ses(self, to_email: str, subject: str, html_body: str) -> bool:
        """Sends email using AWS Amazon SES."""
        try:
            client = boto3.client("ses", region_name=settings.aws_region)
            response = client.send_email(
                Source=self.from_email,
                Destination={"ToAddresses": [to_email]},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {"Html": {"Data": html_body, "Charset": "UTF-8"}},
                },
            )
            logger.info("Amazon SES email sent successfully to %s. MessageId: %s", to_email, response.get("MessageId"))
            return True
        except (BotoCoreError, ClientError) as e:
            logger.error("Amazon SES failed to send email to %s: %s", to_email, e)
            return False

    def _send_via_smtp(self, to_email: str, subject: str, html_body: str) -> bool:
        """Sends email using standard SMTP (e.g. Gmail / Resend)."""
        if not settings.smtp_host:
            logger.warning("SMTP host not configured. Falling back to log.")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = to_email
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
                server.starttls()
                if settings.smtp_user and settings.smtp_password:
                    server.login(settings.smtp_user, settings.smtp_password)
                server.sendmail(self.from_email, [to_email], msg.as_string())

            logger.info("SMTP email sent successfully to %s", to_email)
            return True
        except Exception as e:
            logger.error("SMTP failed to send email to %s: %s", to_email, e)
            return False
