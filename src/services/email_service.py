"""Email delivery service for UPSC evaluation scorecards.

Supports Amazon SES (boto3 send_raw_email with PDF attachment), standard SMTP,
and mock/console preview for local testing.
Generates responsive, student-centric HTML scorecards with zero performance bands.
"""
import html
import logging
import smtplib
from email import encoders
from email.mime.application import MIMEApplication
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
        from_name: str | None = None,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
    ):
        raw_provider = (provider or settings.email_provider).strip().lower()
        if raw_provider in ("google", "gmail", "smtp"):
            self.provider = "google"
        elif raw_provider == "ses":
            self.provider = "ses"
        else:
            self.provider = "mock"

        self.from_name = from_name if from_name is not None else settings.mail_from_name
        self.from_email = from_email or settings.default_from_email
        self.smtp_host = smtp_host or (settings.smtp_host or "smtp.gmail.com")
        self.smtp_port = smtp_port or settings.smtp_port
        self.smtp_user = smtp_user or settings.smtp_user
        self.smtp_password = smtp_password or settings.smtp_password

    def render_scorecard_text(self, report: StudentEvaluationReport) -> str:
        """Builds a concise plain-text scorecard email with marks and 1-2 line summary."""
        total_score = report.summary.total_score
        max_marks = report.summary.max_marks
        pct = report.summary.percentage

        strengths = report.summary.overall_feedback.key_strengths[:2]
        strengths_txt = "\n".join(f"• {s}" for s in strengths) if strengths else "• Solid core attempt across answered questions."

        improvements = report.summary.overall_feedback.top_areas_to_improve[:2]
        improvements_txt = "\n".join(f"• {i}" for i in improvements) if improvements else "• Refer to attached PDF scorecard for actionable suggestions."

        return (
            f"UPSC Mains Evaluation Result\n"
            f"Document: {report.document}\n\n"
            f"Total Marks: {total_score:g} / {max_marks:g} ({pct:.1f}%)\n\n"
            f"What You Did Good:\n{strengths_txt}\n\n"
            f"What Could Be Better:\n{improvements_txt}\n\n"
            f"Detailed question-by-question evaluation and diagnostic scorecard are attached as a PDF.\n"
        )

    def render_scorecard_html(self, report: StudentEvaluationReport) -> str:
        """Builds a concise, executive HTML scorecard email with marks and 1-2 line summary."""
        doc_name = html.escape(report.document)
        total_score = report.summary.total_score
        max_marks = report.summary.max_marks
        pct = report.summary.percentage

        # Strengths: strictly 1-2 high-impact lines
        strengths = report.summary.overall_feedback.key_strengths[:2]
        strengths_html = "".join(
            f"<li style='margin-bottom: 6px;'>{html.escape(s)}</li>"
            for s in strengths
        )
        if not strengths_html:
            strengths_html = "<li>Solid core attempt across answered questions.</li>"

        # Improvements: strictly 1-2 high-impact lines
        improvements = report.summary.overall_feedback.top_areas_to_improve[:2]
        improvements_html = "".join(
            f"<li style='margin-bottom: 6px;'>{html.escape(i)}</li>"
            for i in improvements
        )
        if not improvements_html:
            improvements_html = "<li>Refer to attached PDF scorecard for actionable suggestions.</li>"

        html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UPSC Mains Evaluation Result</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f3f4f6; margin: 0; padding: 24px 16px; color: #111827;">
    <div style="max-width: 580px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; border: 1px solid #e5e7eb; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);">
        
        <!-- Header Banner -->
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: #ffffff; padding: 24px 20px; text-align: center;">
            <h1 style="margin: 0; font-size: 20px; font-weight: 800; letter-spacing: -0.5px;">UPSC Mains Evaluation Result</h1>
            <p style="margin: 6px 0 0 0; font-size: 13px; opacity: 0.9;">Document: {doc_name}</p>
        </div>

        <!-- Score Summary Card -->
        <div style="padding: 24px; text-align: center; border-bottom: 1px solid #e5e7eb; background: #fafafa;">
            <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #6b7280; font-weight: 600;">Total Marks</div>
            <div style="font-size: 36px; font-weight: 800; color: #1e3a8a; margin: 4px 0;">
                {total_score:g} <span style="font-size: 20px; color: #9ca3af; font-weight: 500;">/ {max_marks:g}</span>
            </div>
            <div style="font-size: 14px; font-weight: 600; color: #374151;">Overall Percentage: {pct:.1f}%</div>
        </div>

        <!-- Highlights: What went good & What could be better (Strictly 1-2 lines) -->
        <div style="padding: 20px 24px;">
            <div style="margin-bottom: 16px;">
                <div style="font-weight: 700; color: #15803d; font-size: 14px; margin-bottom: 6px;">✓ What You Did Good:</div>
                <ul style="margin: 0 0 0 20px; padding: 0; font-size: 13.5px; color: #374151; line-height: 1.5;">
                    {strengths_html}
                </ul>
            </div>
            <div>
                <div style="font-weight: 700; color: #b45309; font-size: 14px; margin-bottom: 6px;">▲ What Could Be Better:</div>
                <ul style="margin: 0 0 0 20px; padding: 0; font-size: 13.5px; color: #374151; line-height: 1.5;">
                    {improvements_html}
                </ul>
            </div>
        </div>

        <!-- Attached PDF Banner -->
        <div style="margin: 0 24px 20px 24px; padding: 14px 16px; background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; text-align: center;">
            <div style="font-size: 13.5px; font-weight: 700; color: #1e40af; margin-bottom: 4px;">
                📎 Detailed Scorecard Attached
            </div>
            <div style="font-size: 12.5px; color: #3b82f6;">
                Your complete question-by-question evaluation and diagnostic scorecard are attached as a PDF.
            </div>
        </div>

        <!-- Footer -->
        <div style="background-color: #f9fafb; padding: 14px 24px; text-align: center; border-top: 1px solid #e5e7eb; font-size: 11.5px; color: #9ca3af;">
            Evaluated by UPSC Evaluator Engine • Calibrated against UPSC Civil Services Mains standard.
        </div>
    </div>
</body>
</html>
"""
        return html_body

    # ------------------------------------------------------------------
    # Public API: Send with optional PDF attachment
    # ------------------------------------------------------------------

    def send_evaluation_email(
        self,
        to_email: str,
        report: StudentEvaluationReport,
        job_id: str | None = None,
        pdf_bytes: bytes | None = None,
    ) -> bool:
        """
        Dispatches the evaluation report email to the student.
        If ``pdf_bytes`` is provided, attaches the scorecard PDF to the email.
        Returns True if successful, False otherwise.
        """
        subject = f"[UPSCopilot] Your Evaluation Report - {report.document} (Score: {report.summary.total_score:g}/{report.summary.max_marks:g})"
        html_content = self.render_scorecard_html(report)
        text_content = self.render_scorecard_text(report)

        if self.provider == "mock" or not self.from_email:
            logger.info("Mock EmailService: Dispatched email to %s (Subject: %s)", to_email, subject)
            # Write local preview artifacts for localhost review
            if job_id:
                try:
                    preview_dir = Path(settings.job_dir)
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    preview_path = preview_dir / f"email_preview_{job_id}.html"
                    preview_path.write_text(html_content, encoding="utf-8")
                    preview_txt_path = preview_dir / f"email_preview_{job_id}.txt"
                    preview_txt_path.write_text(text_content, encoding="utf-8")
                    logger.info("Saved local email previews to %s", preview_dir)
                    if pdf_bytes:
                        pdf_path = preview_dir / f"scorecard_{job_id}.pdf"
                        pdf_path.write_bytes(pdf_bytes)
                        logger.info("Saved local scorecard PDF to %s", pdf_path)
                except Exception as e:
                    logger.warning("Failed to save email preview: %s", e)
            return True

        if self.provider in ("google", "gmail", "smtp"):
            return self._send_via_google_smtp(
                to_email, subject, html_content, text_body=text_content, pdf_bytes=pdf_bytes, job_id=job_id
            )
        elif self.provider == "ses":
            return self._send_via_ses(
                to_email, subject, html_content, text_body=text_content, pdf_bytes=pdf_bytes, job_id=job_id
            )
        else:
            logger.warning("Unknown email provider '%s'; falling back to mock logging.", self.provider)
            return True

    def _build_mime_message(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str | None = None,
        pdf_bytes: bytes | None = None,
        job_id: str | None = None,
    ) -> MIMEMultipart:
        """Constructs a MIMEMultipart message with HTML body, plain text alternative, and optional PDF attachment."""
        if pdf_bytes:
            msg = MIMEMultipart("mixed")
            body_container = MIMEMultipart("alternative")
            msg.attach(body_container)
        else:
            msg = MIMEMultipart("alternative")
            body_container = msg

        msg["Subject"] = subject
        if self.from_name and "<" not in self.from_email:
            msg["From"] = f"{self.from_name} <{self.from_email}>"
        else:
            msg["From"] = self.from_email
        msg["To"] = to_email

        if text_body:
            body_container.attach(MIMEText(text_body, "plain", "utf-8"))
        body_container.attach(MIMEText(html_body, "html", "utf-8"))

        if pdf_bytes:
            pdf_part = MIMEApplication(pdf_bytes, _subtype="pdf")
            filename = f"UPSC_Scorecard_{job_id}.pdf" if job_id else "UPSC_Evaluation_Scorecard.pdf"
            pdf_part.add_header(
                "Content-Disposition",
                "attachment",
                filename=filename,
            )
            encoders.encode_base64(pdf_part)
            msg.attach(pdf_part)

        return msg

    def _send_via_google_smtp(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str | None = None,
        pdf_bytes: bytes | None = None,
        job_id: str | None = None,
    ) -> bool:
        """Sends email using Google SMTP (smtp.gmail.com) with SSL or STARTTLS and retry logic."""
        import time

        host = self.smtp_host or "smtp.gmail.com"
        port = self.smtp_port or 587
        envelope_from = self.smtp_user or self.from_email

        if not envelope_from:
            logger.warning("No sender email or SMTP_USER configured for Google SMTP. Falling back to log.")
            return False

        msg = self._build_mime_message(
            to_email, subject, html_body, text_body=text_body, pdf_bytes=pdf_bytes, job_id=job_id
        )

        for attempt in range(1, 4):
            try:
                if port == 465:
                    with smtplib.SMTP_SSL(host, port, timeout=20) as server:
                        if self.smtp_user and self.smtp_password:
                            server.login(self.smtp_user, self.smtp_password)
                        server.sendmail(envelope_from, [to_email], msg.as_string())
                else:
                    with smtplib.SMTP(host, port, timeout=20) as server:
                        server.ehlo()
                        server.starttls()
                        server.ehlo()
                        if self.smtp_user and self.smtp_password:
                            server.login(self.smtp_user, self.smtp_password)
                        server.sendmail(envelope_from, [to_email], msg.as_string())

                logger.info("Google SMTP email sent successfully to %s via %s:%d", to_email, host, port)
                return True
            except smtplib.SMTPAuthenticationError as auth_err:
                logger.error(
                    "Google SMTP authentication failed for user '%s': %s. "
                    "Make sure to use a 16-character Google App Password (not your standard Gmail login password).",
                    self.smtp_user,
                    auth_err,
                )
                return False
            except Exception as e:
                logger.warning(
                    "Google SMTP delivery attempt %d/3 to %s failed: %s",
                    attempt,
                    to_email,
                    e,
                )
                if attempt < 3:
                    time.sleep(2.0)
                else:
                    logger.error("All Google SMTP delivery attempts failed to %s: %s", to_email, e)
                    return False
        return False

    def _send_via_smtp(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str | None = None,
        pdf_bytes: bytes | None = None,
        job_id: str | None = None,
    ) -> bool:
        """Alias for _send_via_google_smtp for backward compatibility."""
        return self._send_via_google_smtp(
            to_email, subject, html_body, text_body=text_body, pdf_bytes=pdf_bytes, job_id=job_id
        )

    def _send_via_ses(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str | None = None,
        pdf_bytes: bytes | None = None,
        job_id: str | None = None,
    ) -> bool:
        """Sends email using AWS Amazon SES send_raw_email with optional PDF attachment."""
        try:
            client = boto3.client("ses", region_name=settings.aws_region)

            if pdf_bytes:
                # send_raw_email for MIME attachment
                msg = self._build_mime_message(
                    to_email, subject, html_body, text_body=text_body, pdf_bytes=pdf_bytes, job_id=job_id
                )
                response = client.send_raw_email(
                    Source=self.from_email,
                    Destinations=[to_email],
                    RawMessage={"Data": msg.as_string()},
                )
            else:
                # Simple HTML/text email
                body_dict = {"Html": {"Data": html_body, "Charset": "UTF-8"}}
                if text_body:
                    body_dict["Text"] = {"Data": text_body, "Charset": "UTF-8"}
                response = client.send_email(
                    Source=self.from_email,
                    Destination={"ToAddresses": [to_email]},
                    Message={
                        "Subject": {"Data": subject, "Charset": "UTF-8"},
                        "Body": body_dict,
                    },
                )

            logger.info("Amazon SES email sent successfully to %s. MessageId: %s", to_email, response.get("MessageId"))
            return True
        except (BotoCoreError, ClientError) as e:
            logger.error("Amazon SES failed to send email to %s: %s", to_email, e)
            return False
