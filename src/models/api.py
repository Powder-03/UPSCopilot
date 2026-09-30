"""API schemas and student-facing scorecard models for UPSC answer copy evaluation.

Strictly student-first: Zero performance bands, zero internal CoT or G-Eval logprob tokens.
Only marks, pros, and actionable what to do better.
"""
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    """Lifecycle states of an asynchronous evaluation job."""

    QUEUED = "QUEUED"
    PARSING = "PARSING"
    EVALUATING = "EVALUATING"
    DISTILLING = "DISTILLING"
    SENDING_EMAIL = "SENDING_EMAIL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StudentQuestionEvaluation(BaseModel):
    """Clean, actionable evaluation result for a single question."""

    model_config = ConfigDict(extra="forbid")

    q_num: int = Field(description="Question number (1 to 20)")
    max_marks: float = Field(description="Maximum marks for this question (10.0 or 15.0)")
    score: float = Field(description="Marks awarded to the candidate")
    percentage: float = Field(description="Percentage score achieved (score / max_marks * 100)")
    question: str = Field(description="Question text")
    pros: list[str] = Field(
        default_factory=list,
        description="Specific strengths and well-executed elements of the candidate's answer",
    )
    what_to_do_better: list[str] = Field(
        default_factory=list,
        description="Concrete, actionable recommendations and missed anchors to gain extra marks",
    )


class OverallFeedback(BaseModel):
    """High-level summary feedback across the entire answer booklet."""

    model_config = ConfigDict(extra="forbid")

    key_strengths: list[str] = Field(
        default_factory=list,
        description="Top overall strengths demonstrated across the paper",
    )
    top_areas_to_improve: list[str] = Field(
        default_factory=list,
        description="High-leverage strategic improvements needed across answers",
    )


class StudentSummary(BaseModel):
    """Overall summary statistics and feedback for the evaluated booklet."""

    model_config = ConfigDict(extra="forbid")

    total_score: float = Field(description="Total marks awarded across all attempted questions")
    max_marks: float = Field(description="Total maximum marks possible for the evaluated questions")
    percentage: float = Field(description="Overall paper score percentage")
    overall_feedback: OverallFeedback = Field(
        description="Synthesized strengths and priority improvements"
    )


class StudentEvaluationReport(BaseModel):
    """The clean student-facing evaluation report."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="success", description="Evaluation status")
    document: str = Field(description="Name or path of the evaluated document")
    summary: StudentSummary = Field(description="Paper-level marks and summary feedback")
    questions: list[StudentQuestionEvaluation] = Field(
        description="Question-by-question evaluation results"
    )


class JobSubmitResponse(BaseModel):
    """Immediate HTTP 202 response returned upon job submission."""

    job_id: str = Field(description="Unique identifier for the evaluation job")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Initial job status")
    check_status_url: str = Field(description="Relative URL to check job progress and result")
    email: str | None = Field(default=None, description="Registered notification email, if provided")
    message: str = Field(description="Human-friendly status message")


class JobStatusResponse(BaseModel):
    """Response returned when polling job status."""

    job_id: str = Field(description="Unique identifier for the evaluation job")
    status: JobStatus = Field(description="Current status of the job")
    progress_pct: int = Field(default=0, ge=0, le=100, description="Completion percentage (0-100)")
    current_step: str = Field(default="", description="Description of the active processing step")
    email: str | None = Field(default=None, description="Registered notification email, if provided")
    email_sent: bool = Field(default=False, description="Whether the scorecard email was dispatched")
    error: str | None = Field(default=None, description="Error message if the job failed")
    result: StudentEvaluationReport | None = Field(
        default=None, description="Complete student scorecard when status is COMPLETED"
    )
