"""FastAPI application for UPSC answer booklet evaluation with asynchronous job submission and email delivery."""
import logging
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.models.api import JobStatusResponse, JobSubmitResponse
from src.services.job_manager import job_manager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="UPSCopilot Answer Evaluation API",
    version="1.0.0",
    description="Calibrated, grounded UPSC Civil Services Mains answer evaluation engine with zero performance bands.",
)

# Enable CORS for cross-origin frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root() -> dict[str, str]:
    """Root landing endpoint."""
    return {
        "service": "UPSCopilot Evaluation API",
        "status": "online",
        "docs_url": "/docs",
        "submit_endpoint": "/api/v1/jobs/submit",
    }


@app.get("/api/v1/health", tags=["Health"])
async def health_check() -> dict[str, object]:
    """Liveness, readiness, and AWS configuration health check."""
    return {
        "status": "healthy",
        "aws_credentials_available": settings.has_aws_credentials,
        "aws_region": settings.aws_region,
        "eval_model": settings.bedrock_eval_model_id,
        "vision_model": settings.bedrock_vision_model_id,
        "embedding_model": settings.bedrock_embedding_model_id,
        "email_provider": settings.email_provider,
    }


@app.post(
    "/api/v1/jobs/submit",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Evaluation Jobs"],
    summary="Submit answer booklet PDF for asynchronous evaluation",
)
async def submit_evaluation_job(
    file: Annotated[UploadFile, File(description="Uploaded UPSC answer booklet PDF")],
    email: Annotated[str | None, Form(description="Student email address for automatic scorecard delivery")] = None,
    start_page: Annotated[int | None, Form(description="1-indexed starting page for Question 1 (e.g. 3)")] = None,
    max_pages: Annotated[int | None, Form(description="Maximum pages to process for partial evaluation")] = None,
) -> JobSubmitResponse:
    """
    Submits a handwritten answer booklet PDF for asynchronous evaluation.
    - Validates and saves the PDF.
    - Queues the background parsing, evaluation, and distillation pipeline.
    - If email is provided, dispatches the scorecard email upon completion.
    - Returns immediately with job_id and check_status_url (<500ms).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a PDF file (.pdf).",
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        submit_resp, pdf_path = job_manager.create_job(
            file_bytes=content,
            filename=file.filename,
            email=email,
            start_page=start_page,
            max_pages=max_pages,
        )

        # Dispatch background worker
        job_manager.dispatch_job(
            job_id=submit_resp.job_id,
            pdf_path=pdf_path,
            start_page=start_page,
            max_pages=max_pages,
        )

        return submit_resp

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to queue evaluation job: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue evaluation job: {str(e)}",
        ) from e


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Evaluation Jobs"],
    summary="Check evaluation job progress and retrieve student scorecard",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Retrieves the real-time status of an evaluation job.
    When status is COMPLETED, the full student scorecard is present in the `result` field.
    """
    job_resp = job_manager.get_job(job_id)
    if not job_resp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job_resp
