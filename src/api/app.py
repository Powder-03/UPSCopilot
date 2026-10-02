import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.models.api import JobStatusResponse, JobSubmitResponse
from src.services.job_manager import job_manager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

INDEX_HTML_PATH = STATIC_DIR / "index.html"
if not INDEX_HTML_PATH.exists():
    INDEX_HTML_PATH = Path(__file__).resolve().parents[1] / "index.html"


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

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")



@app.get("/", tags=["General"])
async def root(request: Request):
    """
    Root landing endpoint. Serves the upload portal for browser visitors,
    and JSON service metadata for API clients.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept and INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))

    return {
        "service": "UPSCopilot Evaluation API",
        "status": "online",
        "docs_url": "/docs",
        "portal_url": "/portal",
        "submit_endpoint": "/api/v1/jobs/submit",
    }


@app.get("/portal", tags=["General"], response_class=HTMLResponse)
async def portal_page():
    """Dedicated web portal endpoint for answer booklet upload and email delivery."""
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Upload portal HTML template not found.")



@app.get("/api/v1/health", tags=["Health"])
async def health_check() -> dict[str, object]:
    """Liveness, readiness, and provider configuration health check."""
    return {
        "status": "healthy",
        "llm_provider": settings.llm_provider,
        "eval_model": settings.active_eval_model_id,
        "vision_model": settings.active_vision_model_id,
        "gcp_configured": bool(settings.gemini_api_key),
        "aws_credentials_available": settings.has_aws_credentials,
        "aws_region": settings.aws_region,
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
