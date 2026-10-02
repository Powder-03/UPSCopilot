"""Unit tests for FastAPI endpoints: root, health check, job submission, and job polling."""
from unittest.mock import patch

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Verifies GET / returns 200 and service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "UPSCopilot" in data["service"]


def test_portal_endpoint():
    """Verifies GET /portal returns 200 with HTML portal content."""
    response = client.get("/portal")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "UPSCopilot" in response.text
    assert "Answer Booklet PDF" in response.text
    assert "Recipient Email Address" in response.text


def test_health_check_endpoint():
    """Verifies GET /api/v1/health returns 200 and system health indicators."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_provider" in data
    assert "eval_model" in data
    assert "vision_model" in data
    assert "gcp_configured" in data



def test_submit_job_valid_pdf():
    """Verifies that submitting a valid PDF returns HTTP 202 with job_id and check_status_url."""
    fake_pdf = b"%PDF-1.4 Fake PDF Content for Unit Test"

    # Patch dispatch_job so background processing doesn't try to invoke real Bedrock during fast unit test
    with patch("src.api.app.job_manager.dispatch_job") as mock_dispatch:
        response = client.post(
            "/api/v1/jobs/submit",
            files={"file": ("test_answer_booklet.pdf", fake_pdf, "application/pdf")},
            data={"email": "aspirant@upsc.test", "start_page": "3", "max_pages": "4"},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "QUEUED"
        assert data["email"] == "aspirant@upsc.test"
        assert f"/api/v1/jobs/{data['job_id']}" in data["check_status_url"]
        mock_dispatch.assert_called_once()

        # Check that GET /api/v1/jobs/{job_id} works immediately
        poll_resp = client.get(f"/api/v1/jobs/{data['job_id']}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        assert poll_data["job_id"] == data["job_id"]
        assert poll_data["status"] == "QUEUED"
        assert poll_data["email"] == "aspirant@upsc.test"


def test_submit_job_invalid_extension():
    """Verifies that uploading a non-PDF file returns HTTP 400."""
    fake_text = b"This is a text file, not a PDF."
    response = client.post(
        "/api/v1/jobs/submit",
        files={"file": ("notes.txt", fake_text, "text/plain")},
    )
    assert response.status_code == 400
    assert "Please upload a PDF file" in response.json()["detail"]


def test_submit_job_empty_file():
    """Verifies that uploading an empty PDF returns HTTP 400."""
    response = client.post(
        "/api/v1/jobs/submit",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "0 bytes" in response.json()["detail"]


def test_get_nonexistent_job():
    """Verifies that polling a non-existent job ID returns HTTP 404."""
    response = client.get("/api/v1/jobs/job_does_not_exist_999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
