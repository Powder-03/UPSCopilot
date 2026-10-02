"""Unit tests for AWS Lambda entrypoint handlers (API Mangum & SQS Worker)."""
import json
from unittest.mock import patch

from src.lambda_worker import handler as worker_handler


def test_lambda_worker_sqs_event():
    """Verifies that the SQS Lambda worker parses SQS records and invokes evaluation."""
    sqs_event = {
        "Records": [
            {
                "messageId": "msg_001",
                "body": json.dumps(
                    {
                        "job_id": "job_lambda_test_1",
                        "pdf_path": "s3://eval-bucket/uploads/job_lambda_test_1_copy.pdf",
                        "start_page": 2,
                        "max_pages": 4,
                    }
                ),
            }
        ]
    }

    with patch("src.lambda_worker.job_manager._run_job_worker") as mock_worker:
        resp = worker_handler(sqs_event)
        assert resp == {"batchItemFailures": []}
        mock_worker.assert_called_once_with(
            job_id="job_lambda_test_1",
            pdf_path="s3://eval-bucket/uploads/job_lambda_test_1_copy.pdf",
            start_page=2,
            max_pages=4,
        )


def test_lambda_worker_direct_invocation():
    """Verifies direct invocation of the Lambda worker."""
    direct_event = {
        "job_id": "job_direct_1",
        "pdf_path": "data/uploads/copy.pdf",
        "start_page": 1,
    }

    with patch("src.lambda_worker.job_manager._run_job_worker") as mock_worker:
        resp = worker_handler(direct_event)
        assert resp == {"status": "success", "job_id": "job_direct_1"}
        mock_worker.assert_called_once_with(
            job_id="job_direct_1",
            pdf_path="data/uploads/copy.pdf",
            start_page=1,
            max_pages=None,
        )


def test_lambda_api_handler_health():
    """Verifies the Mangum ASGI API handler returns health check."""
    from src.lambda_api import handler as api_handler

    # API Gateway HTTP API v2 payload format
    event = {
        "version": "2.0",
        "routeKey": "GET /api/v1/health",
        "rawPath": "/api/v1/health",
        "rawQueryString": "",
        "headers": {
            "accept": "application/json",
            "host": "localhost",
        },
        "requestContext": {
            "http": {
                "method": "GET",
                "path": "/api/v1/health",
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "pytest",
            },
        },
        "isBase64Encoded": False,
    }

    response = api_handler(event, {})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["status"] == "healthy"
