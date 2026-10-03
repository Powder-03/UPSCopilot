"""Unit tests for AWS Lambda API entrypoint handler (Mangum)."""
import json

from src.handlers.api import handler as api_handler


def test_lambda_api_handler_health():
    """Verifies the Mangum ASGI API handler returns health check."""
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
