"""Unit tests for AWS Lambda API entrypoint handler (Mangum)."""
import asyncio
import json

from src.handlers.api import handler as api_handler


def test_lambda_api_handler_health():
    """Verifies the Mangum ASGI API handler returns health check."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
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
    finally:
        loop.close()


def test_single_question_eval_handler_blank():
    """Verifies that an unattempted question is evaluated with authentic 0.0 marks instantly."""
    from src.handlers.single_question_eval import handler as single_eval_handler

    event = {
        "q_num": 5,
        "question": "Discuss the mandate of the Lokpal.",
        "candidate_answer": "",
        "max_marks": 10.0,
        "kb_context": ["[Lokpal Act 2013]\nSection 14: Jurisdiction of Lokpal."],
    }

    result = single_eval_handler(event, {})
    assert result["status"] == "success"
    assert result["q_num"] == 5
    assert result["max_marks"] == 10.0
    eval_data = result["evaluation"]
    assert eval_data["total_score"] == 0.0
    assert eval_data["percentage"] == 0.0
    assert "Needs Foundation" in eval_data["performance_band"]

