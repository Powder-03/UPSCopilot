"""AWS Lambda entrypoint for FastAPI application using Mangum ASGI adapter."""
import logging

from mangum import Mangum

from src.api.app import app

logger = logging.getLogger(__name__)

# Mangum adapter with lifespan="off" for low cold-start latency in AWS Lambda
handler = Mangum(app, lifespan="off")
