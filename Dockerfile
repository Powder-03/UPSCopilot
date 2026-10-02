# Multi-purpose AWS Lambda Container Image for UPSC Evaluator
# Uses official AWS Lambda Python 3.11 image (Amazon Linux 2023)
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for PyMuPDF, Pillow, and ONNX Runtime
RUN yum install -y \
    gcc \
    gcc-c++ \
    mesa-libGL \
    libXext \
    libSM \
    libXrender \
    tar \
    gzip \
    zlib-devel \
    libjpeg-turbo-devel \
    freetype-devel \
    libpng-devel \
    && yum clean all

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    KB_STORAGE_DIR=/var/task/data/storage \
    DATA_DIR=/var/task/data \
    UPLOAD_DIR=/tmp/uploads \
    JOB_DIR=/tmp/jobs

# Copy project definition
WORKDIR ${LAMBDA_TASK_ROOT}

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml README.md ./

# Install dependencies into system Python using uv pip (strictly binary wheels, no source compilation)
RUN uv pip install --system --only-binary :all: -r pyproject.toml

# Copy application source code and constitution raw corpus
COPY src/ ./src/
COPY data/raw/ ./data/raw/

# Create ephemeral directories for runtime /tmp usage
RUN mkdir -p /tmp/uploads /tmp/jobs

# Default handler: API Gateway Lambda entrypoint
# Overridden via ImageConfig.Command in template.yaml for decoupled workers:
# - Stage 1 Vision OCR: src.lambda_vision.handler
# - Stage 2 Evaluation: src.lambda_eval.handler
CMD [ "src.lambda_api.handler" ]
