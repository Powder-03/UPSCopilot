"""Configuration management for AWS Bedrock API, vector store, and evaluation."""
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # AWS Bedrock API Configuration
    bedrock_api_key: str | None = Field(default=None, alias="BEDROCK_API_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    # Evaluation Model Configuration: Moonshot Kimi 2.5
    bedrock_eval_model_id: str = Field(
        default="moonshotai.kimi-k2.5", alias="BEDROCK_EVAL_MODEL_ID"
    )
    eval_temperature: float = Field(default=0.2, alias="EVAL_TEMPERATURE")
    eval_max_tokens: int = Field(default=4096, alias="EVAL_MAX_TOKENS")

    # Multimodal Vision Model Configuration (for handwriting and diagram OCR)
    bedrock_vision_model_id: str = Field(
        default="moonshotai.kimi-k2.5", alias="BEDROCK_VISION_MODEL_ID"
    )
    parsing_dpi: int = Field(default=200, alias="PARSING_DPI")
    parsing_max_workers: int = Field(default=4, alias="PARSING_MAX_WORKERS")

    # Bedrock Embedding Models (for Knowledge Base)
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0", alias="BEDROCK_EMBEDDING_MODEL_ID"
    )
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")

    # Optional IAM fallback if ever needed by embedding client
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: str | None = Field(default=None, alias="AWS_SESSION_TOKEN")

    # Storage paths (repo-root relative; data lives outside the importable package)
    kb_storage_dir: str = Field(default="data/storage", alias="KB_STORAGE_DIR")
    data_dir: str = Field(default="data", alias="DATA_DIR")
    upload_dir: str = Field(default="data/uploads", alias="UPLOAD_DIR")
    job_dir: str = Field(default="data/jobs", alias="JOB_DIR")

    # Email Delivery Configuration (SES, SMTP, or Mock)
    email_provider: str = Field(default="mock", alias="EMAIL_PROVIDER")
    ses_from_email: str = Field(default="evaluator@upscopilot.com", alias="SES_FROM_EMAIL")
    smtp_host: str = Field(default="", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")

    # Confident AI / DeepEval Key
    confident_api_key: str | None = Field(default=None, alias="CONFIDENT_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def has_aws_credentials(self) -> bool:
        """Returns True if Bedrock API key or AWS credentials exist."""
        return bool(
            self.bedrock_api_key
            or (self.aws_access_key_id and self.aws_secret_access_key)
            or os.environ.get("AWS_PROFILE")
            or (os.path.exists(os.path.expanduser("~/.aws/credentials")))
        )


settings = Settings()
