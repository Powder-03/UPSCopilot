"""Configuration management for AWS Bedrock API, vector store, and evaluation."""
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Primary LLM Provider: 'vertex' (Gemini 2.5 Flash via Vertex AI) or 'bedrock' (Moonshot Kimi 2.5)
    llm_provider: str = Field(default="vertex", alias="LLM_PROVIDER")

    # Vertex AI / Google Cloud Configuration
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gcp_project_id: str = Field(default="project-1b52589d-0827-46ab-9be", alias="GCP_PROJECT_ID")
    gcp_location: str = Field(default="us-central1", alias="GCP_LOCATION")
    vertex_eval_model_id: str = Field(default="gemini-2.5-flash", alias="VERTEX_EVAL_MODEL_ID")
    vertex_vision_model_id: str = Field(default="gemini-2.5-flash", alias="VERTEX_VISION_MODEL_ID")

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

    # Embedding Configuration (Vertex AI text-embedding-004 = 768, Titan = 1024)
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0", alias="BEDROCK_EMBEDDING_MODEL_ID"
    )
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION")

    # Pinecone Vector Store Configuration
    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="upsc-kb", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")

    # Optional IAM fallback if ever needed by embedding client
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: str | None = Field(default=None, alias="AWS_SESSION_TOKEN")

    # Storage paths (repo-root relative; data lives outside the importable package)
    kb_storage_dir: str = Field(default="data/storage", alias="KB_STORAGE_DIR")
    data_dir: str = Field(default="data", alias="DATA_DIR")
    upload_dir: str = Field(default="data/uploads", alias="UPLOAD_DIR")
    job_dir: str = Field(default="data/jobs", alias="JOB_DIR")

    # AWS Serverless & Cloud Storage Configuration (Lambda, S3, DynamoDB, SQS)
    s3_bucket: str | None = Field(default=None, alias="S3_BUCKET")
    dynamodb_table: str | None = Field(default=None, alias="DYNAMODB_TABLE")
    vision_queue_url: str | None = Field(default=None, alias="VISION_QUEUE_URL")
    eval_queue_url: str | None = Field(default=None, alias="EVAL_QUEUE_URL")
    parsed_booklets_table: str | None = Field(default=None, alias="PARSED_BOOKLETS_TABLE")

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
    def is_vertex_active(self) -> bool:
        """Returns True if the primary provider is Google Cloud Vertex AI / Gemini."""
        return self.llm_provider.strip().lower() in ("vertex", "gemini", "google")

    @property
    def is_bedrock_active(self) -> bool:
        """Returns True if the primary provider is AWS Bedrock."""
        return self.llm_provider.strip().lower() in ("bedrock", "aws")

    @property
    def active_eval_model_id(self) -> str:
        """Returns the model ID for evaluation based on the active provider."""
        return self.vertex_eval_model_id if self.is_vertex_active else self.bedrock_eval_model_id

    @property
    def active_vision_model_id(self) -> str:
        """Returns the model ID for vision parsing based on the active provider."""
        return self.vertex_vision_model_id if self.is_vertex_active else self.bedrock_vision_model_id

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

