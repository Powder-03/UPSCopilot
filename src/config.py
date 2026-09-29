"""Configuration management for AWS Bedrock API, vector store, and evaluation."""
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # AWS Bedrock API Configuration
    bedrock_api_key: Optional[str] = Field(default=None, alias="BEDROCK_API_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    # Evaluation Model Configuration: Moonshot Kimi 2.5
    bedrock_eval_model_id: str = Field(
        default="moonshotai.kimi-k2.5", alias="BEDROCK_EVAL_MODEL_ID"
    )
    eval_temperature: float = Field(default=0.2, alias="EVAL_TEMPERATURE")
    eval_max_tokens: int = Field(default=4096, alias="EVAL_MAX_TOKENS")

    # Bedrock Embedding Models (for Knowledge Base)
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0", alias="BEDROCK_EMBEDDING_MODEL_ID"
    )
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")

    # Optional IAM fallback if ever needed by embedding client
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = Field(default=None, alias="AWS_SESSION_TOKEN")

    # Storage paths
    kb_storage_dir: str = Field(default="src/kb/storage", alias="KB_STORAGE_DIR")
    data_dir: str = Field(default="src/kb/data", alias="DATA_DIR")

    # Confident AI / DeepEval Key
    confident_api_key: Optional[str] = Field(default=None, alias="CONFIDENT_API_KEY")

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
