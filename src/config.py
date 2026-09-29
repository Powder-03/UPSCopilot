"""Configuration management for AWS Bedrock, vector store, and evaluation thresholds."""
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # AWS Credentials
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = Field(default=None, alias="AWS_SESSION_TOKEN")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    # Bedrock Models
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0", alias="BEDROCK_EMBEDDING_MODEL_ID"
    )
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")
    bedrock_eval_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20240620-v1:0", alias="BEDROCK_EVAL_MODEL_ID"
    )

    # Storage paths
    kb_storage_dir: str = Field(default="src/kb/storage", alias="KB_STORAGE_DIR")
    data_dir: str = Field(default="src/kb/data", alias="DATA_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def has_aws_credentials(self) -> bool:
        """Returns True if AWS credentials or environment profiles are configured."""
        return bool(
            (self.aws_access_key_id and self.aws_secret_access_key)
            or os.environ.get("AWS_PROFILE")
            or (os.path.exists(os.path.expanduser("~/.aws/credentials")))
        )


settings = Settings()
