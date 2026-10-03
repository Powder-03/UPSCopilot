"""Factory for creating evaluation LLM clients (Google Cloud Vertex AI Gemini or AWS Bedrock Moonshot Kimi)."""

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings
from src.providers.vertex import ChatVertexExpress


def get_eval_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
    model_id: str | None = None,
    provider: str | None = None,
) -> BaseChatModel:
    """
    Returns an evaluation ChatModel based on the active provider (Vertex AI Gemini or AWS Bedrock Kimi).
    """
    active_provider = (provider or settings.llm_provider).strip().lower()

    if active_provider in ("vertex", "gemini", "google"):
        return ChatVertexExpress(
            model_name=model_id or settings.vertex_eval_model_id,
            project_id=settings.gcp_project_id,
            location=settings.gcp_location,
            api_key=settings.gemini_api_key or "",
            temperature=temperature if temperature is not None else settings.eval_temperature,
            max_tokens=max_tokens if max_tokens is not None else settings.eval_max_tokens,
        )

    # AWS Bedrock Converse API fallback
    import boto3
    from langchain_aws import ChatBedrockConverse

    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    return ChatBedrockConverse(
        model_id=model_id or settings.bedrock_eval_model_id,
        client=client,
        region_name=settings.aws_region,
        temperature=temperature if temperature is not None else settings.eval_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.eval_max_tokens,
    )

