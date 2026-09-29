"""Factory for creating evaluation LLM clients (Moonshot Kimi 2.5 via AWS Bedrock Converse API)."""
from typing import Optional
from langchain_aws import ChatBedrockConverse
from src.config import settings


def get_eval_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model_id: Optional[str] = None,
) -> ChatBedrockConverse:
    """
    Returns Moonshot Kimi 2.5 ChatModel using AWS Bedrock API key.
    """
    return ChatBedrockConverse(
        model_id=model_id or settings.bedrock_eval_model_id,
        bedrock_api_key=settings.bedrock_api_key,
        region_name=settings.aws_region,
        temperature=temperature if temperature is not None else settings.eval_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.eval_max_tokens,
    )
