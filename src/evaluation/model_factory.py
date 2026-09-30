"""Factory for creating evaluation LLM clients (Moonshot Kimi 2.5 via AWS Bedrock Converse API)."""

import boto3
from langchain_aws import ChatBedrockConverse

from src.config import settings


def get_eval_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
    model_id: str | None = None,
) -> ChatBedrockConverse:
    """
    Returns Moonshot Kimi 2.5 ChatModel using AWS Bedrock Converse API.
    """
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    return ChatBedrockConverse(
        model_id=model_id or settings.bedrock_eval_model_id,
        client=client,
        region_name=settings.aws_region,
        temperature=temperature if temperature is not None else settings.eval_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.eval_max_tokens,
    )
