"""Embedding providers for knowledge base ingestion and retrieval."""
from src.kb.embeddings.bedrock import BedrockEmbeddingProvider

__all__ = ["BedrockEmbeddingProvider"]
