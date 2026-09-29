"""Redirect for backward compatibility: use BedrockEmbeddings via src.kb.vector_store."""
from src.kb.vector_store import get_embedding_function

__all__ = ["get_embedding_function"]
