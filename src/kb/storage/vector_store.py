"""Redirect for backward compatibility: use src.kb.vector_store instead."""
from src.kb.vector_store import get_chroma_vector_store

__all__ = ["get_chroma_vector_store"]
