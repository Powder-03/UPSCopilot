"""LangChain Chroma vector store integration with AWS Bedrock embeddings."""
import os

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src.config import settings


class DeterministicMockEmbeddings(Embeddings):
    """Fallback embedding generator for offline testing when AWS credentials are not configured."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        import hashlib

        import numpy as np

        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**6)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension)
        return (vec / np.linalg.norm(vec)).tolist()


def get_embedding_function(force_mock: bool = False) -> Embeddings:
    """Returns BedrockEmbeddings if credentials exist, or DeterministicMockEmbeddings as fallback."""
    if force_mock or not settings.has_aws_credentials:
        return DeterministicMockEmbeddings(dimension=settings.embedding_dimension)

    return BedrockEmbeddings(
        model_id=settings.bedrock_embedding_model_id,
        region_name=settings.aws_region,
    )


def get_chroma_vector_store(
    persist_dir: str | None = None, force_mock: bool = False
) -> Chroma:
    """Returns a persistent Chroma vector store instance."""
    storage_path = persist_dir or os.path.join(settings.kb_storage_dir, "chroma_db")
    os.makedirs(storage_path, exist_ok=True)

    return Chroma(
        collection_name="upsc_knowledge_base",
        embedding_function=get_embedding_function(force_mock=force_mock),
        persist_directory=storage_path,
    )
