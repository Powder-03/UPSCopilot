"""Pinecone vector store and Vertex AI embeddings integration."""
import logging
import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings

logger = logging.getLogger(__name__)


class DeterministicMockEmbeddings(Embeddings):
    """Fallback deterministic embedding generator for offline testing."""

    def __init__(self, dimension: int = 768):
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


class VertexEmbeddings(Embeddings):
    """Native Google Cloud Vertex AI text-embedding-004 generator."""

    def __init__(
        self,
        model_name: str = "text-embedding-004",
        dimension: int = 768,
        batch_size: int = 20,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True,
                project=settings.gcp_project_id,
                location=settings.gcp_location,
                api_key=settings.gemini_api_key,
            )
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embeds documents efficiently in safe chunks under Vertex AI 20k token limit."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t[:8000] for t in texts[i : i + self.batch_size]]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=chunk,
            )
            for emb in response.embeddings:
                all_embeddings.append(emb.values)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string."""
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return response.embeddings[0].values


def get_embedding_function(force_mock: bool = False) -> Embeddings:
    """Returns Vertex AI embeddings, Bedrock embeddings, or deterministic mock."""
    if force_mock:
        return DeterministicMockEmbeddings(dimension=settings.embedding_dimension)

    if settings.is_vertex_active and settings.gemini_api_key:
        try:
            return VertexEmbeddings(
                dimension=settings.embedding_dimension,
            )
        except Exception as e:
            logger.warning(f"Could not initialize VertexEmbeddings ({e}); using mock fallback.")
            return DeterministicMockEmbeddings(dimension=settings.embedding_dimension)

    if settings.has_aws_credentials:
        try:
            from langchain_aws import BedrockEmbeddings

            return BedrockEmbeddings(
                model_id=settings.bedrock_embedding_model_id,
                region_name=settings.aws_region,
            )
        except Exception as e:
            logger.warning(f"Could not initialize BedrockEmbeddings ({e}); using mock fallback.")
            return DeterministicMockEmbeddings(dimension=settings.embedding_dimension)

    return DeterministicMockEmbeddings(dimension=settings.embedding_dimension)


class MockVectorStore:
    """In-memory mock vector store for offline testing without cloud credentials."""

    def __init__(self, documents: list[Document] | None = None):
        self.documents: list[Document] = documents or []

    def similarity_search(
        self, query: str, k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[Document]:
        if not self.documents:
            try:
                from src.kb.corpus_loader import load_all_corpus_documents
                self.documents = load_all_corpus_documents()
            except Exception:
                pass
        return self.documents[:k]

    def add_documents(self, documents: list[Document]) -> list[str]:
        self.documents.extend(documents)
        return [doc.metadata.get("id", str(i)) for i, doc in enumerate(documents)]


class PineconeVectorStore:
    """Production vector store using Pinecone serverless index."""

    def __init__(
        self,
        index_name: str | None = None,
        embedding_function: Embeddings | None = None,
    ):
        self.index_name = index_name or settings.pinecone_index_name
        self.embedding_function = embedding_function or get_embedding_function()
        self._index = None

    @property
    def index(self):
        if self._index is None:
            from pinecone import Pinecone

            api_key = settings.pinecone_api_key or os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY is not set.")
            pc = Pinecone(api_key=api_key)
            self._index = pc.Index(self.index_name)
        return self._index

    def similarity_search(
        self, query: str, k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[Document]:
        """Queries Pinecone by embedding the query and retrieving top-k matches with optional metadata filter."""
        try:
            query_vector = self.embedding_function.embed_query(query)
            query_kwargs: dict[str, Any] = {
                "vector": query_vector,
                "top_k": k,
                "include_metadata": True,
            }
            if filter:
                query_kwargs["filter"] = filter
            results = self.index.query(**query_kwargs)
            matches = results.get("matches", []) if isinstance(results, dict) else results.matches
            documents: list[Document] = []
            for match in matches:
                metadata = dict(match.get("metadata", {})) if isinstance(match, dict) else dict(match.metadata or {})
                text = metadata.get("text", "")
                documents.append(Document(page_content=text, metadata=metadata))
            return documents
        except Exception as e:
            logger.warning(f"Pinecone similarity search error: {e}")
            return []

    def add_documents(self, documents: list[Document], batch_size: int = 50) -> list[str]:
        """Embeds and upserts documents into Pinecone in batches."""
        ids: list[str] = []
        for i in range(0, len(documents), batch_size):
            chunk = documents[i : i + batch_size]
            texts = [d.page_content for d in chunk]
            vectors = self.embedding_function.embed_documents(texts)

            records: list[dict[str, Any]] = []
            for doc, vec in zip(chunk, vectors, strict=False):
                doc_id = doc.metadata.get("id", f"doc_{len(ids)}")
                ids.append(doc_id)
                meta = dict(doc.metadata)
                meta["text"] = doc.page_content[:30000]  # Pinecone metadata size safeguard
                records.append({
                    "id": doc_id,
                    "values": vec,
                    "metadata": meta,
                })
            self.index.upsert(vectors=records)
        return ids


def get_vector_store(
    persist_dir: str | None = None, force_mock: bool = False
) -> PineconeVectorStore | MockVectorStore:
    """Returns PineconeVectorStore if credentials exist, or MockVectorStore in offline test mode."""
    api_key = settings.pinecone_api_key or os.getenv("PINECONE_API_KEY")
    if force_mock or not api_key:
        return MockVectorStore()

    try:
        return PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding_function=get_embedding_function(force_mock=force_mock),
        )
    except Exception as e:
        logger.warning(f"Failed to initialize PineconeVectorStore ({e}); falling back to mock.")
        return MockVectorStore()


# Backward-compatible alias for existing imports
get_chroma_vector_store = get_vector_store
