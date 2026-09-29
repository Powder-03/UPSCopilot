"""Hybrid Ensemble Retriever combining BM25 and Chroma Vector Store with Bedrock Titan Embeddings."""
import re
import logging
from typing import List, Optional, Dict
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.config import settings
from src.kb.vector_store import get_chroma_vector_store
from src.kb.corpus_loader import load_all_corpus_documents

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 search."""
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """
    Standard LangChain Hybrid Retriever combining:
    1. Sparse Lexical Search (BM25 for exact Article numbers, act sections, case titles)
    2. Dense Semantic Search (Chroma vector store with Amazon Titan Bedrock embeddings)
    3. Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        persist_dir: Optional[str] = None,
        force_mock: bool = False,
        top_k: int = 5,
    ):
        self.persist_dir = persist_dir
        self.force_mock = force_mock
        self.top_k = top_k
        self.documents: List[Document] = documents if documents is not None else load_all_corpus_documents()
        self.doc_by_id: Dict[str, Document] = {
            doc.metadata.get("id", f"doc_{i}"): doc for i, doc in enumerate(self.documents)
        }

        # Initialize BM25 over the real documents
        self.corpus = [f"{doc.metadata.get('title', '')} {doc.page_content}" for doc in self.documents]
        self.tokenized_corpus = [tokenize(t) for t in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

        # Chroma vector store
        self.vector_store = get_chroma_vector_store(persist_dir, force_mock=force_mock)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Hybrid search with Reciprocal Rank Fusion (RRF)."""
        k = top_k or self.top_k
        query_tokens = tokenize(query)
        rrf_scores: Dict[int, float] = {}

        # 1. BM25 scoring
        if self.bm25 and query_tokens:
            bm25_scores = self.bm25.get_scores(query_tokens)
            sorted_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            for rank, idx in enumerate(sorted_bm25[:20]):
                if bm25_scores[idx] > 0.0:
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (60 + rank + 1))

        # 2. Dense vector search
        try:
            dense_docs = self.vector_store.similarity_search(query, k=min(20, len(self.documents)))
            for rank, doc in enumerate(dense_docs):
                doc_id = doc.metadata.get("id")
                for idx, d in enumerate(self.documents):
                    if d.metadata.get("id") == doc_id:
                        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (60 + rank + 1))
                        break
        except Exception:
            pass

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        retrieved_docs = [self.documents[idx] for idx, _ in sorted_indices[:k]]

        # Fallback if no hybrid matches
        if not retrieved_docs:
            try:
                retrieved_docs = self.vector_store.similarity_search(query, k=k)
            except Exception:
                retrieved_docs = []

        return retrieved_docs

    def get_retrieval_context(self, query: str, top_k: int = 5) -> List[str]:
        """Formats retrieved documents into clean context blocks for LLM evaluators."""
        docs = self.retrieve(query, top_k=top_k)
        contexts = []
        for d in docs:
            title = d.metadata.get("title", "Document")
            citation = d.metadata.get("citation", "")
            cite_str = f"\nCitation: {citation}" if citation else ""
            contexts.append(f"[{title}]\n{d.page_content}{cite_str}")
        return contexts
