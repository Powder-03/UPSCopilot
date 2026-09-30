"""Hybrid Ensemble Retriever combining BM25 and Chroma Vector Store with Bedrock Titan Embeddings."""
import logging
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.kb.corpus_loader import load_all_corpus_documents
from src.kb.vector_store import get_chroma_vector_store

logger = logging.getLogger(__name__)

# --- Retrieval tuning constants ---
RRF_K = 60                    # Standard Reciprocal Rank Fusion smoothing constant
CHANNEL_DEPTH = 30            # Candidates pulled from each channel (BM25 / dense) before fusion
RERANK_POOL_MIN = 40          # Floor for the cross-encoder candidate pool
RERANK_POOL_MULTIPLIER = 4    # Pool scales with requested top_k: max(RERANK_POOL_MIN, k * multiplier)
MAX_CHUNKS_PER_SOURCE = 5     # Diversity ceiling: chunks per source document in the final selection


def tokenize(text: str) -> list[str]:
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
        documents: list[Document] | None = None,
        persist_dir: str | None = None,
        force_mock: bool = False,
        top_k: int = 5,
    ):
        self.persist_dir = persist_dir
        self.force_mock = force_mock
        self.top_k = top_k
        self.documents: list[Document] = documents if documents is not None else load_all_corpus_documents()
        self.id_to_idx: dict[str, int] = {
            doc.metadata["id"]: i for i, doc in enumerate(self.documents) if doc.metadata.get("id")
        }
        self.doc_by_id: dict[str, Document] = {
            doc.metadata.get("id", f"doc_{i}"): doc for i, doc in enumerate(self.documents)
        }

        # Initialize BM25 over the real documents
        self.corpus = [f"{doc.metadata.get('title', '')} {doc.page_content}" for doc in self.documents]
        self.tokenized_corpus = [tokenize(t) for t in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

        # Chroma vector store
        self.vector_store = get_chroma_vector_store(persist_dir, force_mock=force_mock)

        # FlashRank Cross-Encoder Reranker
        try:
            from flashrank import Ranker
            self.ranker = Ranker()
            logger.info("FlashRank cross-encoder reranker initialized successfully.")
        except Exception as e:
            logger.warning(f"FlashRank initialization skipped: {e}")
            self.ranker = None

    def retrieve(self, query: str, top_k: int | None = None, use_reranker: bool = True) -> list[Document]:
        """Hybrid search with Reciprocal Rank Fusion (RRF) and FlashRank cross-encoder reranking."""
        k = top_k or self.top_k
        query_tokens = tokenize(query)
        rrf_scores: dict[int, float] = {}

        # 1. BM25 scoring
        if self.bm25 and query_tokens:
            bm25_scores = self.bm25.get_scores(query_tokens)
            sorted_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            for rank, idx in enumerate(sorted_bm25[:CHANNEL_DEPTH]):
                if bm25_scores[idx] > 0.0:
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_K + rank + 1))

        # 2. Dense vector search
        dense_direct_docs: list[Document] = []
        try:
            dense_docs = self.vector_store.similarity_search(query, k=CHANNEL_DEPTH)
            for rank, doc in enumerate(dense_docs):
                doc_id = doc.metadata.get("id")
                if doc_id and doc_id in self.id_to_idx:
                    idx = self.id_to_idx[doc_id]
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_K + rank + 1))
                else:
                    dense_direct_docs.append(doc)
        except Exception as e:
            logger.warning(f"Error in dense similarity search: {e}")

        # Gather an expanded candidate pool for cross-encoder reranking
        candidate_pool_size = max(RERANK_POOL_MIN, k * RERANK_POOL_MULTIPLIER)
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_docs = [self.documents[idx] for idx, _ in sorted_indices[:candidate_pool_size]]

        # Augment from direct dense results if needed
        seen_texts = {d.page_content[:80] for d in candidate_docs}
        for d in dense_direct_docs:
            if d.page_content[:80] not in seen_texts:
                candidate_docs.append(d)
                seen_texts.add(d.page_content[:80])
            if len(candidate_docs) >= candidate_pool_size:
                break

        if not candidate_docs:
            return []

        # 3. Cross-Encoder Reranking via FlashRank with Diversity Filter
        if use_reranker and self.ranker and len(candidate_docs) > 1:
            try:
                from flashrank import RerankRequest
                passages = [
                    {"id": i, "text": f"{d.metadata.get('title', '')}\n{d.page_content}"}
                    for i, d in enumerate(candidate_docs)
                ]
                rerank_request = RerankRequest(query=query, passages=passages)
                reranked = self.ranker.rerank(rerank_request)

                # Diversity filtering: allow up to MAX_CHUNKS_PER_SOURCE chunks per source document
                # so deep statutory or case dossiers are not prematurely truncated, while still
                # preventing total single-doc monopoly.
                selected_docs: list[Document] = []
                source_counts: dict[str, int] = {}
                for r in reranked:
                    doc = candidate_docs[r["id"]]
                    src = doc.metadata.get("source", doc.metadata.get("title", "unknown"))
                    if source_counts.get(src, 0) < MAX_CHUNKS_PER_SOURCE:
                        selected_docs.append(doc)
                        source_counts[src] = source_counts.get(src, 0) + 1
                    if len(selected_docs) >= k:
                        break

                # If k slots not filled, backfill from remaining
                if len(selected_docs) < k:
                    for r in reranked:
                        doc = candidate_docs[r["id"]]
                        if doc not in selected_docs:
                            selected_docs.append(doc)
                        if len(selected_docs) >= k:
                            break

                return selected_docs
            except Exception as e:
                logger.warning(f"FlashRank reranking error, falling back to RRF: {e}")

        return candidate_docs[:k]

    def get_retrieval_context(self, query: str, top_k: int = 8, use_reranker: bool = True) -> list[str]:
        """Formats retrieved documents into clean context blocks for LLM evaluators."""
        docs = self.retrieve(query, top_k=top_k, use_reranker=use_reranker)
        contexts = []
        for d in docs:
            title = d.metadata.get("title", "Document")
            citation = d.metadata.get("citation", "")
            cite_str = f"\nCitation: {citation}" if citation else ""
            contexts.append(f"[{title}]\n{d.page_content}{cite_str}")
        return contexts
