"""Self-Query Cloud-Native Retriever using Pinecone and Vertex AI Embeddings."""
import logging
import re
from typing import Any

from langchain_core.documents import Document

from src.kb.vector_store import get_vector_store
from src.utils.tracing import traceable

logger = logging.getLogger(__name__)

# --- Retrieval tuning constants ---
CHANNEL_DEPTH = 20            # Candidate pool pulled from Pinecone before diversity filtering
MAX_CHUNKS_PER_SOURCE = 4     # Diversity ceiling: chunks per source document in final context


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer helper."""
    return re.findall(r"\w+", text.lower())


def extract_self_query_filter(query: str) -> dict[str, Any] | None:
    """
    Analyzes UPSC question/query to extract structured Pinecone metadata filters.
    Detects GS paper focus and statutory/case law document types.
    """
    q_lower = query.lower()
    conditions: list[dict[str, Any]] = []

    # 1. GS Paper classification
    if any(w in q_lower for w in [
        "geography", "monsoon", "plate tectonics", "earthquake", "volcano",
        "mineral", "indus valley", "freedom struggle", "gandhian phase", "swadeshi",
        "population", "urbanization", "secularism", "regionalism"
    ]):
        conditions.append({"gs_paper": {"$eq": "gs1"}})
    elif any(w in q_lower for w in [
        "constitution", "article", "amendment", "fundamental rights", "dpsp",
        "governor", "president", "parliament", "tribunal", "rti", "dpdp",
        "lokpal", "cvc", "pmla", "electoral bonds", "quad", "i2u2", "brics",
        "unclos", "wto", "civil services", "governance", "sevottam"
    ]):
        conditions.append({"gs_paper": {"$eq": "gs2"}})
    elif any(w in q_lower for w in [
        "fiscal", "inflation", "gdp", "frbm", "monetary policy", "msp",
        "agriculture", "public distribution", "food security", "climate change",
        "biodiversity", "wildlife", "quantum mission", "space", "isro",
        "uapa", "nia", "cyber security", "afspa", "internal security"
    ]):
        conditions.append({"gs_paper": {"$eq": "gs3"}})
    elif any(w in q_lower for w in [
        "ethics", "morality", "probity in governance", "integrity", "nolan",
        "kant", "utilitarian", "rawls", "emotional intelligence", "case study",
        "moral dilemma", "code of conduct"
    ]):
        conditions.append({"gs_paper": {"$eq": "gs4"}})

    # 2. Document type preference
    if any(w in q_lower for w in ["case", "judgement", "judgment", "verdict", "doctrine", "bench", "vs ", "v. ", "sc held"]):
        conditions.append({"doc_type": {"$in": ["case_law", "constitution"]}})
    elif any(w in q_lower for w in ["act", "statute", "section", "legislation", "ordinance", "bill"]):
        conditions.append({"doc_type": {"$in": ["statute", "governance"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class SelfQueryRetriever:
    """
    Cloud-native Self-Query Retriever powered by Pinecone & Vertex AI text-embedding-004.
    1. Extracts query intent & syllabus metadata filter (GS1-GS4, doc_type).
    2. Performs single-pass semantic vector search directly in Pinecone cloud.
    3. Applies Source Diversity Filter to prevent single-document monopoly.
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
        persist_dir: str | None = None,
        force_mock: bool = False,
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.force_mock = force_mock
        self.vector_store = get_vector_store(persist_dir, force_mock=force_mock)

    @traceable(name="Pinecone_SelfQuery_Retriever", run_type="retriever")
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_reranker: bool = False,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieves grounded knowledge documents for a UPSC answer evaluation query.
        Applies self-query metadata filtering and source diversity preservation.
        """
        k = top_k or self.top_k
        active_filter = filter

        # 1. Query Pinecone with semantic embeddings
        candidates = self.vector_store.similarity_search(
            query=query,
            k=CHANNEL_DEPTH,
            filter=active_filter,
        )

        # 2. Backfill fallback if filtered query returned fewer than k candidates
        if len(candidates) < k and active_filter is not None:
            unfiltered_candidates = self.vector_store.similarity_search(
                query=query,
                k=CHANNEL_DEPTH,
                filter=None,
            )
            seen_texts = {c.page_content[:80] for c in candidates}
            for doc in unfiltered_candidates:
                if doc.page_content[:80] not in seen_texts:
                    candidates.append(doc)
                    seen_texts.add(doc.page_content[:80])
                if len(candidates) >= CHANNEL_DEPTH:
                    break

        if not candidates:
            return []

        # 3. Source Diversity Filtering: allow up to MAX_CHUNKS_PER_SOURCE chunks per source
        # Prevents a single large statute (e.g. RTI Act) from monopolizing evaluation context.
        selected_docs: list[Document] = []
        source_counts: dict[str, int] = {}
        for doc in candidates:
            src = doc.metadata.get("source", doc.metadata.get("title", "unknown"))
            if source_counts.get(src, 0) < MAX_CHUNKS_PER_SOURCE:
                selected_docs.append(doc)
                source_counts[src] = source_counts.get(src, 0) + 1
            if len(selected_docs) >= k:
                break

        # Backfill if k slots not yet filled
        if len(selected_docs) < k:
            for doc in candidates:
                if doc not in selected_docs:
                    selected_docs.append(doc)
                if len(selected_docs) >= k:
                    break

        return selected_docs[:k]

    @traceable(name="Format_Retrieval_Context", run_type="tool")
    def get_retrieval_context(self, query: str, top_k: int = 8, use_reranker: bool = False) -> list[str]:
        """Formats retrieved documents into clean context blocks for LLM evaluators."""
        docs = self.retrieve(query, top_k=top_k, use_reranker=use_reranker)
        contexts: list[str] = []
        for d in docs:
            title = d.metadata.get("title", "Document")
            citation = d.metadata.get("citation", "")
            cite_str = f"\nCitation: {citation}" if citation else ""
            contexts.append(f"[{title}]\n{d.page_content}{cite_str}")
        return contexts


# Backward-compatible alias for existing imports across the project
HybridRetriever = SelfQueryRetriever
