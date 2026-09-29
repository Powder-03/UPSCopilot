"""Hybrid Ensemble Retriever combining LangChain BM25Retriever and Chroma Vector Store."""
from typing import List, Optional
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from src.kb.vector_store import get_chroma_vector_store


class HybridRetriever:
    """
    Standard LangChain Hybrid Retriever combining:
    1. Sparse Lexical Search (BM25Retriever for exact section numbers, case names, and statute titles)
    2. Dense Semantic Search (Chroma vector store with Amazon Titan Bedrock embeddings)
    3. Native Reciprocal Rank Fusion (EnsembleRetriever)
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        persist_dir: Optional[str] = None,
        force_mock: bool = False,
        top_k: int = 5,
    ):
        self.persist_dir = persist_dir
        self.vector_store = get_chroma_vector_store(persist_dir, force_mock=force_mock)
        self.documents = documents or []
        self.top_k = top_k
        self._ensemble: Optional[EnsembleRetriever] = None

        if self.documents:
            self.rebuild_index(self.documents, top_k=self.top_k)

    def rebuild_index(self, documents: List[Document], top_k: int = 5):
        """Constructs the BM25 and Chroma EnsembleRetriever over the given documents."""
        self.documents = documents
        self.top_k = top_k

        # 1. Sparse BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = top_k

        # 2. Dense Vector Retriever
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

        # 3. Native LangChain Ensemble (RRF)
        self._ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6],
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Retrieves top matching documents via hybrid RRF search."""
        k = top_k or self.top_k
        if self._ensemble is None:
            return self.vector_store.as_retriever(search_kwargs={"k": k}).invoke(query)
        return self._ensemble.invoke(query)[:k]

    def get_retrieval_context(self, query: str, top_k: int = 5) -> List[str]:
        """Formats retrieved documents into clean context blocks for DeepEval evaluators."""
        docs = self.retrieve(query, top_k=top_k)
        contexts = []
        for d in docs:
            title = d.metadata.get("title", "Document")
            citation = d.metadata.get("citation", "")
            cite_str = f"\nCitation: {citation}" if citation else ""
            contexts.append(f"[{title}]\n{d.page_content}{cite_str}")
        return contexts
