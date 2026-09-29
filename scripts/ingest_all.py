"""Master Ingestion Pipeline for UPSC Knowledge Base using LangChain.
Fetches, parses, and populates Chroma vector store across Constitution, Statutes, and SC Cases.
"""
import logging
from typing import List
from langchain_core.documents import Document

from src.config import settings
from src.kb.vector_store import get_chroma_vector_store
from scripts.ingest_constitution import ingest_constitution, load_constitution_documents
from scripts.ingest_central_acts import ingest_central_acts, load_central_act_documents
from scripts.ingest_sc_cases import ingest_sc_cases, load_sc_case_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("master_ingestion")


def load_all_corpus_documents() -> List[Document]:
    """Loads all unified corpus documents across all domains into a single list."""
    const_docs = load_constitution_documents()
    act_docs = load_central_act_documents()
    case_docs = load_sc_case_documents()
    return const_docs + act_docs + case_docs


def run_master_ingestion(persist_dir: str = None) -> List[Document]:
    """Master ingestion function populating the Chroma vector store."""
    logger.info("=" * 60)
    logger.info("Starting Master UPSC Knowledge Base Ingestion Pipeline (LangChain)")
    logger.info(f"Target Persist Directory: {persist_dir or settings.kb_storage_dir}")
    logger.info(f"Embedding Model: {settings.bedrock_embedding_model_id}")
    logger.info(f"AWS Credentials Configured: {settings.has_aws_credentials}")
    logger.info("=" * 60)

    all_docs = load_all_corpus_documents()
    vector_store = get_chroma_vector_store(persist_dir)
    vector_store.add_documents(all_docs)

    logger.info("=" * 60)
    logger.info(f"Master Ingestion Completed Successfully!")
    logger.info(f"Total Unified Knowledge Documents in Chroma: {len(all_docs)}")
    logger.info("=" * 60)
    return all_docs


if __name__ == "__main__":
    run_master_ingestion()
