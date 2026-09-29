"""Master Ingestion Pipeline for UPSC Knowledge Base.
Ingests real raw sources into Chroma vector store with AWS Bedrock Titan Embeddings:
1. Complete Constitution of India (all 465 articles from authentic legal data in src/kb/data/raw/)
2. Official Legislative Central Acts, Supreme Court judgments, and Reference PDFs (from src/kb/data/documents/)
"""
import logging
from typing import List
from langchain_core.documents import Document

from src.config import settings
from src.kb.vector_store import get_chroma_vector_store
from src.kb.corpus_loader import (
    load_constitution_documents,
    load_all_external_documents,
    load_all_corpus_documents,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("master_ingestion")


def run_master_ingestion(persist_dir: str = None, batch_size: int = 50) -> List[Document]:
    """Master ingestion function populating the Chroma vector store in batches."""
    logger.info("=" * 60)
    logger.info("Starting Master UPSC Knowledge Base Ingestion Pipeline (Zero Hardcoding)")
    logger.info(f"Target Persist Directory: {persist_dir or settings.kb_storage_dir}")
    logger.info(f"Embedding Model: {settings.bedrock_embedding_model_id}")
    logger.info(f"AWS Region: {settings.aws_region}")
    logger.info("=" * 60)

    all_docs = load_all_corpus_documents()
    if not all_docs:
        logger.warning("No documents found to ingest!")
        return []

    vector_store = get_chroma_vector_store(persist_dir)

    logger.info(f"Indexing {len(all_docs)} raw documents into Chroma in batches of {batch_size}...")
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Progress: Ingested {min(i + batch_size, len(all_docs))} / {len(all_docs)} documents...")

    logger.info("=" * 60)
    logger.info(f"Master Ingestion Completed Successfully!")
    logger.info(f"Total Authentic Knowledge Base Documents Indexed in Chroma: {len(all_docs)}")
    logger.info("=" * 60)
    return all_docs


if __name__ == "__main__":
    run_master_ingestion()
