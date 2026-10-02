"""Master Ingestion Pipeline for UPSC Knowledge Base.
Ingests real raw sources into Chroma vector store with AWS Bedrock Titan Embeddings:
1. Complete Constitution of India (all 465 articles from authentic legal data in data/raw/)
2. Official Legislative Central Acts, Supreme Court judgments, and Reference PDFs (from data/documents/)
"""
import logging
import os
from langchain_core.documents import Document

from src.config import settings
from src.kb.vector_store import get_chroma_vector_store
from src.kb.corpus_loader import (
    load_all_corpus_documents,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("master_ingestion")


def run_master_ingestion(persist_dir: str = None, batch_size: int = 50, reset: bool = True) -> list[Document]:
    """Master ingestion function populating the Chroma vector store in batches."""
    logger.info("=" * 60)
    logger.info("Starting Master UPSC Knowledge Base Ingestion Pipeline (Zero Hardcoding)")
    target_persist = persist_dir or os.path.join(settings.kb_storage_dir, "chroma_db")
    logger.info(f"Target Persist Directory: {target_persist}")
    logger.info(f"Embedding Model: {settings.bedrock_embedding_model_id}")
    logger.info(f"AWS Region: {settings.aws_region}")
    logger.info(f"Reset Existing Collection: {reset}")
    logger.info("=" * 60)

    all_docs = load_all_corpus_documents()
    if not all_docs:
        logger.warning("No documents found to ingest!")
        return []

    vector_store = get_chroma_vector_store(persist_dir)

    if reset:
        logger.info("Resetting old Chroma collection to ensure a clean state with new chunk IDs...")
        try:
            vector_store.delete_collection()
            vector_store = get_chroma_vector_store(persist_dir)
            logger.info("Successfully reset Chroma collection.")
        except Exception as e:
            logger.warning(f"Note during collection reset: {e}")

    logger.info(f"Indexing {len(all_docs)} raw documents into Chroma in batches of {batch_size}...")
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Progress: Ingested {min(i + batch_size, len(all_docs))} / {len(all_docs)} documents...")

    logger.info("=" * 60)
    logger.info("Master Ingestion Completed Successfully!")
    logger.info(f"Total Authentic Knowledge Base Documents Indexed in Chroma: {len(all_docs)}")
    logger.info("=" * 60)
    return all_docs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Master UPSC Knowledge Base Ingestion")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset the existing Chroma collection")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding calls")
    args = parser.parse_args()

    run_master_ingestion(reset=not args.no_reset, batch_size=args.batch_size)
