"""Ingestion pipeline for the complete Constitution of India (all 465 articles).
Loads authentic raw constitutional text and indexes into Chroma vector store.
"""
import os
import json
import logging
from typing import List
from langchain_core.documents import Document

from src.config import settings
from src.kb.vector_store import get_chroma_vector_store

logger = logging.getLogger(__name__)

RAW_CONSTITUTION_PATH = os.path.join(settings.data_dir, "raw", "constitution_raw.json")


def load_constitution_documents() -> List[Document]:
    """Parses all 465 constitutional articles from raw JSON into LangChain Documents."""
    if not os.path.exists(RAW_CONSTITUTION_PATH):
        raise FileNotFoundError(f"Raw constitution file not found at: {RAW_CONSTITUTION_PATH}")

    with open(RAW_CONSTITUTION_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    docs: List[Document] = []
    for item in articles:
        art_num = str(item.get("article", "")).strip()
        title = item.get("title", "").strip()
        desc = item.get("description", "").strip()

        if not desc:
            continue

        if art_num == "0" or title.lower() == "preamble":
            doc = Document(
                page_content=f"Preamble to the Constitution of India\n\n{desc}",
                metadata={
                    "id": "art_preamble",
                    "article": "Preamble",
                    "title": "Preamble to the Constitution of India",
                    "doc_type": "constitution",
                    "citation": "Preamble, Constitution of India",
                    "source": "Constitution of India",
                    "gs_paper": "GS2",
                },
            )
        else:
            full_title = f"Article {art_num} - {title}" if not title.lower().startswith("article") else title
            doc = Document(
                page_content=f"{full_title}\n\n{desc}",
                metadata={
                    "id": f"art_{art_num.lower()}",
                    "article": art_num,
                    "title": full_title,
                    "doc_type": "constitution",
                    "citation": f"Article {art_num}, Constitution of India",
                    "source": "Constitution of India",
                    "gs_paper": "GS2",
                },
            )

        docs.append(doc)

    logger.info(f"Loaded {len(docs)} constitutional articles from raw data.")
    return docs


def ingest_constitution(persist_dir: str = None, batch_size: int = 50) -> List[Document]:
    """Ingests full Constitution documents into Chroma vector store in batches."""
    docs = load_constitution_documents()
    vector_store = get_chroma_vector_store(persist_dir)

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Ingested constitutional articles {i + 1} to {min(i + batch_size, len(docs))} of {len(docs)}")

    logger.info(f"Successfully ingested all {len(docs)} constitutional articles into Chroma.")
    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_constitution()
