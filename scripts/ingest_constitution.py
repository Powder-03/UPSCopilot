"""Automated Ingestion Script for the Complete Constitution of India using LangChain.
Fetches all 465 articles from authentic open legal data, creates LangChain Document objects,
and stores them into the Chroma vector store.
"""
import os
import json
import logging
import urllib.request
from typing import List
from langchain_core.documents import Document

from src.config import settings
from src.kb.vector_store import get_chroma_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_CONSTITUTION_URL = (
    "https://raw.githubusercontent.com/civictech-India/constitution-of-india/main/constitution_of_india.json"
)
RAW_LOCAL_PATH = os.path.join(settings.data_dir, "raw", "constitution_raw.json")


def fetch_raw_constitution() -> List[dict]:
    """Downloads or loads cached raw Constitution JSON (all 465 articles)."""
    os.makedirs(os.path.dirname(RAW_LOCAL_PATH), exist_ok=True)

    if os.path.exists(RAW_LOCAL_PATH):
        logger.info(f"Loading cached raw Constitution from {RAW_LOCAL_PATH}")
        with open(RAW_LOCAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"Fetching full Constitution of India from {RAW_CONSTITUTION_URL}...")
    req = urllib.request.Request(RAW_CONSTITUTION_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        raw_data = json.loads(resp.read().decode("utf-8"))

    with open(RAW_LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Downloaded and cached {len(raw_data)} constitutional articles to {RAW_LOCAL_PATH}")
    return raw_data


def load_constitution_documents() -> List[Document]:
    """Parses raw constitutional articles into standard LangChain Document objects."""
    raw_articles = fetch_raw_constitution()
    docs: List[Document] = []

    for item in raw_articles:
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
                    "gs_paper": "GS2",
                },
            )

        docs.append(doc)

    logger.info(f"Created {len(docs)} LangChain constitutional documents.")
    return docs


def ingest_constitution(persist_dir: str = None) -> List[Document]:
    """Ingests full Constitution documents into Chroma vector store."""
    docs = load_constitution_documents()
    vector_store = get_chroma_vector_store(persist_dir)
    vector_store.add_documents(docs)
    logger.info(f"Successfully added {len(docs)} constitutional articles to Chroma vector store.")
    return docs


if __name__ == "__main__":
    ingest_constitution()
