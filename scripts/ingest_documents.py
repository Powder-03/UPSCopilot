"""Document and PDF ingestion pipeline for standard textbooks, NCERT, and reports.
Extracts text using PyMuPDF (fitz) or text loaders, chunks with overlap, and indexes into Chroma.
"""
import os
import glob
import logging
from typing import List, Optional
import pymupdf as fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.kb.vector_store import get_chroma_vector_store

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = os.path.join(settings.data_dir, "documents")


def load_pdf_file(pdf_path: str) -> List[Document]:
    """Extracts pages from a PDF file using PyMuPDF into Document objects."""
    docs: List[Document] = []
    file_name = os.path.basename(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "file_path": pdf_path,
                        "page": page_num + 1,
                        "total_pages": len(doc),
                        "doc_type": "reference_document",
                    },
                )
            )
        logger.info(f"Loaded {len(docs)} pages from {file_name}")
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")

    return docs


def load_text_file(text_path: str) -> List[Document]:
    """Reads .txt or .md files into Document objects."""
    file_name = os.path.basename(text_path)
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return [
                Document(
                    page_content=content,
                    metadata={
                        "source": file_name,
                        "file_path": text_path,
                        "doc_type": "reference_document",
                    },
                )
            ]
    except Exception as e:
        logger.error(f"Error reading text file {text_path}: {e}")
    return []


def load_all_external_documents(docs_dir: Optional[str] = None) -> List[Document]:
    """Loads all PDF, TXT, and MD files from the documents directory and splits into chunks."""
    target_dir = docs_dir or DOCUMENTS_DIR
    os.makedirs(target_dir, exist_ok=True)

    raw_docs: List[Document] = []

    # 1. Load PDFs
    for pdf_path in glob.glob(os.path.join(target_dir, "**", "*.pdf"), recursive=True):
        raw_docs.extend(load_pdf_file(pdf_path))

    # 2. Load TXT & MD
    for ext in ["*.txt", "*.md"]:
        for text_path in glob.glob(os.path.join(target_dir, "**", ext), recursive=True):
            raw_docs.extend(load_text_file(text_path))

    if not raw_docs:
        logger.info(f"No external PDF or text files found in {target_dir}")
        return []

    # Split into chunks suitable for Bedrock Titan embeddings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunked_docs = splitter.split_documents(raw_docs)
    logger.info(f"Split {len(raw_docs)} document pages into {len(chunked_docs)} chunks.")
    return chunked_docs


def ingest_documents(docs_dir: Optional[str] = None, persist_dir: Optional[str] = None, batch_size: int = 50) -> List[Document]:
    """Ingests raw external documents into Chroma vector store."""
    docs = load_all_external_documents(docs_dir)
    if not docs:
        return []

    vector_store = get_chroma_vector_store(persist_dir)
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Ingested document chunks {i + 1} to {min(i + batch_size, len(docs))} of {len(docs)}")

    logger.info(f"Successfully ingested all {len(docs)} document chunks into Chroma.")
    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_documents()
