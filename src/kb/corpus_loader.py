"""Corpus document loader for authentic UPSC Knowledge Base sources.
Loads:
1. Complete Constitution (465 articles from raw JSON)
2. External documents (PDFs, Markdown case dossiers, Central Act dossiers from src/kb/data/documents/)
"""
import os
import glob
import json
import logging
from typing import List, Optional
import pymupdf as fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)

RAW_CONSTITUTION_PATH = os.path.join(settings.data_dir, "raw", "constitution_raw.json")
DOCUMENTS_DIR = os.path.join(settings.data_dir, "documents")


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
            # Determine document type based on subfolder
            doc_type = "reference_document"
            if "sc_cases" in text_path:
                doc_type = "landmark_case"
            elif "central_acts" in text_path:
                doc_type = "central_act"

            return [
                Document(
                    page_content=content,
                    metadata={
                        "source": file_name,
                        "file_path": text_path,
                        "doc_type": doc_type,
                        "title": file_name.replace(".md", "").replace("case_", "").replace("act_", "").replace("_", " ").title(),
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

    # 2. Load TXT & MD (SC Cases and Central Acts)
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

    # Assign deterministic, unique IDs to all external chunks
    for i, doc in enumerate(chunked_docs):
        src = doc.metadata.get("source", "doc").replace(".pdf", "").replace(".md", "").replace(".txt", "")
        page = doc.metadata.get("page")
        page_suffix = f"_p{page}" if page else ""
        doc.metadata["id"] = f"{src}{page_suffix}_chunk_{i}"

    logger.info(f"Split {len(raw_docs)} document pages/files into {len(chunked_docs)} chunks.")
    return chunked_docs


def load_all_corpus_documents() -> List[Document]:
    """Loads all authentic raw corpus documents across all domains into a single list."""
    const_docs = load_constitution_documents()
    ext_docs = load_all_external_documents()

    total = const_docs + ext_docs

    # Guarantee every document has a non-empty unique ID
    for idx, doc in enumerate(total):
        if not doc.metadata.get("id"):
            doc.metadata["id"] = f"corpus_doc_{idx}"

    logger.info(
        f"Corpus Summary: {len(const_docs)} Authentic Constitutional articles, "
        f"{len(ext_docs)} Official legislative/case/document chunks. "
        f"Total: {len(total)} documents."
    )
    return total
