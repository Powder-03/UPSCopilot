"""Document chunking helper producing standard LangChain Document objects."""
from typing import Dict, Any, Optional
from langchain_core.documents import Document


def create_document(
    page_content: str,
    title: str,
    doc_id: str,
    doc_type: str,
    citation: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Document:
    """Helper to create a standard LangChain Document with normalized metadata."""
    meta = {
        "id": doc_id,
        "title": title,
        "doc_type": doc_type,
        "citation": citation or "",
    }
    if extra_metadata:
        meta.update(extra_metadata)

    return Document(page_content=page_content.strip(), metadata=meta)
