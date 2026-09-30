"""Knowledge Base schemas for retrieval evaluation benchmarks."""

from pydantic import BaseModel


class RetrievalEvaluationItem(BaseModel):
    """One golden retrieval query: the corpus IDs a question must surface."""

    query_id: str
    question_text: str
    expected_ids: list[str]  # Must-retrieve knowledge item IDs
    primary_id: str          # Must be ranked at rank #1 or #2 for high MRR
    gs_paper: str = "GS2"
