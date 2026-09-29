"""Knowledge Base schemas for statutory provisions, case laws, and commission reports."""
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class KnowledgeItem(BaseModel):
    id: str = Field(..., description="Unique identifier (e.g. art_163, case_sr_bommai_1994)")
    category: Literal["constitution", "case_law", "commission", "scheme", "data_point"]
    title: str = Field(..., description="Full canonical title of the article/case/report")
    keywords: List[str] = Field(default_factory=list, description="Search keywords, aliases, and citation variants")
    content: str = Field(..., description="Full statutory text, judgment ratio, or recommendation summary")
    high_yield_takeaway: str = Field(..., description="Key UPSC Mains exam angle and application")
    citation_ref: Optional[str] = Field(default=None, description="Exact legal citation e.g. (1994) 3 SCC 1")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_searchable_text(self) -> str:
        """Formats item into high-density searchable text for indexing."""
        kw_str = " ".join(self.keywords)
        return f"{self.title}\nKeywords: {kw_str}\n{self.content}\nExam Application: {self.high_yield_takeaway}"


class RetrievalResult(BaseModel):
    item: KnowledgeItem
    score: float
    rank: int
    matched_via: Literal["bm25", "dense", "hybrid_rrf"]
    matched_keywords: List[str] = Field(default_factory=list)


class RetrievalEvaluationItem(BaseModel):
    query_id: str
    question_text: str
    expected_ids: List[str]  # Must-retrieve knowledge item IDs
    primary_id: str          # Must be ranked at rank #1 or #2 for high MRR
    gs_paper: str = "GS2"
