"""Pydantic schemas for the document parsing and vision extraction pipeline."""
from typing import Any

from pydantic import BaseModel, Field


class ParsedQuestion(BaseModel):
    """Represents a single parsed UPSC answer attempt matching the evaluator's expected input."""
    q_num: int = Field(..., description="1-indexed question number (1 to 20)")
    max_marks: float = Field(default=10.0, description="Allocated marks for the question (usually 10.0 or 15.0)")
    question: str = Field(..., description="Printed question prompt extracted from booklet header or master reference")
    candidate_answer: str = Field(default="", description="Transcribed handwritten answer text with structural annotations")
    page_numbers: list[int] = Field(default_factory=list, description="Original 1-indexed PDF page numbers for this answer")
    diagrams: list[str] = Field(default_factory=list, description="Descriptions of any diagrams, flowcharts, or tables detected")
    is_blank: bool = Field(default=False, description="True if the question space was completely blank / unattempted")
    word_count: int = Field(default=0, description="Approximate transcribed word count")

    def to_evaluation_dict(self) -> dict[str, Any]:
        """Converts to the exact dict format expected by the UPSC evaluation engine."""
        return {
            "q_num": self.q_num,
            "max_marks": self.max_marks,
            "question": self.question,
            "candidate_answer": self.candidate_answer,
        }


class ParsedDocument(BaseModel):
    """Represents the complete parsed result for a candidate answer booklet."""
    source_file: str = Field(..., description="Path or identifier of the source PDF")
    total_pages: int = Field(..., description="Total pages in the source PDF")
    questions: list[ParsedQuestion] = Field(default_factory=list, description="List of all extracted questions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Parsing metadata, timestamps, models used")

    def to_evaluation_list(self) -> list[dict[str, Any]]:
        """Exports the entire document as a list of question dicts for the evaluator."""
        return [q.to_evaluation_dict() for q in self.questions]
