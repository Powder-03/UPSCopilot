"""Pydantic schemas for calibrated UPSC Mains Evaluation Engine."""
from typing import Literal

from pydantic import BaseModel, Field

from src.models.enums import (
    CitationStatus,
    DemandStatus,
    DirectiveType,
    PillarType,
    PresentationArchetype,
    UPSCPerformanceBand,
)


class MicroDemandItem(BaseModel):
    """Specific micro-demand or sub-question extracted from the main prompt."""
    demand: str
    status: DemandStatus
    marks_allocated: float
    marks_obtained: float
    comment: str


class CitationItem(BaseModel):
    """Specific constitutional article, case law, act section, or committee reference."""
    name: str
    status: CitationStatus
    source: Literal["knowledge_base", "open_world"] = "knowledge_base"
    notes: str = ""


class CitationAudit(BaseModel):
    """Audit of legal and factual grounding comparing candidate answer to KB and open-world."""
    mandatory_kb_anchors: list[CitationItem] = Field(default_factory=list)
    open_world_credits: list[CitationItem] = Field(default_factory=list)
    hallucinated_citations: list[CitationItem] = Field(default_factory=list)
    summary: str = ""


class PresentationEvaluation(BaseModel):
    """Evaluation of structural ergonomics, readability, and visual aids."""
    detected_archetype: PresentationArchetype
    visual_density_score: float = Field(..., ge=0.0, le=10.0)
    diagrams_and_tables_found: list[str] = Field(default_factory=list)
    presentation_bonus: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Bounded bonus; Pillar 2 already scores structure, so this is a capped tie-breaker.",
    )
    examiner_critique: str
    topper_reformatting_tip: str


class PillarGEvalScore(BaseModel):
    """Continuous G-Eval probability-weighted score for a single evaluation pillar."""
    pillar: PillarType
    pillar_name: str
    weight_pct: float
    max_marks: float
    discrete_probabilities: dict[int, float] = Field(default_factory=dict)
    raw_expected_rating: float = Field(..., description="Continuous expected rating on 1-5 scale: sum(s * P(s))")
    calibrated_score: float = Field(..., description="Continuous scaled marks awarded for this pillar")
    feedback: str
    scoring_method: Literal["logprob", "text_fallback", "heuristic_fallback"] = Field(
        default="logprob",
        description="How the rating was obtained: native token logprobs, regex text fallback, or offline heuristic fallback.",
    )


class EvaluationResult(BaseModel):
    """Complete calibrated UPSC Mains Answer Evaluation Scorecard."""
    question: str
    candidate_answer: str
    max_marks: float = 10.0
    total_score: float
    percentage: float
    performance_band: UPSCPerformanceBand
    is_off_topic: bool = False
    demand_relevance_gate: float = 1.0
    directive_detected: DirectiveType | None = None
    cot_reasoning_trail: str
    micro_demands: list[MicroDemandItem] = Field(default_factory=list)
    pillars: dict[str, PillarGEvalScore] = Field(default_factory=dict)
    presentation: PresentationEvaluation
    citation_audit: CitationAudit
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    topper_action_plan: list[str] = Field(default_factory=list)
