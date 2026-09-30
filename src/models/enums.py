"""Enums for the UPSC Evaluation Engine."""
from enum import Enum


class DirectiveType(str, Enum):
    DISCUSS = "discuss"
    CRITICALLY_ANALYZE = "critically_analyze"
    CRITICALLY_EXAMINE = "critically_examine"
    ELUCIDATE = "elucidate"
    EVALUATE = "evaluate"
    COMMENT = "comment"
    EXPLAIN = "explain"
    TO_WHAT_EXTENT = "to_what_extent"


class UPSCPerformanceBand(str, Enum):
    NEEDS_FOUNDATION = "Needs Foundation (<35%)"
    AVERAGE = "Average (35-45%)"
    GOOD = "Good (46-55%)"
    TOPPER = "Topper Benchmark (56-65%+)"


class PresentationArchetype(str, Enum):
    PARAGRAPH_HEAVY = "paragraph_heavy"
    BULLET_STRUCTURED = "bullet_structured"
    HYBRID_DIAGRAMMATIC = "hybrid_diagrammatic"
    TABULAR = "tabular"


class DemandStatus(str, Enum):
    FULLY_ADDRESSED = "fully_addressed"
    PARTIALLY_ADDRESSED = "partially_addressed"
    OMITTED = "omitted"
    OFF_TOPIC = "off_topic"


class CitationStatus(str, Enum):
    MANDATORY_FOUND = "mandatory_found"
    MANDATORY_MISSING = "mandatory_missing"
    OPEN_WORLD_CREDITED = "open_world_credited"
    HALLUCINATED_OR_WRONG = "hallucinated_or_wrong"


class PillarType(str, Enum):
    DEMAND_FULFILLMENT = "demand_fulfillment"
    STRUCTURE_PRESENTATION = "structure_presentation"
    MULTIDIMENSIONAL_BREADTH = "multidimensional_breadth"
    GROUNDED_CITATIONS = "grounded_citations"
    CONCLUSION_WAY_FORWARD = "conclusion_way_forward"
    INTRODUCTION = "introduction"
