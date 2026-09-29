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
