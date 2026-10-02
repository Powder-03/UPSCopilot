"""Domain-specific exceptions for the UPSC Answer Evaluation Engine."""


class EvaluationError(Exception):
    """Base exception for all UPSC evaluation errors."""

    pass


class ModelInvocationError(EvaluationError):
    """Raised when an evaluation or vision model API fails."""

    pass


class RatingExtractionError(EvaluationError):
    """Raised when the model output cannot be parsed into valid UPSC pillar ratings."""

    pass


class DocumentParsingError(EvaluationError):
    """Raised when PDF booklet OCR or segmentation encounters an unrecoverable error."""

    pass
