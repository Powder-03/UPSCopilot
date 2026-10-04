"""AWS Lambda Single-Question Evaluator: Evaluates a single question with pre-fetched KB context.

This handler represents a pure compute micro-worker. It receives a single question payload,
runs Call 1 (CoT Diagnostic) and Call 2 (G-Eval Logprob scoring), applies UPSC marking policy,
and returns the EvaluationResult JSON.
Zero vector DB / Pinecone / PyMuPDF dependencies are loaded.
"""
import logging
from typing import Any

from src.evaluation.engine import UPSCEvaluationEngine
from src.utils.tracing import flush_traces, traceable

logger = logging.getLogger(__name__)

# Reusable engine singleton across warm Lambda invocations
_engine: UPSCEvaluationEngine | None = None


def get_engine() -> UPSCEvaluationEngine:
    global _engine
    if _engine is None:
        # Retriever is passed as None: never initialized because kb_context is pre-provided
        _engine = UPSCEvaluationEngine(retriever=None)
    return _engine


@traceable(name="LambdaSingleQuestionEvaluator.handler", run_type="chain")
def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """
    AWS Lambda entrypoint for evaluating a single question.
    Expected event payload:
    {
        "q_num": int,
        "question": str,
        "candidate_answer": str,
        "max_marks": float (default: 10.0),
        "kb_context": list[str] (prefetched ground truth blocks)
    }
    """
    try:
        q_num = int(event.get("q_num", 1))
        question = event.get("question", f"Question {q_num}")
        candidate_answer = event.get("candidate_answer", "")
        max_marks = float(event.get("max_marks", 10.0))
        kb_context = event.get("kb_context", [])

        logger.info("SingleQuestionEvaluator invoked for Q%02d (%.1f marks)", q_num, max_marks)

        engine = get_engine()
        result = engine.evaluate_answer(
            question=question,
            candidate_answer=candidate_answer,
            max_marks=max_marks,
            kb_context=kb_context,
        )

        return {
            "status": "success",
            "q_num": q_num,
            "max_marks": max_marks,
            "evaluation": result.model_dump(mode="json"),
        }
    finally:
        flush_traces()
