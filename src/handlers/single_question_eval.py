"""AWS Lambda Single-Question Evaluator: Evaluates a single question with pre-fetched KB context.

This handler represents a pure compute micro-worker. It receives a single question payload,
runs Call 1 (CoT Diagnostic) and Call 2 (G-Eval Logprob scoring), applies UPSC marking policy,
and returns the EvaluationResult JSON.
Zero vector DB / Pinecone / PyMuPDF dependencies are loaded.

Tracing: Receives `trace_headers` from the EC2 coordinator's parent trace, so this Lambda's
trace appears as a child of the parent `EC2_FanOut_20_Lambda_Evaluation` trace in LangSmith.
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


@traceable(name="LambdaSingleQuestionEval", run_type="chain")
def _evaluate_question(
    engine: UPSCEvaluationEngine,
    q_num: int,
    question: str,
    candidate_answer: str,
    max_marks: float,
    kb_context: list[str],
) -> dict[str, Any]:
    """Evaluates a single question under the parent trace context.

    This @traceable function accepts langsmith_extra={"parent": trace_headers}
    so the trace nests under the EC2 coordinator's fan-out trace.
    """
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


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """AWS Lambda entrypoint for evaluating a single question.

    Expected event payload:
    {
        "q_num": int,
        "question": str,
        "candidate_answer": str,
        "max_marks": float (default: 10.0),
        "kb_context": list[str] (prefetched ground truth blocks),
        "trace_headers": dict[str, str] (optional LangSmith parent trace headers)
    }
    """
    q_num = int(event.get("q_num", 1))
    question = event.get("question", f"Question {q_num}")
    candidate_answer = event.get("candidate_answer", "")
    max_marks = float(event.get("max_marks", 10.0))
    kb_context = event.get("kb_context", [])
    trace_headers = event.get("trace_headers")

    logger.info("SingleQuestionEvaluator invoked for Q%02d (%.1f marks)", q_num, max_marks)

    engine = get_engine()

    try:
        # Build langsmith_extra to parent this trace under the coordinator's fan-out trace
        ls_extra: dict[str, Any] | None = None
        if trace_headers:
            ls_extra = {"parent": trace_headers}

        return _evaluate_question(
            engine=engine,
            q_num=q_num,
            question=question,
            candidate_answer=candidate_answer,
            max_marks=max_marks,
            kb_context=kb_context,
            langsmith_extra=ls_extra,
        )
    finally:
        flush_traces()
