"""Knowledge Base Ground Truth Prefetcher: Fetches authoritative statutory & case law blocks in parallel.

Propagates LangSmith parent trace context into ThreadPoolExecutor workers so
KB_Prefetch_Single traces nest under the parent KB_Batch_Prefetch trace.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.kb.retriever import SelfQueryRetriever
from src.models.parsing import ParsedQuestion
from src.utils.tracing import traceable

logger = logging.getLogger(__name__)


def _get_trace_headers() -> dict[str, str]:
    """Capture current LangSmith RunTree headers for propagation to child threads."""
    try:
        from langsmith.run_helpers import get_current_run_tree

        current_run = get_current_run_tree()
        if current_run:
            return current_run.to_headers()
    except Exception:
        pass
    return {}


@traceable(name="KB_Prefetch_Single", run_type="retriever")
def _prefetch_single_question(
    retriever: SelfQueryRetriever,
    q: dict[str, Any] | ParsedQuestion,
    top_k: int = 6,
) -> tuple[int, list[str]]:
    """Retrieves ground truth context blocks for a single question."""
    if isinstance(q, ParsedQuestion):
        q_num = q.q_num
        question_text = q.question
        answer_text = q.candidate_answer
    else:
        q_num = int(q.get("q_num", 1))
        question_text = q.get("question", "")
        answer_text = q.get("candidate_answer", "")

    # Fast path: unattempted / blank answers do not require KB context
    if not answer_text or not answer_text.strip():
        logger.info("Q%02d is blank; skipping KB context retrieval.", q_num)
        return q_num, []

    try:
        contexts = retriever.get_retrieval_context(question_text, top_k=top_k)
        logger.info("Prefetched %d KB context blocks for Q%02d.", len(contexts), q_num)
        return q_num, contexts
    except Exception as e:
        logger.warning("KB prefetch failed for Q%02d: %s. Continuing with empty context.", q_num, e)
        return q_num, []


class KnowledgeBasePrefetcher:
    """Pre-retrieves ground-truth context blocks for all 20 questions in parallel."""

    def __init__(
        self,
        retriever: SelfQueryRetriever | None = None,
        top_k: int = 6,
        max_workers: int = 10,
    ):
        self.retriever = retriever or SelfQueryRetriever(top_k=top_k)
        self.top_k = top_k
        self.max_workers = max_workers

    def prefetch_single_question(self, q: dict[str, Any] | ParsedQuestion) -> tuple[int, list[str]]:
        """Retrieves ground truth context blocks for a single question (instance method wrapper)."""
        return _prefetch_single_question(
            retriever=self.retriever,
            q=q,
            top_k=self.top_k,
        )

    @traceable(name="KB_Batch_Prefetch", run_type="retriever")
    def prefetch_all(self, questions: list[dict[str, Any] | ParsedQuestion]) -> dict[int, list[str]]:
        """
        Prefetches KB ground truth blocks for all questions concurrently.
        Returns a mapping of q_num -> list[str] of ground truth blocks.

        Captures the current LangSmith RunTree headers and passes them via
        langsmith_extra to each thread-pool worker so KB_Prefetch_Single traces
        nest under this KB_Batch_Prefetch trace.
        """
        total = len(questions)
        logger.info("Starting parallel KB prefetch for %d questions with %d workers...", total, self.max_workers)
        results: dict[int, list[str]] = {}

        if not questions:
            return results

        # Capture parent trace context for propagation into thread pool workers
        parent_headers = _get_trace_headers()
        ls_extra = {"parent": parent_headers} if parent_headers else None

        with ThreadPoolExecutor(max_workers=min(self.max_workers, total or 1)) as executor:
            future_to_q = {
                executor.submit(
                    _prefetch_single_question,
                    retriever=self.retriever,
                    q=q,
                    top_k=self.top_k,
                    langsmith_extra=ls_extra,
                ): q
                for q in questions
            }
            for future in future_to_q:
                try:
                    q_num, contexts = future.result()
                    results[q_num] = contexts
                except Exception as e:
                    logger.error("Error in parallel prefetch future: %s", e)

        return results
