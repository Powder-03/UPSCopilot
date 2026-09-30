"""DeepEval retrieval benchmark over the GS-2 golden dataset.

Scores the hybrid retriever with Contextual Recall / Precision using Moonshot Kimi 2.5
on Bedrock as the judge model.

Usage:
    uv run python scripts/evaluate/retrieval.py [--limit N] [--top-k K] [--fixture PATH]
"""
import argparse
import json
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from src.config import settings
from src.evaluation.model_factory import get_eval_llm
from src.kb.retriever import HybridRetriever
from src.utils.cli import configure_console, setup_logging
from src.utils.json import clean_json_text

DEFAULT_FIXTURE = Path("tests/fixtures/golden_dataset.json")


class BedrockKimiJudge(DeepEvalBaseLLM):
    """Bridges Moonshot Kimi 2.5 on AWS Bedrock to DeepEval."""

    def __init__(self, model_name: str = "moonshotai.kimi-k2.5"):
        self.name = model_name
        self.llm = get_eval_llm()
        super().__init__(model=model_name)

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, schema=None) -> str:
        res = self.llm.invoke(prompt).content
        return clean_json_text(res)

    async def a_generate(self, prompt: str, schema=None) -> str:
        res = await self.llm.ainvoke(prompt)
        return clean_json_text(res.content)

    def get_model_name(self) -> str:
        return self.name


def run_evaluation(limit: int | None = None, top_k: int = 8, fixture: Path = DEFAULT_FIXTURE):
    """Builds test cases from the golden dataset and runs DeepEval's native benchmark."""
    with open(fixture, encoding="utf-8") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    retriever = HybridRetriever(top_k=top_k)
    judge = BedrockKimiJudge()

    test_cases = [
        LLMTestCase(
            input=item["query"],
            actual_output=item["ideal_answer"],
            expected_output=item["ideal_answer"],
            retrieval_context=retriever.get_retrieval_context(item["query"], top_k=top_k),
        )
        for item in data
    ]

    metrics = [
        ContextualRecallMetric(threshold=0.7, model=judge, include_reason=True, async_mode=False),
        ContextualPrecisionMetric(threshold=0.7, model=judge, include_reason=True, async_mode=False),
    ]

    # Authenticate with Confident AI if API key is configured
    if settings.confident_api_key:
        try:
            from deepeval.confident.api import set_confident_api_key
            set_confident_api_key(settings.confident_api_key)
        except Exception as e:
            print(f"Note on Confident AI login: {e}")

    hyperparameters = {
        "model": settings.bedrock_eval_model_id,
        "embeddings": settings.bedrock_embedding_model_id,
        "retriever": "Hybrid BM25 + Chroma RRF + FlashRank Cross-Encoder",
        "reranker": "ms-marco-TinyBERT-L-2-v2",
        "top_k": top_k,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "dataset": "UPSC GS-2 Golden Benchmark (15 queries)",
    }

    # Use max_concurrent=4 for optimal Bedrock throughput without rate-limiting
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters=hyperparameters,
        identifier="UPSC-GS2-Retrieval-Benchmark",
        async_config=AsyncConfig(run_async=True, max_concurrent=4),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepEval retrieval benchmark.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N golden queries")
    parser.add_argument("--top-k", type=int, default=8, help="Chunks retrieved per query")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="Golden dataset JSON path")
    args = parser.parse_args()

    configure_console()
    setup_logging()

    if not args.fixture.exists():
        raise SystemExit(f"Error: golden dataset not found: {args.fixture}")

    run_evaluation(limit=args.limit, top_k=args.top_k, fixture=args.fixture)


if __name__ == "__main__":
    main()
