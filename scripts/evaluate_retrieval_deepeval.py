import sys
import os
import re
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"
from typing import Optional
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate.configs import AsyncConfig

import logging
logging.getLogger("langchain_aws").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)

from src.config import settings
from src.evaluation.model_factory import get_eval_llm
from src.kb.retriever import HybridRetriever


def _clean_json_output(text: str) -> str:
    """Extracts and validates a clean JSON object, discarding trailing commentary."""
    if not text:
        return "{}"

    # 1. Extract from markdown code block if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    else:
        start = text.find("{")
        if start == -1:
            return text
        candidate = text[start:]

    # 2. Use raw_decode to parse the first JSON object and strip trailing text
    try:
        obj, _ = json.JSONDecoder().raw_decode(candidate)
        return json.dumps(obj)
    except Exception:
        return candidate


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
        return _clean_json_output(res)

    async def a_generate(self, prompt: str, schema=None) -> str:
        res = await self.llm.ainvoke(prompt)
        return _clean_json_output(res.content)

    def get_model_name(self) -> str:
        return self.name


def run_evaluation(limit: Optional[int] = None, top_k: int = 8):
    """Builds test cases from the golden dataset and runs DeepEval's native benchmark."""
    with open("tests/data/golden_dataset.json", "r", encoding="utf-8") as f:
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


if __name__ == "__main__":
    run_evaluation()
