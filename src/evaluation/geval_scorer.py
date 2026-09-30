"""G-Eval probability-weighted continuous scoring engine using Bedrock Kimi 2.5 logprobs."""
import json
import logging
import math
import re
from typing import Any

import boto3

from src.config import settings
from src.evaluation.prompt_templates import build_geval_scoring_prompt
from src.models.enums import PillarType
from src.models.evaluation import PillarGEvalScore

logger = logging.getLogger(__name__)

# Canonical UPSC pillar weights and descriptions (must sum to 1.0)
PILLAR_CONFIGS = {
    PillarType.DEMAND_FULFILLMENT: {
        "name": "Directive & Demand Fulfillment",
        "weight": 0.25,
        "key": "P1",
    },
    PillarType.STRUCTURE_PRESENTATION: {
        "name": "Structural Architecture & Presentation",
        "weight": 0.15,
        "key": "P2",
    },
    PillarType.MULTIDIMENSIONAL_BREADTH: {
        "name": "Multi-Dimensional Breadth (GS-2 Governance)",
        "weight": 0.20,
        "key": "P3",
    },
    PillarType.GROUNDED_CITATIONS: {
        "name": "Grounded Legal Citations & Accuracy",
        "weight": 0.20,
        "key": "P4",
    },
    PillarType.CONCLUSION_WAY_FORWARD: {
        "name": "Conclusion & Constructive Way Forward",
        "weight": 0.15,
        "key": "P5",
    },
    PillarType.INTRODUCTION: {
        "name": "Introduction & Context Setting",
        "weight": 0.05,
        "key": "P6",
    },
}


def calibrate_rating_to_upsc_marks(expected_rating: float, max_marks: float) -> float:
    """Calibrates 1.0 - 5.0 G-Eval rating to official UPSC Mains percentage band:
    - Rating 1.0 -> 25% (Poor)
    - Rating 2.0 -> 35% (Below Average)
    - Rating 3.0 -> 45% (Average)
    - Rating 4.0 -> 55% (Good)
    - Rating 5.0 -> 65% (Topper Benchmark)
    Linear calibration formula: pct = 0.15 + (R * 0.10)
    """
    bounded_r = max(1.0, min(5.0, expected_rating))
    upsc_pct = 0.15 + (bounded_r * 0.10)
    return round(upsc_pct * max_marks, 3)


class BedrockGEvalScorer:
    """Extracts authentic token logprobs from Moonshot Kimi 2.5 and computes G-Eval scores."""

    def __init__(self, model_id: str | None = None, region: str | None = None):
        self.model_id = model_id or settings.bedrock_eval_model_id
        self.region = region or settings.aws_region
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def score_pillars(
        self,
        question: str,
        candidate_answer: str,
        cot_reasoning_trail: str,
        pillar_summary: dict[str, str],
        max_marks: float = 10.0,
    ) -> dict[str, PillarGEvalScore]:
        """Runs Call 2: Single-pass G-Eval multi-pillar scoring with authentic token logprobs."""
        prompt = build_geval_scoring_prompt(
            question=question,
            candidate_answer=candidate_answer,
            cot_reasoning_trail=cot_reasoning_trail,
            pillar_summary=pillar_summary,
        )

        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5,
        }

        try:
            res = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            res_body = json.loads(res["body"].read().decode("utf-8"))
            first_choice = res_body["choices"][0]
            content_tokens = first_choice.get("logprobs", {}).get("content", [])
            output_text = first_choice.get("message", {}).get("content", "")
            logger.info(f"G-Eval raw score tokens output: {output_text.strip()}")
            return self._parse_logprob_pillars(content_tokens, output_text, pillar_summary, max_marks)
        except Exception as e:
            logger.warning(f"Error in Bedrock G-Eval logprob call, falling back to heuristic scoring: {e}")
            return self._fallback_pillar_scoring(pillar_summary, max_marks)

    def _parse_logprob_pillars(
        self,
        content_tokens: list,
        output_text: str,
        pillar_summary: dict[str, str],
        total_max_marks: float,
    ) -> dict[str, PillarGEvalScore]:
        """Parses output tokens to identify P1..P6 score positions and computes E[S]."""
        results: dict[str, PillarGEvalScore] = {}

        # Map each pillar key (P1..P6) to its extracted discrete rating probabilities
        key_to_probs = self._map_keys_to_token_logprobs(content_tokens)

        # Build PillarGEvalScore objects
        for pillar_type, cfg in PILLAR_CONFIGS.items():
            key = cfg["key"]
            weight = cfg["weight"]
            max_pillar_marks = total_max_marks * weight
            feedback = pillar_summary.get(pillar_type.value, "Evaluated according to UPSC rubric.")

            probs = key_to_probs.get(key)
            if probs:
                expected_r = sum(score * p for score, p in probs.items())
                scoring_method = "logprob"
            else:
                # Text parsing fallback if token logprobs were unavailable at the score position
                expected_r = self._extract_digit_fallback(output_text, key)
                probs = {int(round(expected_r)): 1.0}
                scoring_method = "text_fallback"
                logger.warning(
                    f"No usable token logprobs for {key}; using text-parsed rating {expected_r}."
                )

            calibrated_marks = calibrate_rating_to_upsc_marks(expected_r, max_pillar_marks)
            results[pillar_type.value] = PillarGEvalScore(
                pillar=pillar_type,
                pillar_name=cfg["name"],
                weight_pct=weight,
                max_marks=round(max_pillar_marks, 2),
                discrete_probabilities=probs,
                raw_expected_rating=round(expected_r, 3),
                calibrated_score=calibrated_marks,
                feedback=feedback,
                scoring_method=scoring_method,
            )

        return results

    def _map_keys_to_token_logprobs(self, content_tokens: list) -> dict[str, dict[int, float]]:
        """Maps each pillar key (P1..P6) to the discrete rating probabilities at its score-token position.

        Reconstructs the decoded output with per-token character offsets so the score token is
        located correctly regardless of how the model tokenizes "P1: 4" (e.g. "P1:" + " 4",
        or "P" + "1" + ":" + " 4"). Each key's scan is bounded by the next key marker so a
        missing distribution can never borrow the following pillar's logprobs.
        """
        key_to_probs: dict[str, dict[int, float]] = {}
        if not content_tokens:
            return key_to_probs

        token_texts = [t.get("token", "") for t in content_tokens]
        full_text = "".join(token_texts)
        token_spans: list[tuple[int, int]] = []
        pos = 0
        for tok_text in token_texts:
            token_spans.append((pos, pos + len(tok_text)))
            pos += len(tok_text)

        matches: list[tuple[str, Any]] = []
        for key in (cfg["key"] for cfg in PILLAR_CONFIGS.values()):
            m = re.search(rf"{key}\s*:", full_text)
            if m:
                matches.append((key, m))
        matches.sort(key=lambda km: km[1].start())

        for i, (key, m) in enumerate(matches):
            window_end = matches[i + 1][1].start() if i + 1 < len(matches) else len(full_text)
            for idx, (start, end) in enumerate(token_spans):
                if end <= m.end() or start >= window_end:
                    continue
                prob_map = self._digit_probabilities(content_tokens[idx].get("top_logprobs", []))
                if prob_map:
                    total_p = sum(prob_map.values())
                    key_to_probs[key] = {k: round(v / total_p, 4) for k, v in prob_map.items()}
                    break
        return key_to_probs

    @staticmethod
    def _digit_probabilities(top_logprobs: list) -> dict[int, float]:
        """Extracts P(rating) for 1-5 candidates from one token position's top_logprobs.

        Uses the trailing digit so merged candidates such as ": 4" or "P1: 4" still resolve.
        """
        prob_map: dict[int, float] = {}
        for item in top_logprobs:
            t_str = item.get("token", "").strip()
            if not t_str or len(t_str) > 8:
                continue
            digits = re.findall(r"\d", t_str)
            if digits and int(digits[-1]) in (1, 2, 3, 4, 5):
                prob_map[int(digits[-1])] = math.exp(item.get("logprob", 0.0))
        return prob_map

    def _extract_digit_fallback(self, text: str, key: str) -> float:
        """Fallback digit extractor from raw text."""
        m = re.search(rf"{key}\s*:\s*([1-5])", text)
        if m:
            return float(m.group(1))
        return 3.0

    def _fallback_pillar_scoring(
        self, pillar_summary: dict[str, str], total_max_marks: float
    ) -> dict[str, PillarGEvalScore]:
        """Safe heuristic fallback if Bedrock API call encounters transient network error."""
        results = {}
        for pillar_type, cfg in PILLAR_CONFIGS.items():
            weight = cfg["weight"]
            max_pillar_marks = total_max_marks * weight
            results[pillar_type.value] = PillarGEvalScore(
                pillar=pillar_type,
                pillar_name=cfg["name"],
                weight_pct=weight,
                max_marks=round(max_pillar_marks, 2),
                discrete_probabilities={3: 1.0},
                raw_expected_rating=3.0,
                calibrated_score=round(0.45 * max_pillar_marks, 3),
                feedback=pillar_summary.get(pillar_type.value, "Standard baseline evaluated."),
                scoring_method="heuristic_fallback",
            )
        return results
