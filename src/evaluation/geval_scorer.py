"""G-Eval probability-weighted continuous scoring engine using Bedrock Kimi 2.5 logprobs."""
import json
import math
import logging
from typing import Dict, Tuple, Optional, Any
import boto3
from botocore.exceptions import ClientError

from src.config import settings
from src.models.enums import PillarType, UPSCPerformanceBand
from src.models.schema import PillarGEvalScore
from src.evaluation.prompt_templates import build_geval_scoring_prompt

logger = logging.getLogger(__name__)

# Canonical UPSC pillar weights and descriptions
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
        "weight": 0.25,
        "key": "P4",
    },
    PillarType.CONCLUSION_WAY_FORWARD: {
        "name": "Conclusion & Constructive Way Forward",
        "weight": 0.15,
        "key": "P5",
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

    def __init__(self, model_id: Optional[str] = None, region: Optional[str] = None):
        self.model_id = model_id or settings.bedrock_eval_model_id
        self.region = region or settings.aws_region
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def score_pillars(
        self,
        question: str,
        candidate_answer: str,
        cot_reasoning_trail: str,
        pillar_summary: Dict[str, str],
        max_marks: float = 10.0,
    ) -> Dict[str, PillarGEvalScore]:
        """Runs Call 2: Single-pass G-Eval multi-pillar scoring with authentic token logprobs."""
        prompt = build_geval_scoring_prompt(
            question=question,
            candidate_answer=candidate_answer,
            cot_reasoning_trail=cot_reasoning_trail,
            pillar_summary=pillar_summary,
        )

        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 40,
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
        pillar_summary: Dict[str, str],
        total_max_marks: float,
    ) -> Dict[str, PillarGEvalScore]:
        """Parses output tokens to identify P1..P5 score positions and computes E[S]."""
        results: Dict[str, PillarGEvalScore] = {}

        # Map each P1..P5 key to its extracted discrete probabilities
        key_to_probs: Dict[str, Dict[int, float]] = {}
        
        # Track position in tokens
        i = 0
        while i < len(content_tokens):
            tok_text = content_tokens[i].get("token", "").strip()
            for key in ["P1", "P2", "P3", "P4", "P5"]:
                if key in tok_text and key not in key_to_probs:
                    # Look ahead up to 4 tokens for the score digit
                    for offset in range(1, 4):
                        if i + offset < len(content_tokens):
                            candidate_tok = content_tokens[i + offset]
                            top_lp = candidate_tok.get("top_logprobs", [])
                            prob_map: Dict[int, float] = {}
                            for item in top_lp:
                                t_str = item.get("token", "").strip()
                                if t_str.isdigit() and int(t_str) in [1, 2, 3, 4, 5]:
                                    prob_map[int(t_str)] = math.exp(item.get("logprob", 0.0))
                            
                            if prob_map:
                                total_p = sum(prob_map.values())
                                normalized = {k: round(v / total_p, 4) for k, v in prob_map.items()}
                                key_to_probs[key] = normalized
                                break
            i += 1

        # Build PillarGEvalScore objects
        for pillar_type, cfg in PILLAR_CONFIGS.items():
            key = cfg["key"]
            weight = cfg["weight"]
            max_pillar_marks = total_max_marks * weight
            feedback = pillar_summary.get(pillar_type.value, "Evaluated according to UPSC rubric.")

            probs = key_to_probs.get(key)
            if probs:
                expected_r = sum(score * p for score, p in probs.items())
            else:
                # Text parsing fallback if token scanning missed the position
                expected_r = self._extract_digit_fallback(output_text, key)
                probs = {int(round(expected_r)): 1.0}

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
            )

        return results

    def _extract_digit_fallback(self, text: str, key: str) -> float:
        """Fallback digit extractor from raw text."""
        import re
        m = re.search(rf"{key}\s*:\s*([1-5])", text)
        if m:
            return float(m.group(1))
        return 3.0

    def _fallback_pillar_scoring(
        self, pillar_summary: Dict[str, str], total_max_marks: float
    ) -> Dict[str, PillarGEvalScore]:
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
            )
        return results
