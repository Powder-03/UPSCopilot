"""Unit tests for the calibrated UPSC Mains Evaluation Engine."""
import math

import pytest
from pydantic import ValidationError
from src.evaluation.engine import (
    _classify_performance_band,
    _match_enum,
    _parse_citation_status,
    _parse_diagnostic,
)
from src.evaluation.geval_scorer import (
    PILLAR_CONFIGS,
    BedrockGEvalScorer,
    VertexGEvalScorer,
    calibrate_rating_to_upsc_marks,
)
from src.evaluation.prompt_templates import build_geval_scoring_prompt
from src.models.enums import (
    CitationStatus,
    DemandStatus,
    DirectiveType,
    PillarType,
    PresentationArchetype,
    UPSCPerformanceBand,
)
from src.models.evaluation import PillarGEvalScore, PresentationEvaluation
from src.models.exceptions import ModelInvocationError, RatingExtractionError
from src.utils.json import clean_json_text, extract_json_dict


def test_pillar_weights_sum_to_one():
    """The 6 canonical pillar weights must always total exactly 100%."""
    weights = {cfg["key"]: cfg["weight"] for cfg in PILLAR_CONFIGS.values()}
    assert len(PILLAR_CONFIGS) == 6
    assert abs(sum(weights.values()) - 1.0) < 1e-9, f"Pillar weights do not sum to 1.0: {weights}"
    assert weights["P4"] == 0.20  # Grounded Legal Citations
    assert weights["P6"] == 0.05  # Introduction & Context Setting


def test_calibrate_rating_to_upsc_marks():
    """Verify authentic linear calibration from 1-5 scale to UPSC percentage bands."""
    # Rating 1.0 (Poor) -> 25% of 10 = 2.5 marks
    assert calibrate_rating_to_upsc_marks(1.0, 10.0) == 2.5

    # Rating 3.0 (Average) -> 45% of 10 = 4.5 marks
    assert calibrate_rating_to_upsc_marks(3.0, 10.0) == 4.5

    # Rating 4.0 (Good) -> 55% of 10 = 5.5 marks
    assert calibrate_rating_to_upsc_marks(4.0, 10.0) == 5.5

    # Rating 5.0 (Topper Benchmark) -> 65% of 10 = 6.5 marks
    assert calibrate_rating_to_upsc_marks(5.0, 10.0) == 6.5

    # Scaling with 15-markers (max_marks = 15.0)
    # Rating 3.0 -> 45% of 15 = 6.75 marks
    assert calibrate_rating_to_upsc_marks(3.0, 15.0) == 6.75
    # Rating 5.0 -> 65% of 15 = 9.75 marks
    assert calibrate_rating_to_upsc_marks(5.0, 15.0) == 9.75


def test_extract_json_dict():
    """Verify robust JSON extraction from raw text and markdown fences."""
    raw_clean = '{"is_off_topic": false, "score": 5}'
    assert extract_json_dict(raw_clean) == {"is_off_topic": False, "score": 5}

    markdown_fenced = 'Here is the response:\n```json\n{"status": "ok", "value": 42}\n```\nExtra commentary.'
    assert extract_json_dict(markdown_fenced) == {"status": "ok", "value": 42}

    trailing_garbage = '{"a": 1, "b": 2} Note: Hope this was helpful!'
    assert extract_json_dict(trailing_garbage) == {"a": 1, "b": 2}


def test_clean_json_text_strips_commentary():
    """The judge-path helper must reduce fenced/annotated LLM output to compact JSON."""
    fenced = 'Sure!\n```json\n{"status": "ok"}\n```\nLet me know if you need more.'
    assert clean_json_text(fenced) == '{"status": "ok"}'

    bare = '{"a": 1} trailing note'
    assert clean_json_text(bare) == '{"a": 1}'

    assert clean_json_text("no json here at all") == "no json here at all"
    assert clean_json_text("") == "{}"


def test_pillar_geval_score_math():
    """Verify mathematical expected value sum(s * P(s)) calculation."""
    # Given distribution: P(3)=0.10, P(4)=0.60, P(5)=0.30
    # Expected rating = (3 * 0.1) + (4 * 0.6) + (5 * 0.3) = 0.3 + 2.4 + 1.5 = 4.20
    probs = {3: 0.10, 4: 0.60, 5: 0.30}
    expected_rating = sum(s * p for s, p in probs.items())
    assert abs(expected_rating - 4.20) < 1e-4

    max_marks = 2.5
    calibrated = calibrate_rating_to_upsc_marks(expected_rating, max_marks)
    # expected_rating 4.2 -> pct = 0.15 + (4.2 * 0.10) = 0.57 (57%)
    # 57% of 2.5 = 1.425
    assert abs(calibrated - 1.425) < 1e-3

    pillar = PillarGEvalScore(
        pillar=PillarType.DEMAND_FULFILLMENT,
        pillar_name="Demand Fulfillment",
        weight_pct=0.25,
        max_marks=max_marks,
        discrete_probabilities=probs,
        raw_expected_rating=expected_rating,
        calibrated_score=calibrated,
        feedback="High demand compliance.",
    )
    assert pillar.raw_expected_rating == 4.2
    assert pillar.calibrated_score == 1.425


def test_classify_performance_band_boundaries():
    """Rating 3.0 calibrates to exactly 45%, which must land in the Average band (35-45%)."""
    assert _classify_performance_band(34.9) == UPSCPerformanceBand.NEEDS_FOUNDATION
    assert _classify_performance_band(35.0) == UPSCPerformanceBand.AVERAGE
    assert _classify_performance_band(45.0) == UPSCPerformanceBand.AVERAGE
    assert _classify_performance_band(45.1) == UPSCPerformanceBand.GOOD
    assert _classify_performance_band(55.9) == UPSCPerformanceBand.GOOD
    assert _classify_performance_band(56.0) == UPSCPerformanceBand.TOPPER


def test_parse_citation_status_never_reads_not_found_as_found():
    assert _parse_citation_status("not_found") == CitationStatus.MANDATORY_MISSING
    assert _parse_citation_status("not found") == CitationStatus.MANDATORY_MISSING
    assert _parse_citation_status("mandatory_missing") == CitationStatus.MANDATORY_MISSING
    assert _parse_citation_status("hallucinated_or_wrong") == CitationStatus.HALLUCINATED_OR_WRONG
    assert _parse_citation_status("mandatory_found") == CitationStatus.MANDATORY_FOUND
    assert _parse_citation_status("found") == CitationStatus.MANDATORY_FOUND
    assert _parse_citation_status("") == CitationStatus.MANDATORY_MISSING


def test_match_enum_normalizes_and_prefers_specific_values():
    assert _match_enum(DirectiveType, "to what extent", None) == DirectiveType.TO_WHAT_EXTENT
    assert _match_enum(DirectiveType, "critically analyze and discuss", None) == DirectiveType.CRITICALLY_ANALYZE
    assert _match_enum(DirectiveType, "Discuss", None) == DirectiveType.DISCUSS
    assert _match_enum(DirectiveType, "gibberish", None) is None
    assert _match_enum(DemandStatus, "fully_addressed", DemandStatus.PARTIALLY_ADDRESSED) == DemandStatus.FULLY_ADDRESSED
    assert _match_enum(DemandStatus, "omitted", DemandStatus.PARTIALLY_ADDRESSED) == DemandStatus.OMITTED
    assert (
        _match_enum(PresentationArchetype, "hybrid-diagrammatic", PresentationArchetype.PARAGRAPH_HEAVY)
        == PresentationArchetype.HYBRID_DIAGRAMMATIC
    )


def test_geval_prompt_includes_cot_trail():
    """Call 2 must receive the full Call 1 reasoning trail, not just the pillar summaries."""
    trail = "1. Decomposed the demand into three sub-parts. 2. Compared against KB grounding."
    prompt = build_geval_scoring_prompt(
        question="Examine Article 123 safeguards.",
        candidate_answer="Ordinances lapse after six weeks.",
        cot_reasoning_trail=trail,
        pillar_summary={"demand_fulfillment": "Strong coverage"},
    )
    assert trail in prompt
    assert "P1:" in prompt and "P5:" in prompt and "P6:" in prompt


def test_presentation_bonus_is_bounded():
    """An unbounded LLM-returned bonus must be rejected by the schema."""
    with pytest.raises(ValidationError):
        PresentationEvaluation(
            detected_archetype=PresentationArchetype.PARAGRAPH_HEAVY,
            visual_density_score=5.0,
            presentation_bonus=2.0,
            examiner_critique="x",
            topper_reformatting_tip="y",
        )


def _digit_top_logprobs(digit: str):
    """Builds a synthetic top_logprobs list centered on `digit` (0.7/0.2/0.1 split)."""
    others = [d for d in "54321" if d != digit][:2]
    return [
        {"token": f" {digit}", "logprob": math.log(0.7)},
        {"token": f" {others[0]}", "logprob": math.log(0.2)},
        {"token": f" {others[1]}", "logprob": math.log(0.1)},
    ]


def _build_geval_tokens(split_keys: bool):
    """Synthetic Kimi-style logprob tokens for 'P1: 4 / P2: 3 / P3: 4 / P4: 5 / P5: 3 / P6: 4'."""
    ratings = [("P1", "4"), ("P2", "3"), ("P3", "4"), ("P4", "5"), ("P5", "3"), ("P6", "4")]
    tokens = []
    for i, (key, digit) in enumerate(ratings):
        if i > 0:
            tokens.append({"token": "\n", "logprob": -0.01, "top_logprobs": []})
        if split_keys:
            for chunk in (key[0], key[1], ":"):
                tokens.append({"token": chunk, "logprob": -0.01, "top_logprobs": []})
        else:
            tokens.append({"token": f"{key}:", "logprob": -0.01, "top_logprobs": []})
        tokens.append({
            "token": f" {digit}",
            "logprob": math.log(0.7),
            "top_logprobs": _digit_top_logprobs(digit),
        })
    return tokens


def test_parse_logprob_pillars_handles_merged_and_split_tokenization():
    """E[S] must be identical whether the model emits 'P1:' as one token or 'P'+'1'+':'."""
    scorer = BedrockGEvalScorer()
    output_text = "P1: 4\nP2: 3\nP3: 4\nP4: 5\nP5: 3\nP6: 4"
    for split_keys in (False, True):
        pillars = scorer._parse_logprob_pillars(
            _build_geval_tokens(split_keys), output_text, {}, 10.0
        )
        assert all(p.scoring_method == "logprob" for p in pillars.values())
        p1 = pillars[PillarType.DEMAND_FULFILLMENT.value]
        assert p1.discrete_probabilities == {4: 0.7, 5: 0.2, 3: 0.1}
        assert abs(p1.raw_expected_rating - 4.1) < 1e-3
        p5 = pillars[PillarType.CONCLUSION_WAY_FORWARD.value]
        assert p5.discrete_probabilities == {3: 0.7, 5: 0.2, 4: 0.1}
        assert abs(p5.raw_expected_rating - 3.5) < 1e-3
        p6 = pillars[PillarType.INTRODUCTION.value]
        assert p6.weight_pct == 0.05
        assert abs(p6.raw_expected_rating - 4.1) < 1e-3


def test_parse_logprob_pillars_merged_score_token_uses_trailing_digit():
    """A candidate token like 'P1: 4' must still yield P(4) via its trailing digit."""
    scorer = BedrockGEvalScorer()
    tokens = [{
        "token": "P1: 4",
        "logprob": math.log(0.7),
        "top_logprobs": [
            {"token": "P1: 4", "logprob": math.log(0.7)},
            {"token": "P1: 5", "logprob": math.log(0.2)},
            {"token": "P1: 3", "logprob": math.log(0.1)},
        ],
    }]
    output_text = "P1: 4\nP2: 3\nP3: 4\nP4: 5\nP5: 3\nP6: 4"
    pillars = scorer._parse_logprob_pillars(tokens, output_text, {}, 10.0)
    p1 = pillars[PillarType.DEMAND_FULFILLMENT.value]
    assert p1.scoring_method == "logprob"
    assert p1.discrete_probabilities == {4: 0.7, 5: 0.2, 3: 0.1}


def test_parse_logprob_pillars_text_fallback_without_logprobs():
    """Without top_logprobs the scorer must degrade visibly to text-parsed point estimates."""
    scorer = BedrockGEvalScorer()
    output_text = "P1: 4\nP2: 3\nP3: 4\nP4: 5\nP5: 3\nP6: 4"
    tokens = [
        {"token": t, "logprob": 0.0}
        for t in ["P1: 4", "\n", "P2: 3", "\n", "P3: 4", "\n", "P4: 5", "\n", "P5: 3", "\n", "P6: 4"]
    ]
    pillars = scorer._parse_logprob_pillars(tokens, output_text, {}, 10.0)
    assert all(p.scoring_method == "text_fallback" for p in pillars.values())
    p1 = pillars[PillarType.DEMAND_FULFILLMENT.value]
    assert p1.discrete_probabilities == {4: 1.0}
    assert p1.raw_expected_rating == 4.0
    assert pillars[PillarType.CONCLUSION_WAY_FORWARD.value].raw_expected_rating == 3.0
    assert pillars[PillarType.INTRODUCTION.value].raw_expected_rating == 4.0


def test_missing_pillar_raises_rating_extraction_error():
    """Missing pillar in output text must raise RatingExtractionError instead of defaulting to 3.0."""
    scorer = BedrockGEvalScorer()
    # P6 is missing from output
    output_text = "P1: 4\nP2: 3\nP3: 4\nP4: 5\nP5: 3"
    tokens = [
        {"token": t, "logprob": 0.0}
        for t in ["P1: 4", "\n", "P2: 3", "\n", "P3: 4", "\n", "P4: 5", "\n", "P5: 3"]
    ]
    with pytest.raises(RatingExtractionError) as exc_info:
        scorer._parse_logprob_pillars(tokens, output_text, {}, 10.0)
    assert "P6" in str(exc_info.value)


def test_scorer_fails_fast_on_api_error():
    """Scorer must raise ModelInvocationError on API errors instead of returning fake 45% marks."""
    from unittest.mock import MagicMock

    # Bedrock failure
    mock_bedrock = MagicMock()
    mock_bedrock.invoke_model.side_effect = RuntimeError("AWS Bedrock connection timed out")
    bedrock_scorer = BedrockGEvalScorer(client=mock_bedrock)
    with pytest.raises(ModelInvocationError) as exc_info:
        bedrock_scorer.score_pillars("Q", "A", "CoT", {})
    assert "Bedrock G-Eval scoring failed" in str(exc_info.value)

    # Vertex failure
    mock_vertex = MagicMock()
    mock_vertex.models.generate_content.side_effect = RuntimeError("Vertex AI quota exceeded")
    vertex_scorer = VertexGEvalScorer(client=mock_vertex)
    with pytest.raises(ModelInvocationError) as exc_info:
        vertex_scorer.score_pillars("Q", "A", "CoT", {})
    assert "Vertex G-Eval scoring failed" in str(exc_info.value)


def test_parse_diagnostic_fails_fast_on_invalid_output():
    """Call 1 diagnostic parser must raise ModelInvocationError if model response is empty or invalid."""
    with pytest.raises(ModelInvocationError):
        _parse_diagnostic({})

    with pytest.raises(ModelInvocationError):
        _parse_diagnostic({"is_off_topic": False})

    with pytest.raises(ModelInvocationError):
        _parse_diagnostic({"cot_reasoning_trail": "trail without pillar summary"})

