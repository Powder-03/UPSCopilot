"""Unit tests for the calibrated UPSC Mains Evaluation Engine."""
import pytest
from src.evaluation.geval_scorer import calibrate_rating_to_upsc_marks
from src.evaluation.engine import _extract_json_dict
from src.models.enums import PillarType, PresentationArchetype, UPSCPerformanceBand
from src.models.schema import PillarGEvalScore


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
    assert _extract_json_dict(raw_clean) == {"is_off_topic": False, "score": 5}

    markdown_fenced = 'Here is the response:\n```json\n{"status": "ok", "value": 42}\n```\nExtra commentary.'
    assert _extract_json_dict(markdown_fenced) == {"status": "ok", "value": 42}

    trailing_garbage = '{"a": 1, "b": 2} Note: Hope this was helpful!'
    assert _extract_json_dict(trailing_garbage) == {"a": 1, "b": 2}


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
