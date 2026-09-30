"""Opt-in scoring-consistency test.

Evaluates the SAME copy multiple times against live AWS Bedrock and asserts the total score is
stable across runs. This is where the "run the copy 4-5 times to check consistency" workflow
lives - the multi-run CLI is at `scripts/evaluate/topper_copy_evaluation.py`.

Scoring uses temperature=0.0 + logprob-weighted G-Eval, so runs should be near-identical; this
test guards against surprising run-to-run variance (model non-determinism, throttling fallbacks).

Run it explicitly (it is skipped by default because it costs live API calls):
    uv run pytest tests/test_copy_consistency.py -m bedrock --run-bedrock -s
"""
import json
import statistics
from pathlib import Path
from typing import Any

import pytest
from src.evaluation.engine import UPSCEvaluationEngine

# Small copy (2 questions) to bound API cost: NUM_RUNS x questions x 3 Bedrock calls.
CONSISTENCY_FIXTURE = "topper_copy.json"
NUM_RUNS = 5
# Tolerated population std-dev of the total score across runs, as a fraction of total max marks.
TOLERANCE_FRACTION = 0.10


def _load_copy(fixtures_dir: Path, name: str) -> list[dict[str, Any]]:
    with open(fixtures_dir / name, encoding="utf-8") as f:
        return json.load(f)


def _score_copy_once(engine: UPSCEvaluationEngine, copy: list[dict[str, Any]]) -> tuple[float, float]:
    """Evaluates every question in the copy once; returns (total_score, total_max)."""
    total_score = 0.0
    total_max = 0.0
    for item in copy:
        max_marks = float(item.get("max_marks", 10.0))
        result = engine.evaluate_answer(
            question=item["question"],
            candidate_answer=item["candidate_answer"],
            max_marks=max_marks,
        )
        total_score += result.total_score
        total_max += max_marks
    return round(total_score, 2), total_max


@pytest.mark.bedrock
def test_copy_scoring_is_consistent_across_runs(fixtures_dir: Path) -> None:
    copy = _load_copy(fixtures_dir, CONSISTENCY_FIXTURE)
    engine = UPSCEvaluationEngine()

    totals: list[float] = []
    total_max = 0.0
    for run in range(1, NUM_RUNS + 1):
        total, total_max = _score_copy_once(engine, copy)
        totals.append(total)
        print(f"\n[consistency] run {run}/{NUM_RUNS}: total={total:.2f}/{total_max:.0f}")

    mean = statistics.mean(totals)
    std = statistics.pstdev(totals)
    spread = max(totals) - min(totals)
    print(f"\n[consistency] totals={totals}")
    print(f"[consistency] mean={mean:.2f} std={std:.2f} spread={spread:.2f}")

    tolerance = TOLERANCE_FRACTION * total_max
    assert std <= tolerance, (
        f"Scoring inconsistent across {NUM_RUNS} runs: std={std:.2f} exceeds tolerance "
        f"{tolerance:.2f} ({TOLERANCE_FRACTION:.0%} of {total_max:.0f}). totals={totals}"
    )
