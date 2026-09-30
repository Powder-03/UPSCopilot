"""Demonstration CLI: runs the evaluation engine over three answer archetypes.

Shows the calibration spread on one question — a topper benchmark answer, an average
paragraph answer, and a factually-correct but off-topic answer (which must trip the
Hard Demand Relevance Gatekeeper).

The question and answers live in tests/fixtures/sample_answers.json, so this script
stays a thin entry point.

Usage:
    uv run python scripts/evaluate/sample.py [--fixture PATH]
"""
import argparse
import json
from pathlib import Path

from src.evaluation.engine import UPSCEvaluationEngine
from src.models.evaluation import EvaluationResult
from src.utils.cli import configure_console, setup_logging

DEFAULT_FIXTURE = Path("tests/fixtures/sample_answers.json")


def print_evaluation_card(title: str, res: EvaluationResult) -> None:
    """Prints the full examiner scorecard for one evaluated answer."""
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {title.upper()}")
    print("=" * 80)
    print(f"Question: {res.question}")
    print(f"\nFinal Marks Awarded: {res.total_score:.2f} / {res.max_marks} ({res.percentage}%)")
    print(f"Performance Band:    {res.performance_band.value}")
    print(f"Off-Topic Status:    {'[!] OFF-TOPIC ALERT' if res.is_off_topic else 'ON-TOPIC'}")
    print(f"Relevance Gate:      {res.demand_relevance_gate:.2f}")

    print("\n--- 6-PILLAR G-EVAL CONTINUOUS SCORECARD ---")
    for _key, p in res.pillars.items():
        probs_str = ", ".join(f"{score}: {prob*100:.1f}%" for score, prob in p.discrete_probabilities.items())
        print(f"• {p.pillar_name:<42} | Score: {p.calibrated_score:5.2f} / {p.max_marks:4.2f} | E[S]: {p.raw_expected_rating:.2f}/5 | Probs: [{probs_str}]")

    print("\n--- PRESENTATION AUDIT ---")
    print(f"Detected Format:   {res.presentation.detected_archetype.value}")
    print(f"Visual Density:    {res.presentation.visual_density_score} / 10.0")
    print(f"Diagrams Found:    {res.presentation.diagrams_and_tables_found}")
    print(f"Critique:          {res.presentation.examiner_critique}")
    print(f"Topper Reformat:   {res.presentation.topper_reformatting_tip}")

    print("\n--- CITATION & LEGAL GROUNDING AUDIT ---")
    print("Mandatory KB Anchors:")
    for c in res.citation_audit.mandatory_kb_anchors:
        print(f"  - [{c.status.value.upper()}] {c.name}: {c.notes}")
    if res.citation_audit.open_world_credits:
        print("Open-World Credited Points:")
        for c in res.citation_audit.open_world_credits:
            print(f"  - [{c.status.value.upper()}] {c.name}: {c.notes}")
    if res.citation_audit.hallucinated_citations:
        print("Hallucinated / Wrong Citations:")
        for c in res.citation_audit.hallucinated_citations:
            print(f"  ! [{c.status.value.upper()}] {c.name}: {c.notes}")

    print("\n--- EXAMINER ACTIONABLE FEEDBACK ---")
    print("Strengths:")
    for s in res.strengths:
        print(f"  + {s}")
    print("Weaknesses / Gaps:")
    for w in res.weaknesses:
        print(f"  - {w}")
    print("Topper Action Plan (+1.5 Marks Roadmap):")
    for t in res.topper_action_plan:
        print(f"  * {t}")
    print("=" * 80 + "\n")


def run_demo(fixture: Path) -> None:
    """Evaluates every answer archetype in the fixture against its shared question."""
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    question = payload["question"]
    max_marks = float(payload.get("max_marks", 10.0))

    engine = UPSCEvaluationEngine()

    for entry in payload["answers"]:
        print(f"\nEvaluating {entry['label']}...")
        result = engine.evaluate_answer(question, entry["text"], max_marks=max_marks)
        print_evaluation_card(entry["label"], result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the sample answer archetypes.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="Sample answers JSON fixture")
    args = parser.parse_args()

    configure_console()
    setup_logging()

    if not args.fixture.exists():
        raise SystemExit(f"Error: fixture not found: {args.fixture}")

    run_demo(args.fixture)


if __name__ == "__main__":
    main()
