"""Evaluate a full UPSC answer copy from a JSON fixture with the calibrated evaluation engine.

Replaces the former one-off `evaluate_parsed_copy.py` / `evaluate_topper_copy.py` scripts:
the question set is now data, not hardcoded Python, and the checkpoint/resume behaviour is
shared by every copy.

Usage:
    uv run python scripts/evaluate/answer_copy.py --input tests/fixtures/topper_copy.json
    uv run python scripts/evaluate/answer_copy.py --input tests/fixtures/gs2_copy_parsed.json ^
        --output tests/outputs/gs2_evaluation_results.json --title "GS-II Test Copy"

Input format: a JSON list of objects with `question`, `candidate_answer`, `max_marks`
and an optional `q_num` (defaults to the 1-based position in the list).
"""
import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.engine import UPSCEvaluationEngine
from src.models.evaluation import EvaluationResult
from src.utils.cli import configure_console, setup_logging

# Loggers that are pure noise during a long copy evaluation run.
QUIET_LOGGERS = ("src.kb.corpus_loader", "src.kb.retriever", "src.evaluation.geval_scorer")


def load_questions(input_path: Path) -> list[dict[str, Any]]:
    """Loads a copy fixture and normalizes `q_num` / `max_marks`."""
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    questions: list[dict[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        questions.append(
            {
                "q_num": int(item.get("q_num", idx)),
                "max_marks": float(item.get("max_marks", 10.0)),
                "question": item["question"],
                "candidate_answer": item["candidate_answer"],
            }
        )
    return questions


def load_checkpoint(output_path: Path | None) -> list[dict[str, Any]]:
    """Loads previously evaluated results so an interrupted run can resume."""
    if output_path is None or not output_path.exists():
        return []
    try:
        with open(output_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_checkpoint(output_path: Path | None, results: list[dict[str, Any]]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def pillar_line(result: dict[str, Any]) -> str:
    """One-line summary of the 6 G-Eval pillar scores."""
    pillars = result.get("pillars") or {}
    parts = [
        f"{p.get('pillar_name', name).split()[0]}: {p.get('calibrated_score', 0.0):.2f}/{p.get('max_marks', 0.0):.2f}"
        for name, p in pillars.items()
    ]
    return " | ".join(parts) if parts else "n/a"


def print_compact_card(q_num: int, max_marks: float, question: str, result: dict[str, Any]) -> None:
    total_score = float(result.get("total_score", 0.0))
    pct = float(result.get("percentage", (total_score / max_marks) * 100 if max_marks else 0.0))
    band = str(result.get("performance_band", "")).upper()
    presentation = result.get("presentation") or {}
    strengths = result.get("strengths") or [""]
    weaknesses = result.get("weaknesses") or [""]
    action_plan = result.get("topper_action_plan") or [""]
    status = "OFF-TOPIC" if result.get("is_off_topic") else "ON-TOPIC"

    print("\n" + "=" * 78)
    print(f"QUESTION {q_num:02d} [{int(max_marks)} Marks] | {status}")
    print(f"  {question[:96]}{'...' if len(question) > 96 else ''}")
    print("=" * 78)
    print(f"Score:   {total_score:.2f} / {int(max_marks)} ({pct:.0f}%) | Band: {band}")
    print(f"Pillars: {pillar_line(result)}")
    print(
        f"Format:  {presentation.get('detected_archetype', 'n/a')} "
        f"(visual density: {presentation.get('visual_density_score', 'n/a')}/10)"
    )
    print(f"+ Strength:   {strengths[0]}")
    print(f"- Key Gap:    {weaknesses[0]}")
    print(f"* Topper Tip: {action_plan[0]}")


def overall_band(pct: float) -> str:
    """Maps a whole-copy percentage onto the selection-trajectory bands."""
    if pct >= 55.0:
        return "TOPPER LEVEL (Rank 1 - 50 Trajectory)"
    if pct >= 45.0:
        return "GOOD / SELECTION ZONE (Rank 50 - 300 Trajectory)"
    if pct >= 35.0:
        return "AVERAGE / INTERVIEW CALL BOUNDARY"
    return "BELOW AVERAGE / NEEDS FUNDAMENTAL VALUE ADD"


def print_final_scorecard(results: list[dict[str, Any]], title: str) -> None:
    total_score = sum(float(r["result"].get("total_score", 0.0)) for r in results)
    total_max = sum(float(r.get("max_marks", 0.0)) for r in results)
    pct = (total_score / total_max) * 100 if total_max > 0 else 0.0

    print("\n" + "=" * 78)
    print(f"FINAL CONSOLIDATED SCORECARD: {title.upper()}")
    print("=" * 78)
    print(f"Questions Evaluated: {len(results)}")
    print(f"Total Score:         {total_score:.2f} / {total_max:.0f}  ({pct:.1f}%)")
    print(f"Candidate Band:      {overall_band(pct)}")
    print("=" * 78 + "\n")


def evaluate_copy(input_path: Path, output_path: Path | None, title: str) -> None:
    questions = load_questions(input_path)
    evaluated = load_checkpoint(output_path)
    cached_by_num = {r.get("q_num"): r for r in evaluated if "q_num" in r}

    engine = UPSCEvaluationEngine()

    print("\n" + "=" * 78)
    print(f"EVALUATING: {title}")
    print(f"Input: {input_path} | Questions: {len(questions)} | Already evaluated: {len(cached_by_num)}")
    print("=" * 78)

    for q in questions:
        q_num, max_marks = q["q_num"], q["max_marks"]
        cached = cached_by_num.get(q_num)
        if cached:
            print(f"\n>>> Question {q_num} served from checkpoint.")
            print_compact_card(q_num, max_marks, q["question"], cached["result"])
            continue

        print(f"\n>>> Processing Question {q_num} ({int(max_marks)} Marks)...")
        result: EvaluationResult = engine.evaluate_answer(
            question=q["question"],
            candidate_answer=q["candidate_answer"],
            max_marks=max_marks,
        )
        record = {
            "q_num": q_num,
            "max_marks": max_marks,
            "question": q["question"],
            "result": result.model_dump(),
        }
        evaluated.append(record)
        cached_by_num[q_num] = record
        save_checkpoint(output_path, evaluated)
        print_compact_card(q_num, max_marks, q["question"], record["result"])

    print_final_scorecard(evaluated, title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a UPSC answer copy from a JSON fixture.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the copy fixture JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional checkpoint file; enables resume across interrupted runs",
    )
    parser.add_argument("--title", default=None, help="Report label (defaults to the input file stem)")
    args = parser.parse_args()

    configure_console()
    setup_logging(quiet=QUIET_LOGGERS)

    if not args.input.exists():
        raise SystemExit(f"Error: input fixture not found: {args.input}")

    evaluate_copy(args.input, args.output, args.title or args.input.stem)


if __name__ == "__main__":
    main()
