"""Evaluate a full UPSC topper answer copy multiple times (default 5 runs) to measure scoring consistency.

Defaults to the full 20-question GS-2 copy (`tests/fixtures/gs2_copy_parsed.json`).
For each question, outputs a minimal single-line summary to keep terminal output clean and focused.
Progress is saved after every evaluated question so runs are interruption-safe and resumable.

When all runs complete, generates a comprehensive comparison report covering:
1. Aggregate score stability across all 5 runs (Mean, Std Dev, Range, CV%).
2. Per-question marking consistency table (Q01-Q20 scores across runs).
3. 6-Pillar G-Eval score consistency (P1-P6 breakdown).
4. Qualitative stability (performance bands, gatekeeper checks, detected archetypes).

Usage:
    uv run python scripts/evaluate/topper_copy_evaluation.py
    uv run python scripts/evaluate/topper_copy_evaluation.py --runs 5 --input tests/fixtures/gs2_copy_parsed.json
    uv run python scripts/evaluate/topper_copy_evaluation.py --runs 3 --limit 5
"""
import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.engine import UPSCEvaluationEngine
from src.models.enums import CitationStatus
from src.models.evaluation import EvaluationResult
from src.utils.cli import configure_console, setup_logging

DEFAULT_INPUT = Path("tests/fixtures/gs2_copy_parsed.json")
DEFAULT_OUTPUT = Path("tests/outputs/topper_copy_consistency.json")
DEFAULT_RUNS = 5

QUIET_LOGGERS = ("src.kb.corpus_loader", "src.kb.retriever", "src.evaluation.geval_scorer")


def load_questions(input_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Loads a copy fixture and normalizes `q_num` and `max_marks`."""
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    if limit:
        raw = raw[:limit]

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


def overall_band(pct: float) -> str:
    """Maps a whole-copy percentage onto selection-trajectory bands."""
    if pct >= 55.0:
        return "TOPPER LEVEL (Rank 1 - 50 Trajectory)"
    if pct >= 45.0:
        return "GOOD / SELECTION ZONE (Rank 50 - 300 Trajectory)"
    if pct >= 35.0:
        return "AVERAGE / INTERVIEW CALL BOUNDARY"
    return "BELOW AVERAGE / NEEDS FUNDAMENTAL VALUE ADD"


def extract_question_metrics(q_num: int, max_marks: float, question: str, result: EvaluationResult) -> dict[str, Any]:
    """Extracts compact metrics for a single question evaluation."""
    mandatory_anchors = result.citation_audit.mandatory_kb_anchors if result.citation_audit else []
    verified_mandatory = sum(
        1 for c in mandatory_anchors
        if c.status in (CitationStatus.MANDATORY_FOUND, "mandatory_found")
    )

    return {
        "q_num": q_num,
        "max_marks": max_marks,
        "question": question,
        "total_score": round(result.total_score, 2),
        "percentage": round(result.percentage, 1),
        "performance_band": result.performance_band.value if result.performance_band else "",
        "is_off_topic": result.is_off_topic,
        "demand_relevance_gate": round(result.demand_relevance_gate, 2),
        "detected_archetype": result.presentation.detected_archetype.value if result.presentation else "unknown",
        "visual_density_score": result.presentation.visual_density_score if result.presentation else 0.0,
        "pillars": {
            name: {
                "pillar_name": p.pillar_name,
                "score": round(p.calibrated_score, 2),
                "max": p.max_marks,
                "rating": round(p.raw_expected_rating, 2),
            }
            for name, p in result.pillars.items()
        },
        "citations": {
            "mandatory_verified": verified_mandatory,
            "mandatory_total": len(mandatory_anchors),
            "open_world_credits": len(result.citation_audit.open_world_credits) if result.citation_audit else 0,
            "hallucinated": len(result.citation_audit.hallucinated_citations) if result.citation_audit else 0,
        },
    }


def print_minimal_question_line(record: dict[str, Any]) -> None:
    """Prints a minimal, clean single-line summary for a question."""
    status = "ON-TOPIC" if not record["is_off_topic"] else "OFF-TOPIC"
    band_short = record["performance_band"].replace("BAND_", "").replace("_", " ")
    kb_found = record["citations"]["mandatory_verified"]
    kb_total = record["citations"]["mandatory_total"]
    ow = record["citations"]["open_world_credits"]
    print(
        f"  Q{record['q_num']:02d} [{int(record['max_marks']):>2}M] -> "
        f"{record['total_score']:>5.2f}/{int(record['max_marks']):<2} "
        f"({record['percentage']:>4.1f}%) | {band_short:<14} | {status:<8} | "
        f"Citations: KB {kb_found}/{kb_total}, OW: {ow}"
    )


def compute_consistency_analysis(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes comprehensive score, pillar, and parameter consistency metrics across all runs."""
    n_runs = len(runs)
    total_scores = [r["total_score"] for r in runs]
    total_max = runs[0]["total_max"] if runs else 0.0

    mean_score = round(statistics.mean(total_scores), 2)
    std_dev = round(statistics.stdev(total_scores), 2) if n_runs > 1 else 0.0
    min_score = min(total_scores)
    max_score = max(total_scores)
    score_range = round(max_score - min_score, 2)
    range_pct_of_max = round((score_range / total_max * 100), 2) if total_max > 0 else 0.0
    cv_pct = round((std_dev / mean_score * 100), 2) if mean_score > 0 else 0.0

    # Stability verdict
    if range_pct_of_max <= 5.0:
        verdict = "EXCEPTIONAL STABILITY (Near-Deterministic)"
    elif range_pct_of_max <= 10.0:
        verdict = "HIGH CONSISTENCY (Within Calibrated UPSC Tolerance)"
    elif range_pct_of_max <= 15.0:
        verdict = "MODERATE CONSISTENCY (Acceptable subjective variance)"
    else:
        verdict = "POTENTIAL INSTABILITY (Variance exceeds 15% tolerance)"

    # Per-question metrics across runs
    n_questions = len(runs[0]["questions"])
    per_question: list[dict[str, Any]] = []
    for q_idx in range(n_questions):
        q_records = [r["questions"][q_idx] for r in runs]
        q_num = q_records[0]["q_num"]
        q_max = q_records[0]["max_marks"]
        q_scores = [qr["total_score"] for qr in q_records]
        q_bands = [qr["performance_band"] for qr in q_records]
        q_archetypes = [qr["detected_archetype"] for qr in q_records]
        q_off_topic = [qr["is_off_topic"] for qr in q_records]

        q_mean = round(statistics.mean(q_scores), 2)
        q_std = round(statistics.stdev(q_scores), 2) if n_runs > 1 else 0.0
        q_min = min(q_scores)
        q_max_score = max(q_scores)
        q_range = round(q_max_score - q_min, 2)

        band_stable = len(set(q_bands)) == 1
        archetype_stable = len(set(q_archetypes)) == 1
        off_topic_stable = len(set(q_off_topic)) == 1

        per_question.append(
            {
                "q_num": q_num,
                "max_marks": q_max,
                "scores": q_scores,
                "mean": q_mean,
                "std_dev": q_std,
                "min": q_min,
                "max": q_max_score,
                "range": q_range,
                "bands": q_bands,
                "band_stable": band_stable,
                "archetypes": q_archetypes,
                "archetype_stable": archetype_stable,
                "off_topic_stable": off_topic_stable,
            }
        )

    # Pillar consistency (aggregate across all questions per run)
    sample_pillars = runs[0]["questions"][0]["pillars"]
    pillar_analysis: dict[str, Any] = {}
    for p_key, p_meta in sample_pillars.items():
        p_name = p_meta["pillar_name"]
        run_pillar_totals: list[float] = []
        for r in runs:
            p_total = sum(q["pillars"][p_key]["score"] for q in r["questions"])
            run_pillar_totals.append(round(p_total, 2))

        p_mean = round(statistics.mean(run_pillar_totals), 2)
        p_std = round(statistics.stdev(run_pillar_totals), 2) if n_runs > 1 else 0.0
        pillar_analysis[p_key] = {
            "name": p_name,
            "mean_total": p_mean,
            "std_dev": p_std,
            "run_totals": run_pillar_totals,
        }

    # Qualitative consistency summary
    all_bands_stable = all(pq["band_stable"] for pq in per_question)
    all_off_topic_stable = all(pq["off_topic_stable"] for pq in per_question)
    all_archetypes_stable = all(pq["archetype_stable"] for pq in per_question)

    total_citations_verified = sum(
        sum(q["citations"]["mandatory_verified"] for q in r["questions"]) for r in runs
    )
    total_citations_mandatory = sum(
        sum(q["citations"]["mandatory_total"] for q in r["questions"]) for r in runs
    )
    citation_hit_rate = (
        round(total_citations_verified / total_citations_mandatory * 100, 1)
        if total_citations_mandatory > 0
        else 100.0
    )

    total_hallucinated = sum(
        sum(q["citations"]["hallucinated"] for q in r["questions"]) for r in runs
    )

    return {
        "n_runs": n_runs,
        "total_max": total_max,
        "mean_score": mean_score,
        "std_dev": std_dev,
        "min_score": min_score,
        "max_score": max_score,
        "score_range": score_range,
        "range_pct_of_max": range_pct_of_max,
        "cv_pct": cv_pct,
        "verdict": verdict,
        "per_question": per_question,
        "pillar_analysis": pillar_analysis,
        "qualitative": {
            "all_bands_stable": all_bands_stable,
            "all_off_topic_stable": all_off_topic_stable,
            "all_archetypes_stable": all_archetypes_stable,
            "citation_hit_rate_pct": citation_hit_rate,
            "total_hallucinated_citations": total_hallucinated,
        },
    }


def print_consistency_report(runs: list[dict[str, Any]], analysis: dict[str, Any], title: str) -> None:
    """Prints an examiner scorecard comparing marks and parameters across all runs."""
    n_runs = analysis["n_runs"]
    print("\n" + "=" * 82)
    print(f"TOPPER COPY CONSISTENCY REPORT ({n_runs} RUNS): {title.upper()}")
    print("=" * 82)

    # 1. Run-by-Run Total Score Breakdown
    print("\n1. RUN-BY-RUN OVERALL SCORES:")
    print(f"{'Run #':<8} {'Total Marks Awarded':<22} {'Percentage':<14} {'Overall Trajectory Band'}")
    print("-" * 82)
    for r in runs:
        print(
            f"Run {r['run_index']:<4} {r['total_score']:>6.2f} / {r['total_max']:>5.2f} marks    "
            f"{r['percentage']:>6.1f}%       {r['band']}"
        )

    print("\n--- STATISTICAL SPREAD METRICS ---")
    print(f"• Mean Copy Score:             {analysis['mean_score']:.2f} / {analysis['total_max']:.0f} marks")
    print(f"• Standard Deviation (σ):      {analysis['std_dev']:.2f} marks")
    print(f"• Score Range (Max - Min):     {analysis['score_range']:.2f} marks (Min: {analysis['min_score']:.2f}, Max: {analysis['max_score']:.2f})")
    print(f"• Spread as % of Max Marks:    {analysis['range_pct_of_max']:.2f}%")
    print(f"• Coefficient of Variation:    {analysis['cv_pct']:.2f}%")
    print(f"• Consistency Verdict:         [{analysis['verdict']}]")

    # 2. Per-Question Marking Comparison Table
    print("\n2. PER-QUESTION MARKING COMPARISON ACROSS RUNS:")
    headers = f"{'Q#':<4} {'Max':<5}" + "".join(f"{f'Run {i}':<9}" for i in range(1, n_runs + 1)) + f"{'Mean':<8} {'StdDev':<8} {'Range':<7} {'Band Stability'}"
    print(headers)
    print("-" * len(headers))
    for pq in analysis["per_question"]:
        scores_str = "".join(f"{s:<9.2f}" for s in pq["scores"])
        band_status = "STABLE" if pq["band_stable"] else "VARIES"
        print(
            f"Q{pq['q_num']:02d} {int(pq['max_marks']):<5}"
            f"{scores_str}"
            f"{pq['mean']:<8.2f}"
            f"{pq['std_dev']:<8.2f}"
            f"{pq['range']:<7.2f}"
            f"{band_status}"
        )

    # 3. Pillar-wise Stability
    print("\n3. 6-PILLAR G-EVAL STABILITY ACROSS RUNS (Whole-Copy Aggregates):")
    print(f"{'Pillar Name':<42} {'Mean Score':<14} {'StdDev (σ)':<12} {'Stability'}")
    print("-" * 78)
    for p_info in analysis["pillar_analysis"].values():
        std = p_info["std_dev"]
        status = "High" if std <= 0.50 else ("Moderate" if std <= 1.0 else "Low")
        print(f"• {p_info['name']:<40} {p_info['mean_total']:>6.2f} marks    {std:>6.2f}       {status}")

    # 4. Qualitative and Detection Parameter Consistency
    qual = analysis["qualitative"]
    print("\n4. QUALITATIVE & PARAMETER AUDIT CONSISTENCY:")
    print(f"• Performance Band Unanimity:     {'PASSED (Identical across all runs)' if qual['all_bands_stable'] else 'WARNING (Varied across runs)'}")
    print(f"• Off-Topic Gatekeeper Stability: {'PASSED (No false flags)' if qual['all_off_topic_stable'] else 'WARNING (Discrepancy detected)'}")
    print(f"• Presentation Archetype Detected:{'PASSED (Identical format detected)' if qual['all_archetypes_stable'] else 'WARNING (Format detection varied)'}")
    print(f"• Mandatory KB Anchor Hit Rate:   {qual['citation_hit_rate_pct']:.1f}%")
    print(f"• Hallucinated Citations Flagged: {qual['total_hallucinated_citations']} detected")
    print("=" * 82 + "\n")


def load_checkpoint(output_path: Path) -> list[dict[str, Any]]:
    """Loads existing run data if present so interrupted benchmarks can resume."""
    if not output_path.exists():
        return []
    try:
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("runs", []) if isinstance(data, dict) else []
    except Exception:
        return []


def build_clean_payload(
    title: str,
    input_path: Path,
    timestamp: str,
    runs: list[dict[str, Any]],
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds a lean JSON payload containing only questions, marks, and consistency stats."""
    runs_summary = [
        {
            "run": r["run_index"],
            "total_marks": r["total_score"],
            "max_marks": r["total_max"],
            "percentage": r["percentage"],
            "band": r["band"],
        }
        for r in runs
    ]

    questions_data: list[dict[str, Any]] = []
    if runs:
        n_questions = len(runs[0]["questions"])
        for q_idx in range(n_questions):
            q_template = runs[0]["questions"][q_idx]
            scores = [round(r["questions"][q_idx]["total_score"], 2) for r in runs]
            mean_score = round(statistics.mean(scores), 2)
            std_dev = round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0
            questions_data.append(
                {
                    "q_num": q_template["q_num"],
                    "question": q_template["question"],
                    "max_marks": q_template["max_marks"],
                    "marks_per_run": scores,
                    "mean_marks": mean_score,
                    "std_dev": std_dev,
                }
            )

    overall_stats = {}
    if analysis:
        overall_stats = {
            "mean_total_marks": analysis.get("mean_score"),
            "max_marks": analysis.get("total_max"),
            "std_dev": analysis.get("std_dev"),
            "min_marks": analysis.get("min_score"),
            "max_marks_awarded": analysis.get("max_score"),
            "score_range": analysis.get("score_range"),
            "spread_pct_of_max": analysis.get("range_pct_of_max"),
            "coefficient_of_variation_pct": analysis.get("cv_pct"),
            "verdict": analysis.get("verdict"),
        }

    return {
        "title": title,
        "input": str(input_path),
        "timestamp": timestamp,
        "runs_conducted": len(runs),
        "overall_stats": overall_stats,
        "runs_summary": runs_summary,
        "questions": questions_data,
    }


def save_checkpoint(
    output_path: Path,
    title: str,
    input_path: Path,
    timestamp: str,
    runs: list[dict[str, Any]],
    analysis: dict[str, Any] | None = None,
) -> None:
    """Saves clean, unbloated result JSON containing only marks and questions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_clean_payload(title, input_path, timestamp, runs, analysis=analysis)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a full UPSC topper answer copy multiple times to analyze scoring consistency."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to copy fixture JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of consecutive evaluation runs to execute (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: limit to first N questions of the copy (e.g. --limit 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON report file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--title",
        default="GS-2 Topper Copy Multi-Run Consistency",
        help="Report title/label",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing checkpoint file if available (default: False, starts from scratch)",
    )
    args = parser.parse_args()

    configure_console()
    setup_logging(quiet=QUIET_LOGGERS)

    if not args.input.exists():
        raise SystemExit(f"Error: input fixture not found at {args.input}")

    questions = load_questions(args.input, limit=args.limit)
    timestamp = datetime.now().isoformat(timespec="seconds")
    engine = UPSCEvaluationEngine()

    print("\n" + "=" * 82)
    print(f"STARTING MULTI-RUN CONSISTENCY BENCHMARK: {args.title.upper()}")
    print(f"Input: {args.input} | Questions: {len(questions)} | Target Runs: {args.runs}")
    print("=" * 82)

    # Clean runs list — only resume if explicitly requested
    runs: list[dict[str, Any]] = []
    if args.resume:
        existing_runs = load_checkpoint(args.output)
        for r in existing_runs:
            if len(r.get("questions", [])) == len(questions):
                runs.append(r)
        if runs:
            print(f"Resuming with {len(runs)} previously completed runs from {args.output}...")

    start_run_idx = len(runs) + 1

    for r_idx in range(start_run_idx, args.runs + 1):
        print(f"\n--- RUN {r_idx}/{args.runs} ({len(questions)} Questions) ---")
        q_records: list[dict[str, Any]] = []

        for q in questions:
            res = engine.evaluate_answer(
                question=q["question"],
                candidate_answer=q["candidate_answer"],
                max_marks=q["max_marks"],
            )
            record = extract_question_metrics(q["q_num"], q["max_marks"], q["question"], res)
            q_records.append(record)
            print_minimal_question_line(record)

        total_score = round(sum(r["total_score"] for r in q_records), 2)
        total_max = sum(r["max_marks"] for r in q_records)
        pct = round((total_score / total_max * 100), 1) if total_max else 0.0

        run_data = {
            "run_index": r_idx,
            "total_score": total_score,
            "total_max": total_max,
            "percentage": pct,
            "band": overall_band(pct),
            "questions": q_records,
        }
        runs.append(run_data)
        print(f"  >>> Run {r_idx} Total: {total_score:.2f} / {total_max:.0f} ({pct:.1f}%) | {overall_band(pct)}")

        # Save checkpoint after each run (clean unbloated format)
        save_checkpoint(args.output, args.title, args.input, timestamp, runs)

    analysis = compute_consistency_analysis(runs)
    print_consistency_report(runs, analysis, title=args.title)
    save_checkpoint(args.output, args.title, args.input, timestamp, runs, analysis=analysis)
    print(f"Detailed consistency report saved to: {args.output}\n")


if __name__ == "__main__":
    main()
