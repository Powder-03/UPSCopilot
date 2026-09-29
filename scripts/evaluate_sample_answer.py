"""CLI script to test and demonstrate the calibrated UPSC Mains Answer Evaluation Engine."""
import sys
import json
import logging

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("langchain_aws").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)

from src.evaluation.engine import UPSCEvaluationEngine
from src.models.schema import EvaluationResult

SAMPLE_QUESTION = (
    "Examine the constitutional safeguards against executive abuse of the ordinance-making power. "
    "How did the Supreme Court curb mechanical re-promulgation in D.C. Wadhwa and Krishna Kumar Singh?"
)

# 1. TOPPER BENCHMARK ANSWER (Structured, diagrams, accurate case laws & articles)
TOPPER_ANSWER = """
**Introduction:**
Ordinance-making power under **Article 123** (President) and **Article 213** (Governor) is an extraordinary legislative power designed to address urgent, unforeseen contingencies when Parliament/State Assembly is not in session. It is an emergency power, not a parallel source of legislation.

**Constitutional Safeguards Against Executive Abuse:**
1. **Recess Prerequisite**: Can only be promulgated when either House of Parliament/State Legislature is not in session.
2. **Immediate Action Necessity**: Objective satisfaction of the executive that circumstances exist requiring immediate action.
3. **Mandatory Tabling & Temporal Limitation**:
   - Must be laid before the legislature upon reassembly.
   - Ceases to operate **6 weeks** from the date of reassembly unless approved earlier.
   
```
[Executive Promulgation] -> [Legislative Reassembly] -> [Lapses in 6 Weeks without Approval]
```

**Judicial Curb on Mechanical Re-promulgation:**
- **D.C. Wadhwa v. State of Bihar (1987)**:
  - A 5-judge Constitution Bench struck down Bihar's practice of keeping 256 ordinances alive for up to 14 years without tabling them.
  - Held: Repeated re-promulgation is a **subversion of the democratic legislative process**, a **colorable exercise of power**, and a **"fraud on the Constitution"**.
- **Krishna Kumar Singh v. State of Bihar (2017)**:
  - A 7-judge Constitution Bench reaffirmed *Wadhwa* and held that:
    1. Tabling the ordinance before the legislature is a **mandatory constitutional obligation**.
    2. Executive satisfaction is open to **judicial review** to check for malafide or extraneous grounds.
    3. No rights or liabilities can permanently survive an ordinance that was never placed before the legislature.

**Conclusion:**
Ordinances must remain a constitutional safety valve rather than a governance norm. As held in *Krishna Kumar Singh*, preserving legislative supremacy and constitutional morality is indispensable to the rule of law.
"""

# 2. AVERAGE ANSWER (Dense paragraphs, missing Article 213, missing Krishna Kumar Singh)
AVERAGE_ANSWER = """
Ordinances are laws made by the government when the parliament is not in session. The President can issue ordinances under Article 123 of the Constitution of India. However, the government sometimes abuses this power to bypass the parliament and make laws directly without debate. The Constitution says that an ordinance must be approved by the Parliament within six weeks of its meeting, otherwise it will lapse and become invalid.

In the D.C. Wadhwa case, the Supreme Court of India said that the government cannot keep re-promulgating ordinances again and again without passing them in the assembly. The Bihar government had passed hundreds of ordinances without taking them to the legislature, which was wrong. The court said this is an abuse of power and against democracy. 

Therefore, the executive should be careful and not misuse ordinances frequently. The parliament should discuss all important bills to ensure good governance in the country.
"""

# 3. OFF-TOPIC ANSWER (Accurate essay on Article 356 Emergency, but wrong topic!)
OFF_TOPIC_ANSWER = """
Under Article 356 of the Constitution, President's Rule can be imposed in a State if the Governor reports or the President is satisfied that governance cannot be carried on in accordance with the Constitution. This power was historically misused by the Union executive to dismiss opposition-ruled state governments on partisan grounds.

In the landmark judgment of S.R. Bommai v. Union of India (1994), a 9-judge Constitution Bench held that federalism is part of the basic structure of the Constitution. The Court laid down that presidential proclamations under Article 356 are subject to judicial review. The Court mandated the Floor Test as the sole constitutional forum to determine a government's majority and held that the Assembly cannot be dissolved until Parliament approves the proclamation. This landmark ruling effectively curbed the partisan abuse of emergency powers in India.
"""


def print_evaluation_card(title: str, res: EvaluationResult):
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {title.upper()}")
    print("=" * 80)
    print(f"Question: {res.question}")
    print(f"\nFinal Marks Awarded: {res.total_score:.2f} / {res.max_marks} ({res.percentage}%)")
    print(f"Performance Band:    {res.performance_band.value}")
    print(f"Off-Topic Status:    {'[!] OFF-TOPIC ALERT' if res.is_off_topic else 'ON-TOPIC'}")
    print(f"Relevance Gate:      {res.demand_relevance_gate:.2f}")

    print("\n--- 5-PILLAR G-EVAL CONTINUOUS SCORECARD ---")
    for key, p in res.pillars.items():
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


def run_demo():
    engine = UPSCEvaluationEngine()

    print("Evaluating Answer 1: Topper Benchmark Answer...")
    topper_res = engine.evaluate_answer(SAMPLE_QUESTION, TOPPER_ANSWER, max_marks=10.0)
    print_evaluation_card("Candidate 1: Topper Benchmark Answer", topper_res)

    print("Evaluating Answer 2: Average Paragraph Answer...")
    avg_res = engine.evaluate_answer(SAMPLE_QUESTION, AVERAGE_ANSWER, max_marks=10.0)
    print_evaluation_card("Candidate 2: Average Paragraph Answer", avg_res)

    print("Evaluating Answer 3: Off-Topic Answer (Emergency written for Ordinance)...")
    off_topic_res = engine.evaluate_answer(SAMPLE_QUESTION, OFF_TOPIC_ANSWER, max_marks=10.0)
    print_evaluation_card("Candidate 3: Off-Topic Answer", off_topic_res)


if __name__ == "__main__":
    run_demo()
