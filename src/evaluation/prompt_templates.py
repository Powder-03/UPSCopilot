"""UPSC Mains evaluation prompts, rubrics, and calibrated scoring templates."""
from typing import List

SYSTEM_PROMPT_UPSC_EXAMINER = """You are a Senior UPSC Civil Services Examination (CSE) Mains Evaluator with decades of experience evaluating General Studies Paper II (Governance, Constitution, Polity, Social Justice).

You operate under the strict, authentic marking calibration of the Union Public Service Commission:
- POOR ANSWER (<35%): 1.0 - 3.0 / 10 (Fails core demands, vague generalities, missing articles/cases).
- AVERAGE ANSWER (35-45%): 3.5 - 4.5 / 10 (Answers part of question, basic legal recall, unstructured paragraphs).
- GOOD ANSWER (46-55%): 4.5 - 5.5 / 10 (Addresses all sub-demands, cites key cases and articles, clear subheadings).
- TOPPER BENCHMARK (56-65%+): 6.0 - 7.0 / 10 (Multi-dimensional breadth, precise judicial ratios, diagrams/flowcharts, balanced forward-looking conclusion).
- NOTE: UPSC examiners NEVER award 8.0+ or 9.0+ out of 10. Grade inflation is strictly prohibited.

EVALUATION PRINCIPLES:
1. HARD DEMAND RELEVANCE GATEKEEPER:
   If an answer is factually accurate and well-written but addresses a completely different topic than what was asked (e.g. writing about Emergency Art 356 when asked about Ordinances Art 123), you MUST flag is_off_topic=true, set demand_relevance_gate=0.10, and penalize total marks to <= 1.0 / 10. Never award marks for unrequested essays.

2. DUAL-GROUNDING (KB FLOOR + OPEN-WORLD CEILING):
   - Mandatory Core Legal Provisions: Check whether mandatory constitutional articles, landmark cases, and acts required by the question prompt are present and accurately cited (verified against the provided Knowledge Base context).
   - Valid Open-World Insights: If the candidate cites valid external committees (e.g. 2nd ARC, Punchhi, Sarkaria), international comparative examples, or interdisciplinary angles NOT present in the retrieved KB context, verify their factual truth and logical relevance. If genuine, award FULL CREDIT and mark them as open_world_credited. Never penalize a valid point simply because it wasn't in the local database.

3. PRESENTATION ARCHETYPE:
   - If PARAGRAPH_HEAVY: Do not penalize legal content, but critique visual fatigue and explain how converting prose into subheadings/bullet points gains marks in a 7-minute exam setting.
   - If HYBRID_DIAGRAMMATIC or TABULAR: Evaluate whether the diagram/table adds genuine conceptual density or is just empty decoration.

4. 5 CANONICAL PILLARS (Weights for 10-markers / 15-markers):
   - Pillar 1: Directive & Demand Fulfillment (25% weight)
   - Pillar 2: Structural Architecture & Presentation (15% weight)
   - Pillar 3: Multi-Dimensional Breadth (PESTLE / GS-2 Angles) (20% weight)
   - Pillar 4: Grounded Legal Citations & Accuracy (25% weight)
   - Pillar 5: Conclusion & Constructive Way Forward (15% weight)
"""

def build_cot_diagnostic_prompt(
    question: str,
    candidate_answer: str,
    kb_context: List[str],
    max_marks: float = 10.0,
) -> str:
    """Builds the deep Chain-of-Thought diagnostic prompt for Call 1."""
    context_str = "\n\n---\n\n".join(kb_context) if kb_context else "No specific KB context retrieved."
    
    return f"""Evaluate the following UPSC Mains candidate answer against the authentic question and ground truth knowledge base.

QUESTION ({int(max_marks)} Marks):
{question}

CANDIDATE ANSWER:
{candidate_answer}

AUTHENTIC KNOWLEDGE BASE CONTEXT (For Grounding Verification):
{context_str}

Perform a rigorous evaluation and output ONLY a valid JSON object matching this schema:
{{
  "cot_reasoning_trail": "Step-by-step reasoning trail: 1. Demand decomposition; 2. KB grounding comparison; 3. Open-world insights verification; 4. Presentation & visual layout audit; 5. Off-topic check; 6. Deduction & justification.",
  "is_off_topic": false,
  "demand_relevance_gate": 1.0,
  "directive_detected": "discuss / critically_analyze / etc",
  "micro_demands": [
    {{
      "demand": "Description of sub-demand 1",
      "status": "fully_addressed / partially_addressed / omitted / off_topic",
      "marks_allocated": 3.0,
      "marks_obtained": 2.0,
      "comment": "Specific reason for score"
    }}
  ],
  "citation_audit": {{
    "mandatory_kb_anchors": [
      {{
        "name": "Article 123",
        "status": "mandatory_found / mandatory_missing / hallucinated_or_wrong",
        "source": "knowledge_base",
        "notes": "Properly cited President ordinance power"
      }}
    ],
    "open_world_credits": [
      {{
        "name": "Second ARC 4th Report",
        "status": "open_world_credited",
        "source": "open_world",
        "notes": "Valid external governance recommendation appropriately applied"
      }}
    ],
    "hallucinated_citations": [],
    "summary": "Brief summary of factual and statutory grounding."
  }},
  "presentation": {{
    "detected_archetype": "paragraph_heavy / bullet_structured / hybrid_diagrammatic / tabular",
    "visual_density_score": 6.5,
    "diagrams_and_tables_found": ["Timeline of 6 weeks ordinance lapse"],
    "presentation_bonus": 0.5,
    "examiner_critique": "Assessment of visual ergonomics and readability under exam conditions.",
    "topper_reformatting_tip": "Concrete restructuring tip: e.g. how to convert dense paragraphs into structured subheadings with bullet points."
  }},
  "strengths": [
    "Key strength 1",
    "Key strength 2"
  ],
  "weaknesses": [
    "Key deficiency 1",
    "Key deficiency 2"
  ],
  "topper_action_plan": [
    "Actionable upgrade 1 to gain +1.0 mark",
    "Actionable upgrade 2"
  ],
  "pillar_summary": {{
    "demand_fulfillment": "Rating summary for Pillar 1 (1 to 5 level rationale)",
    "structure_presentation": "Rating summary for Pillar 2 (1 to 5 level rationale)",
    "multidimensional_breadth": "Rating summary for Pillar 3 (1 to 5 level rationale)",
    "grounded_citations": "Rating summary for Pillar 4 (1 to 5 level rationale)",
    "conclusion_way_forward": "Rating summary for Pillar 5 (1 to 5 level rationale)"
  }}
}}
"""

def build_geval_scoring_prompt(
    question: str,
    candidate_answer: str,
    cot_reasoning_trail: str,
    pillar_summary: dict,
) -> str:
    """Builds the single-call 5-pillar G-Eval scoring prompt for Call 2."""
    p1_summary = pillar_summary.get("demand_fulfillment", "")
    p2_summary = pillar_summary.get("structure_presentation", "")
    p3_summary = pillar_summary.get("multidimensional_breadth", "")
    p4_summary = pillar_summary.get("grounded_citations", "")
    p5_summary = pillar_summary.get("conclusion_way_forward", "")

    return f"""Based on your detailed UPSC examiner Chain-of-Thought reasoning below, assign a rating from 1 to 5 for each of the 5 canonical UPSC pillars.

RATING SCALE (1 = Poor, 2 = Below Average, 3 = Average, 4 = Good, 5 = Topper Benchmark):
1: Poor (<35% benchmark, severe omissions)
2: Below Average (35-45%, partial coverage, major gaps)
3: Average (46-55%, standard textbook answer, basic structure)
4: Good (56-65%, all demands met, accurate citations, clear headings)
5: Topper Benchmark (66%+, exemplary depth, multi-dimensional, diagram/table, innovative way-forward)

QUESTION:
{question}

CANDIDATE ANSWER:
{candidate_answer}

EXAMINER DIAGNOSTIC SUMMARY:
- Pillar 1 (Demand Fulfillment): {p1_summary}
- Pillar 2 (Structure & Presentation): {p2_summary}
- Pillar 3 (Multi-Dimensional Breadth): {p3_summary}
- Pillar 4 (Grounded Citations): {p4_summary}
- Pillar 5 (Conclusion & Way Forward): {p5_summary}

Output ONLY the ratings in this exact format with single-digit integers (1, 2, 3, 4, or 5) immediately after the colon:
P1: 4
P2: 3
P3: 4
P4: 5
P5: 3"""
