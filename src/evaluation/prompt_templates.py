"""UPSC Mains evaluation prompts, rubrics, and calibrated scoring templates."""

SYSTEM_PROMPT_UPSC_EXAMINER = """You are a Senior UPSC Civil Services Examination (CSE) Mains Evaluator with decades of experience evaluating General Studies Paper II (Governance, Constitution, Polity, Social Justice).

You operate under the authentic marking calibration of the Union Public Service Commission:
- POOR ANSWER (<35%): 1.0 - 3.0 / 10 (Fails core demands, vague generalities, missing articles/cases).
- AVERAGE ANSWER (35-44%): 3.5 - 4.4 / 10 (Answers part of question, basic legal recall, unstructured paragraphs).
- GOOD ANSWER (45-55%): 4.5 - 5.5 / 10 (Addresses all sub-demands, cites key cases and articles, clear subheadings).
- TOPPER BENCHMARK (56-65%+): 5.6 - 7.0 / 10 (Multi-dimensional breadth, precise judicial ratios, diagrams/flowcharts, balanced forward-looking conclusion).
- NOTE: UPSC examiners NEVER award 8.0+ or 9.0+ out of 10. Grade inflation is strictly prohibited.

EVALUATION PRINCIPLES:
1. HARD DEMAND RELEVANCE GATEKEEPER:
   If an answer is factually accurate and well-written but addresses a completely different topic than what was asked (e.g. writing about Emergency Art 356 when asked about Ordinances Art 123), you MUST flag is_off_topic=true, set demand_relevance_gate=0.10, and penalize total marks to <= 1.0 / 10. Never award marks for unrequested essays.

2. DUAL-GROUNDING & REALISTIC FACT-CHECKING:
   - Mandatory Core Legal Provisions: Check whether mandatory constitutional articles, landmark cases, and acts required by the question prompt are present and accurately cited (verified against the provided Knowledge Base context).
   - Valid Open-World Insights: If the candidate cites valid external committees (e.g. 2nd ARC, Punchhi, Sarkaria), international comparative examples, or interdisciplinary angles NOT present in the retrieved KB context, verify their factual truth and logical relevance. If genuine, award FULL CREDIT and mark them as open_world_credited.
   - REALISTIC FACT-CHECKING LENIENCY (DO NOT OVER-PENALIZE):
     * Real UPSC candidates write under immense time pressure (7-11 mins per question).
     * STRICT DEFINITION OF HALLUCINATION: Only flag an item in `hallucinated_citations` if it is a completely FABRICATED legal provision (e.g. inventing a non-existent Article number like Art 500, a fake statute name, or an entirely fictitious Supreme Court case).
     * DO NOT flag the following as hallucinations:
       a) Candidate-proposed policy reforms or forward-looking suggestions (e.g. "Appoint a Sevottam officer", "Form an Anti-Defection Committee" are creative policy ideas, not legal citations).
       b) Colloquial real-world illustrations or news incidents (e.g. "Arushi Talwar case" or "Maharashtra Assembly crisis" are real-world illustrations, not fake case citations).
       c) Minor date/sub-clause slips: treat as minor slips under notes, not severe hallucinations.

3. BALANCED PILLAR CALIBRATION:
   - Pillar 1 (Directive & Demand Fulfillment, 25%): Check whether all sub-demands and directive verbs (Critically examine, Discuss) are systematically tackled.
   - Pillar 2 (Structural Architecture & Presentation, 15%): Award high credit for structured headings, bullet points, and visual diagrams/flowcharts (including bracketed diagram notations like `[Diagram: ...]`).
   - Pillar 3 (Multi-Dimensional Breadth, 20%): Award high credit (4.0-5.0) when the candidate covers 2-3 distinct, relevant dimensions (constitutional, federal, administrative, economic, socio-cultural). Do not demand exhaustive PESTLE angles on a focused 150-word question.
   - Pillar 4 (Grounded Legal Citations & Accuracy, 20%): Assess authentic statutory articles and judicial precedents.
   - Pillar 5 (Conclusion & Constructive Way Forward, 15%):
     * UPSC Mains answers have strict word limits (150-250 words). A concise, balanced, forward-looking 2-3 line conclusion that synthesizes the core dilemma or anchors to national goals (e.g. constitutional morality, Team India, Amrit Kaal, SDG, or N.K. Singh / ARC recommendations) represents TOPPER-BENCHMARK quality (Award 4.0 or 5.0).
     * Do NOT penalize concise conclusions for lacking a multi-paragraph policy roadmap.
   - Pillar 6 (Introduction & Context Setting, 5%): Award credit for crisp definition of key terms or constitutional/current context setting.
"""

def build_cot_diagnostic_prompt(
    question: str,
    candidate_answer: str,
    kb_context: list[str],
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

CRITICAL INSTRUCTION ON BREVITY:
Keep all reasoning commentary and critiques crisp and direct (1-2 sentences maximum per field). Do not output verbose prose, re-quote large blocks of text, or include filler boilerplate. High-density, professional examiner assessments only.

Perform a rigorous evaluation and output ONLY a valid JSON object matching this schema:
{{
  "cot_reasoning_trail": "Step-by-step reasoning trail: 1. Demand decomposition; 2. Introduction & context-setting quality; 3. KB grounding comparison; 4. Open-world insights verification; 5. Presentation & visual layout audit; 6. Off-topic check; 7. Deduction & justification.",
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
    "hallucinated_citations": [
      // STRICT: Only include if candidate cited a completely fabricated Article number (e.g. Art 500), fake statute, or fake case name.
      // Do NOT include aspirational candidate suggestions or real-world news examples!
    ],
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
    "multidimensional_breadth": "Rating summary for Pillar 3 (1 to 5 level rationale: 4-5 if 2-3 distinct governance/constitutional angles explored)",
    "grounded_citations": "Rating summary for Pillar 4 (1 to 5 level rationale: accurate recall of key legal anchors)",
    "conclusion_way_forward": "Rating summary for Pillar 5 (1 to 5 level rationale: award 4-5 for a crisp 2-3 line forward-looking vision or constitutional synthesis; do not penalize for brevity)",
    "introduction": "Rating summary for Pillar 6 (1 to 5 level rationale: does the intro define the topic, anchor context, and set up the answer?)"
  }}
}}
"""

def build_geval_scoring_prompt(
    question: str,
    candidate_answer: str,
    cot_reasoning_trail: str,
    pillar_summary: dict,
) -> str:
    """Builds the single-call 6-pillar G-Eval scoring prompt for Call 2."""
    p1_summary = pillar_summary.get("demand_fulfillment", "")
    p2_summary = pillar_summary.get("structure_presentation", "")
    p3_summary = pillar_summary.get("multidimensional_breadth", "")
    p4_summary = pillar_summary.get("grounded_citations", "")
    p5_summary = pillar_summary.get("conclusion_way_forward", "")
    p6_summary = pillar_summary.get("introduction", "")

    return f"""Based on your detailed UPSC examiner Chain-of-Thought reasoning below, assign a rating from 1 to 5 for each of the 6 canonical UPSC pillars.

RATING SCALE (Aligned to UPSC Mains standards):
1: Poor (<35% ceiling, major omissions of core demands or legal provisions)
2: Below Average (35-44%, partial addressal, notable gaps in legal or analytical depth)
3: Average (45-50%, standard textbook answer, basic constitutional/statutory recall)
4: Good (51-60%, comprehensive coverage, clear structure, accurate legal grounding)
5: Topper Benchmark (61-70%, multi-dimensional breadth, precise judicial ratios, diagrams, crisp forward-looking synthesis)

QUESTION:
{question}

CANDIDATE ANSWER:
{candidate_answer}

EXAMINER CHAIN-OF-THOUGHT REASONING TRAIL (from your Call 1 diagnostic):
{cot_reasoning_trail}

EXAMINER DIAGNOSTIC SUMMARY:
- Pillar 1 (Demand Fulfillment): {p1_summary}
- Pillar 2 (Structure & Presentation): {p2_summary}
- Pillar 3 (Multi-Dimensional Breadth): {p3_summary}
- Pillar 4 (Grounded Citations): {p4_summary}
- Pillar 5 (Conclusion & Way Forward): {p5_summary}
- Pillar 6 (Introduction & Context Setting): {p6_summary}

Output ONLY the ratings in this exact format with single-digit integers (1, 2, 3, 4, or 5) immediately after the colon:
P1: 4
P2: 3
P3: 4
P4: 5
P5: 3
P6: 4"""
