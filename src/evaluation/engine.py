"""Core UPSC Mains Answer Evaluation Engine.
Orchestrates:
1. Hybrid KB Retrieval (Amazon Titan Embeddings + BM25 + FlashRank RRF)
2. Call 1: Deep Chain-of-Thought (CoT) Diagnostic Analysis (Moonshot Kimi 2.5)
3. Call 2: G-Eval Logprob Continuous Expected Value Scoring (sum(s * P(s)))
4. Hard Demand Relevance Gatekeeper against off-topic answers
5. Dual Grounding (Mandatory KB provisions vs Valid Open-World insights)
6. Presentation Archetype analysis (Paragraphs vs Diagrams)
"""
import re
import json
import logging
from typing import Optional, Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.kb.retriever import HybridRetriever
from src.evaluation.model_factory import get_eval_llm
from src.evaluation.geval_scorer import BedrockGEvalScorer
from src.evaluation.prompt_templates import (
    SYSTEM_PROMPT_UPSC_EXAMINER,
    build_cot_diagnostic_prompt,
)
from src.models.enums import (
    UPSCPerformanceBand,
    PresentationArchetype,
    DemandStatus,
    CitationStatus,
    DirectiveType,
)
from src.models.schema import (
    EvaluationResult,
    MicroDemandItem,
    CitationItem,
    CitationAudit,
    PresentationEvaluation,
    PillarGEvalScore,
)

logger = logging.getLogger(__name__)


def _extract_json_dict(text: str) -> Dict[str, Any]:
    """Robustly extracts the primary JSON object from LLM response text."""
    if not text:
        return {}

    # Check for markdown code fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    else:
        start = text.find("{")
        if start == -1:
            return {}
        candidate = text[start:]

    try:
        obj, _ = json.JSONDecoder().raw_decode(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        logger.warning(f"Error parsing JSON from LLM output: {e}")

    return {}


class UPSCEvaluationEngine:
    """Enterprise-grade, calibrated UPSC Civil Services Mains Answer Evaluation Engine."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        eval_llm=None,
        geval_scorer: Optional[BedrockGEvalScorer] = None,
    ):
        self.retriever = retriever or HybridRetriever(top_k=6)
        self.llm = eval_llm or get_eval_llm()
        self.geval_scorer = geval_scorer or BedrockGEvalScorer()

    def evaluate_answer(
        self,
        question: str,
        candidate_answer: str,
        max_marks: float = 10.0,
    ) -> EvaluationResult:
        """Evaluates a candidate answer using 2-call CoT + G-Eval probability trailing."""
        logger.info(f"Starting UPSC evaluation for question ({int(max_marks)} marks)...")

        # Step 1: Ground Truth Retrieval from authentic Knowledge Base
        kb_context: List[str] = self.retriever.get_retrieval_context(question, top_k=6)
        logger.info(f"Retrieved {len(kb_context)} ground-truth context blocks from Knowledge Base.")

        # Step 2: Call 1 — CoT Diagnostic Analysis & Qualitative Audit
        cot_prompt = build_cot_diagnostic_prompt(
            question=question,
            candidate_answer=candidate_answer,
            kb_context=kb_context,
            max_marks=max_marks,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT_UPSC_EXAMINER),
            HumanMessage(content=cot_prompt),
        ]

        logger.info("Executing Call 1: Qualitative Diagnostic & CoT reasoning...")
        res = self.llm.invoke(messages)
        diagnostic = _extract_json_dict(res.content)

        # Parse Call 1 diagnostic fields with safe defaults
        cot_trail = diagnostic.get("cot_reasoning_trail", "Chain of thought generated.")
        is_off_topic = bool(diagnostic.get("is_off_topic", False))
        demand_relevance_gate = float(diagnostic.get("demand_relevance_gate", 1.0 if not is_off_topic else 0.1))
        
        # Directive detection
        raw_directive = str(diagnostic.get("directive_detected", "")).lower()
        directive = None
        for d in DirectiveType:
            if d.value in raw_directive:
                directive = d
                break

        # Micro-demands
        micro_demands: List[MicroDemandItem] = []
        for d_raw in diagnostic.get("micro_demands", []):
            try:
                status_val = str(d_raw.get("status", "partially_addressed")).lower()
                status = DemandStatus.PARTIALLY_ADDRESSED
                for s in DemandStatus:
                    if s.value in status_val:
                        status = s
                        break

                micro_demands.append(
                    MicroDemandItem(
                        demand=d_raw.get("demand", "Sub-demand"),
                        status=status,
                        marks_allocated=float(d_raw.get("marks_allocated", 2.5)),
                        marks_obtained=float(d_raw.get("marks_obtained", 1.0)),
                        comment=d_raw.get("comment", ""),
                    )
                )
            except Exception as e:
                logger.warning(f"Error parsing micro-demand: {e}")

        # Citation Audit
        raw_citations = diagnostic.get("citation_audit", {})
        mandatory_anchors = [
            CitationItem(
                name=c.get("name", ""),
                status=CitationStatus.MANDATORY_FOUND if "found" in str(c.get("status", "")).lower() else CitationStatus.MANDATORY_MISSING,
                source="knowledge_base",
                notes=c.get("notes", ""),
            )
            for c in raw_citations.get("mandatory_kb_anchors", [])
        ]
        open_world_credits = [
            CitationItem(
                name=c.get("name", ""),
                status=CitationStatus.OPEN_WORLD_CREDITED,
                source="open_world",
                notes=c.get("notes", ""),
            )
            for c in raw_citations.get("open_world_credits", [])
        ]
        hallucinated = [
            CitationItem(
                name=c.get("name", ""),
                status=CitationStatus.HALLUCINATED_OR_WRONG,
                source="knowledge_base",
                notes=c.get("notes", ""),
            )
            for c in raw_citations.get("hallucinated_citations", [])
        ]
        citation_audit = CitationAudit(
            mandatory_kb_anchors=mandatory_anchors,
            open_world_credits=open_world_credits,
            hallucinated_citations=hallucinated,
            summary=raw_citations.get("summary", "Factual and statutory grounding evaluated."),
        )

        # Presentation Evaluation
        raw_pres = diagnostic.get("presentation", {})
        arch_val = str(raw_pres.get("detected_archetype", "paragraph_heavy")).lower()
        archetype = PresentationArchetype.PARAGRAPH_HEAVY
        for a in PresentationArchetype:
            if a.value in arch_val:
                archetype = a
                break

        presentation = PresentationEvaluation(
            detected_archetype=archetype,
            visual_density_score=float(raw_pres.get("visual_density_score", 5.0)),
            diagrams_and_tables_found=raw_pres.get("diagrams_and_tables_found", []),
            presentation_bonus=float(raw_pres.get("presentation_bonus", 0.0)),
            examiner_critique=raw_pres.get("examiner_critique", "Adequate presentation format."),
            topper_reformatting_tip=raw_pres.get(
                "topper_reformatting_tip",
                "Structure key points under explicit subheadings with numbered bullet points.",
            ),
        )

        strengths = diagnostic.get("strengths", ["Addressed core themes of the question."])
        weaknesses = diagnostic.get("weaknesses", ["Expand multi-dimensional governance angles."])
        topper_action_plan = diagnostic.get("topper_action_plan", ["Incorporate precise constitutional articles and landmark case ratios."])
        pillar_summary = diagnostic.get("pillar_summary", {})

        # Step 3: Call 2 — G-Eval Multi-Pillar Logprob Scoring Head
        logger.info("Executing Call 2: G-Eval probabilistic logprob scoring head...")
        pillars = self.geval_scorer.score_pillars(
            question=question,
            candidate_answer=candidate_answer,
            cot_reasoning_trail=cot_trail,
            pillar_summary=pillar_summary,
            max_marks=max_marks,
        )

        # Step 4: Calibrate Total Score and Apply Relevance Gating
        raw_total_score = sum(p.calibrated_score for p in pillars.values())
        raw_total_score += presentation.presentation_bonus

        if is_off_topic:
            logger.warning("Hard Demand Relevance Gate triggered: Off-topic answer detected!")
            total_score = round(min(1.0, raw_total_score * demand_relevance_gate), 2)
            band = UPSCPerformanceBand.NEEDS_FOUNDATION
        else:
            # Enforce authentic UPSC limits: top ceiling is ~65-70% max
            topper_ceiling = max_marks * 0.70
            min_floor = 0.5
            total_score = round(min(topper_ceiling, max(min_floor, raw_total_score)), 2)
            
            pct = (total_score / max_marks) * 100.0
            if pct < 35.0:
                band = UPSCPerformanceBand.NEEDS_FOUNDATION
            elif pct < 45.0:
                band = UPSCPerformanceBand.AVERAGE
            elif pct < 56.0:
                band = UPSCPerformanceBand.GOOD
            else:
                band = UPSCPerformanceBand.TOPPER

        pct = round((total_score / max_marks) * 100.0, 1)

        result = EvaluationResult(
            question=question,
            candidate_answer=candidate_answer,
            max_marks=max_marks,
            total_score=total_score,
            percentage=pct,
            performance_band=band,
            is_off_topic=is_off_topic,
            demand_relevance_gate=demand_relevance_gate,
            directive_detected=directive,
            cot_reasoning_trail=cot_trail,
            micro_demands=micro_demands,
            pillars=pillars,
            presentation=presentation,
            citation_audit=citation_audit,
            strengths=strengths,
            weaknesses=weaknesses,
            topper_action_plan=topper_action_plan,
        )

        logger.info(f"Evaluation completed: Score = {total_score} / {max_marks} ({pct}%) | Band = {band.value}")
        return result
