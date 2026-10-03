"""Core UPSC Mains Answer Evaluation Engine.
Orchestrates:
1. Hybrid KB Retrieval (Amazon Titan Embeddings + BM25 + FlashRank RRF)
2. Call 1: Deep Chain-of-Thought (CoT) Diagnostic Analysis (Moonshot Kimi 2.5)
3. Call 2: G-Eval Logprob Continuous Expected Value Scoring (sum(s * P(s)))
4. Hard Demand Relevance Gatekeeper against off-topic answers
5. Dual Grounding (Mandatory KB provisions vs Valid Open-World insights)
6. Presentation Archetype analysis (Paragraphs vs Diagrams)
"""
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.evaluation.geval_scorer import (
    PILLAR_CONFIGS,
    BaseGEvalScorer,
    get_geval_scorer,
)
from src.evaluation.prompt_templates import (
    SYSTEM_PROMPT_UPSC_EXAMINER,
    build_cot_diagnostic_prompt,
)
from src.kb.retriever import HybridRetriever
from src.models.enums import (
    CitationStatus,
    DemandStatus,
    DirectiveType,
    PresentationArchetype,
    UPSCPerformanceBand,
)
from src.models.evaluation import (
    CitationAudit,
    CitationItem,
    EvaluationResult,
    MicroDemandItem,
    PillarGEvalScore,
    PresentationEvaluation,
)
from src.models.exceptions import ModelInvocationError
from src.providers.factory import get_eval_llm
from src.utils.json import extract_json_dict
from src.utils.tracing import traceable

logger = logging.getLogger(__name__)

# --- Calibration constants: authentic UPSC marking policy ---
TOPPER_CEILING_PCT = 0.56       # Authentic national topper ceiling (~56%); grade inflation strictly forbidden
MIN_SCORE_FLOOR = 0.5           # An on-topic attempt is never awarded a bare zero
OFF_TOPIC_GATE_DEFAULT = 0.10   # Fallback relevance gate, matching the policy stated in the system prompt
OFF_TOPIC_MAX_MARKS = 1.0       # Hard cap for an answer that addresses a different question
MAX_PRESENTATION_BONUS = 0.5    # Capped tie-breaker; Pillar 2 already scores structure


def _match_enum(enum_cls, raw: Any, default):
    """Matches a raw LLM string to an enum member: exact match first, then longest substring.

    Normalizes spaces/hyphens to underscores so natural phrasings like "to what extent"
    resolve correctly, and prefers the most specific value ("critically_analyze" over "discuss").
    """
    val = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    for member in enum_cls:
        if member.value == val:
            return member
    for member in sorted(enum_cls, key=lambda m: len(m.value), reverse=True):
        if member.value in val:
            return member
    return default


def _parse_citation_status(raw: Any) -> CitationStatus:
    """Parses a mandatory-anchor citation status, checking negative statuses first.

    Prevents values like "not_found"/"not found" from matching the substring "found"
    and being misclassified as MANDATORY_FOUND.
    """
    val = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if "hallucinat" in val or "wrong" in val:
        return CitationStatus.HALLUCINATED_OR_WRONG
    if "missing" in val or "not_found" in val or "notfound" in val or val.startswith("not"):
        return CitationStatus.MANDATORY_MISSING
    if "found" in val or "present" in val or "cited" in val:
        return CitationStatus.MANDATORY_FOUND
    return CitationStatus.MANDATORY_MISSING


def _classify_performance_band(pct: float) -> UPSCPerformanceBand:
    """Maps a percentage to the authentic UPSC band (Interview Cutoff: 32-40%, Selection: 41-47%, Topper: 48-55%+)."""
    if pct < 32.0:
        return UPSCPerformanceBand.NEEDS_FOUNDATION
    if pct <= 40.0:
        return UPSCPerformanceBand.AVERAGE
    if pct < 48.0:
        return UPSCPerformanceBand.GOOD
    return UPSCPerformanceBand.TOPPER


def _parse_micro_demands(raw_demands: Any) -> list[MicroDemandItem]:
    """Builds typed micro-demand items, skipping (never failing on) malformed entries."""
    micro_demands: list[MicroDemandItem] = []
    for d_raw in raw_demands or []:
        try:
            micro_demands.append(
                MicroDemandItem(
                    demand=d_raw.get("demand", "Sub-demand"),
                    status=_match_enum(DemandStatus, d_raw.get("status"), DemandStatus.PARTIALLY_ADDRESSED),
                    marks_allocated=float(d_raw.get("marks_allocated", 2.5)),
                    marks_obtained=float(d_raw.get("marks_obtained", 1.0)),
                    comment=d_raw.get("comment", ""),
                )
            )
        except Exception as e:
            logger.warning(f"Error parsing micro-demand: {e}")
    return micro_demands


def _parse_citation_audit(raw_citations: Any) -> CitationAudit:
    """Splits the LLM citation audit into KB anchors, open-world credits, and hallucinations."""
    raw_citations = raw_citations or {}
    return CitationAudit(
        mandatory_kb_anchors=[
            CitationItem(
                name=c.get("name", ""),
                status=_parse_citation_status(c.get("status")),
                source="knowledge_base",
                notes=c.get("notes", ""),
            )
            for c in raw_citations.get("mandatory_kb_anchors", [])
        ],
        open_world_credits=[
            CitationItem(
                name=c.get("name", ""),
                status=CitationStatus.OPEN_WORLD_CREDITED,
                source="open_world",
                notes=c.get("notes", ""),
            )
            for c in raw_citations.get("open_world_credits", [])
        ],
        hallucinated_citations=[
            CitationItem(
                name=c.get("name", ""),
                status=CitationStatus.HALLUCINATED_OR_WRONG,
                source="knowledge_base",
                notes=c.get("notes", ""),
            )
            for c in raw_citations.get("hallucinated_citations", [])
        ],
        summary=raw_citations.get("summary", "Factual and statutory grounding evaluated."),
    )


def _parse_presentation(raw_pres: Any) -> PresentationEvaluation:
    """Normalizes the presentation audit, clamping the density score and the bounded bonus."""
    raw_pres = raw_pres or {}
    return PresentationEvaluation(
        detected_archetype=_match_enum(
            PresentationArchetype,
            raw_pres.get("detected_archetype"),
            PresentationArchetype.PARAGRAPH_HEAVY,
        ),
        visual_density_score=max(0.0, min(10.0, float(raw_pres.get("visual_density_score", 5.0)))),
        diagrams_and_tables_found=raw_pres.get("diagrams_and_tables_found", []),
        presentation_bonus=max(
            0.0, min(MAX_PRESENTATION_BONUS, float(raw_pres.get("presentation_bonus", 0.0)))
        ),
        examiner_critique=raw_pres.get("examiner_critique", "Adequate presentation format."),
        topper_reformatting_tip=raw_pres.get(
            "topper_reformatting_tip",
            "Structure key points under explicit subheadings with numbered bullet points.",
        ),
    )


def _parse_diagnostic(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """Normalizes the raw Call-1 JSON into typed evaluation parts, failing fast on malformed outputs."""
    if not diagnostic:
        raise ModelInvocationError("Call 1 Diagnostic LLM returned empty or unparseable JSON response.")

    cot_trail = str(diagnostic.get("cot_reasoning_trail", "")).strip()
    if not cot_trail:
        raise ModelInvocationError("Call 1 Diagnostic LLM omitted 'cot_reasoning_trail'.")

    pillar_summary = diagnostic.get("pillar_summary")
    if not isinstance(pillar_summary, dict) or not pillar_summary:
        raise ModelInvocationError("Call 1 Diagnostic LLM omitted or produced invalid 'pillar_summary'.")

    is_off_topic = bool(diagnostic.get("is_off_topic", False))
    return {
        "cot_trail": cot_trail,
        "is_off_topic": is_off_topic,
        "demand_relevance_gate": float(
            diagnostic.get("demand_relevance_gate", 1.0 if not is_off_topic else OFF_TOPIC_GATE_DEFAULT)
        ),
        "directive": _match_enum(DirectiveType, diagnostic.get("directive_detected"), None),
        "micro_demands": _parse_micro_demands(diagnostic.get("micro_demands")),
        "citation_audit": _parse_citation_audit(diagnostic.get("citation_audit")),
        "presentation": _parse_presentation(diagnostic.get("presentation")),
        "strengths": [str(s) for s in diagnostic.get("strengths", []) if s],
        "weaknesses": [str(w) for w in diagnostic.get("weaknesses", []) if w],
        "topper_action_plan": [str(a) for a in diagnostic.get("topper_action_plan", []) if a],
        "pillar_summary": pillar_summary,
    }


def _warn_on_degraded_scoring(pillars: dict[str, PillarGEvalScore]) -> None:
    """Surfaces degraded scoring so a fallback run is never mistaken for a real logprob G-Eval."""
    degraded = [name for name, p in pillars.items() if p.scoring_method != "logprob"]
    if degraded:
        logger.warning(
            f"G-Eval ran without native logprobs for pillars {degraded} "
            f"(method: {pillars[degraded[0]].scoring_method}); ratings are point estimates."
        )


@traceable(name="Apply_UPSC_Marking_Policy", run_type="tool")
def _apply_marking_policy(
    pillars: dict[str, PillarGEvalScore],
    presentation: PresentationEvaluation,
    is_off_topic: bool,
    demand_relevance_gate: float,
    max_marks: float,
) -> tuple[float, UPSCPerformanceBand]:
    """Applies the presentation bonus, the off-topic gate, and authentic UPSC score ceilings."""
    raw_total = sum(p.calibrated_score for p in pillars.values()) + presentation.presentation_bonus

    if is_off_topic:
        logger.warning("Hard Demand Relevance Gate triggered: Off-topic answer detected!")
        total = round(min(OFF_TOPIC_MAX_MARKS, raw_total * demand_relevance_gate), 2)
        return total, UPSCPerformanceBand.NEEDS_FOUNDATION

    topper_ceiling = max_marks * TOPPER_CEILING_PCT
    total = round(min(topper_ceiling, max(MIN_SCORE_FLOOR, raw_total)), 2)
    return total, _classify_performance_band((total / max_marks) * 100.0)


class UPSCEvaluationEngine:
    """Enterprise-grade, calibrated UPSC Civil Services Mains Answer Evaluation Engine."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        eval_llm: Any = None,
        geval_scorer: BaseGEvalScorer | None = None,
    ):
        self.retriever = retriever or HybridRetriever(top_k=6)
        self.llm = eval_llm or get_eval_llm()
        self.geval_scorer = geval_scorer or get_geval_scorer()
        self._kb_cache: dict[str, list[str]] = {}


    @traceable(name="Evaluate_Question_Answer", run_type="chain")
    def evaluate_answer(
        self,
        question: str,
        candidate_answer: str,
        max_marks: float = 10.0,
    ) -> EvaluationResult:
        """Evaluates a candidate answer using 2-call CoT + G-Eval probability trailing."""
        logger.info(f"Starting UPSC evaluation for question ({int(max_marks)} marks)...")

        # Fast-path for unattempted / blank answers (award authentic 0.0 marks with zero API waste)
        if not candidate_answer or not candidate_answer.strip():
            logger.info("Question left blank/unattempted. Awarding 0.0 marks.")
            empty_pillars = {
                pillar_type.value: PillarGEvalScore(
                    pillar=pillar_type,
                    pillar_name=cfg["name"],
                    weight_pct=cfg["weight"],
                    max_marks=round(max_marks * cfg["weight"], 2),
                    discrete_probabilities={1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
                    raw_expected_rating=1.0,
                    calibrated_score=0.0,
                    feedback="Question left completely unattempted.",
                )
                for pillar_type, cfg in PILLAR_CONFIGS.items()
            }
            return EvaluationResult(
                question=question,
                candidate_answer=candidate_answer,
                max_marks=max_marks,
                total_score=0.0,
                percentage=0.0,
                performance_band=UPSCPerformanceBand.NEEDS_FOUNDATION,
                is_off_topic=False,
                demand_relevance_gate=0.0,
                directive_detected=DirectiveType.DISCUSS,
                cot_reasoning_trail="Question left unattempted by the candidate. Zero marks awarded.",
                micro_demands=[],
                pillars=empty_pillars,
                presentation=PresentationEvaluation(
                    detected_archetype=PresentationArchetype.PARAGRAPH_HEAVY,
                    visual_density_score=0.0,
                    examiner_critique="No attempt made.",
                    topper_reformatting_tip="Attempt all questions under exam time constraints.",
                ),
                citation_audit=CitationAudit(),
                strengths=[],
                weaknesses=["Question left completely unattempted."],
                topper_action_plan=["Attempt all questions in GS-2 to capture step marks."],
            )

        kb_context = self._retrieve_ground_truth(question)
        parts = self._run_diagnostic(question, candidate_answer, kb_context, max_marks)

        logger.info("Executing Call 2: G-Eval probabilistic logprob scoring head...")
        pillars = self.geval_scorer.score_pillars(
            question=question,
            candidate_answer=candidate_answer,
            cot_reasoning_trail=parts["cot_trail"],
            pillar_summary=parts["pillar_summary"],
            max_marks=max_marks,
        )
        _warn_on_degraded_scoring(pillars)

        total_score, band = _apply_marking_policy(
            pillars=pillars,
            presentation=parts["presentation"],
            is_off_topic=parts["is_off_topic"],
            demand_relevance_gate=parts["demand_relevance_gate"],
            max_marks=max_marks,
        )
        pct = round((total_score / max_marks) * 100.0, 1)

        result = EvaluationResult(
            question=question,
            candidate_answer=candidate_answer,
            max_marks=max_marks,
            total_score=total_score,
            percentage=pct,
            performance_band=band,
            is_off_topic=parts["is_off_topic"],
            demand_relevance_gate=parts["demand_relevance_gate"],
            directive_detected=parts["directive"],
            cot_reasoning_trail=parts["cot_trail"],
            micro_demands=parts["micro_demands"],
            pillars=pillars,
            presentation=parts["presentation"],
            citation_audit=parts["citation_audit"],
            strengths=parts["strengths"],
            weaknesses=parts["weaknesses"],
            topper_action_plan=parts["topper_action_plan"],
        )

        logger.info(f"Evaluation completed: Score = {total_score} / {max_marks} ({pct}%) | Band = {band.value}")
        return result

    @traceable(name="KB_Ground_Truth_Retrieval", run_type="retriever")
    def _retrieve_ground_truth(self, question: str) -> list[str]:
        """Step 1: hybrid retrieval of authentic KB context used for grounding verification."""
        if question in self._kb_cache:
            return self._kb_cache[question]
        kb_context: list[str] = self.retriever.get_retrieval_context(question, top_k=self.retriever.top_k)
        self._kb_cache[question] = kb_context
        logger.info(f"Retrieved {len(kb_context)} ground-truth context blocks from Knowledge Base.")
        return kb_context

    @traceable(name="Call1_Diagnostic_CoT", run_type="chain")
    def _run_diagnostic(
        self,
        question: str,
        candidate_answer: str,
        kb_context: list[str],
        max_marks: float,
    ) -> dict[str, Any]:
        """Step 2: Call 1 deep CoT diagnostic, returned as typed evaluation parts."""
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
        response = self.llm.invoke(messages)
        parsed = extract_json_dict(response.content)
        if not parsed:
            logger.warning(
                f"Call 1 Diagnostic JSON parse failed on initial attempt. Raw snippet: {response.content[:200]}... Retrying once..."
            )
            response = self.llm.invoke(messages)
            parsed = extract_json_dict(response.content)
            if not parsed:
                logger.error(f"Call 1 Diagnostic raw output (first 1000 chars): {response.content[:1000]}")

        return _parse_diagnostic(parsed)
