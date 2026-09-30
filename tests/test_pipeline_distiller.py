"""Unit tests for UnifiedEvaluationPipeline distillation logic (offline, zero API calls)."""
from src.models.enums import DirectiveType, PresentationArchetype, UPSCPerformanceBand
from src.models.evaluation import CitationAudit, EvaluationResult, PresentationEvaluation
from src.models.parsing import ParsedQuestion
from src.pipeline import UnifiedEvaluationPipeline


def _mock_eval_result(
    score: float,
    max_marks: float = 10.0,
    strengths: list[str] | None = None,
    weaknesses: list[str] | None = None,
    action_plan: list[str] | None = None,
) -> EvaluationResult:
    """Helper to mock an EvaluationResult."""
    return EvaluationResult(
        question="Test Question",
        candidate_answer="Test Answer",
        max_marks=max_marks,
        total_score=score,
        percentage=(score / max_marks) * 100,
        performance_band=UPSCPerformanceBand.GOOD,
        is_off_topic=False,
        demand_relevance_gate=1.0,
        directive_detected=DirectiveType.DISCUSS,
        cot_reasoning_trail="Internal CoT trace",
        micro_demands=[],
        pillars={},
        presentation=PresentationEvaluation(
            detected_archetype=PresentationArchetype.PARAGRAPH_HEAVY,
            visual_density_score=5.0,
            examiner_critique="Good presentation",
            topper_reformatting_tip="Use diagrams",
        ),
        citation_audit=CitationAudit(),
        strengths=strengths or ["Strong conceptual clarity", "Accurate constitutional citations"],
        weaknesses=weaknesses or ["Could deepen economic impacts"],
        topper_action_plan=action_plan or ["Cite 2nd ARC 6th Report", "Add a forward-looking conclusion"],
    )


def test_pipeline_distill_results_offline():
    """Verifies that distill_results creates a pure student-first report with zero performance bands."""
    # Q1: Attempted, strong answer
    q1 = ParsedQuestion(
        q_num=1,
        max_marks=10.0,
        question="Discuss the significance of the 73rd Constitutional Amendment.",
        candidate_answer="The 73rd Amendment gave constitutional status to Panchayati Raj Institutions...",
        is_blank=False,
    )
    res1 = _mock_eval_result(
        score=6.0,
        max_marks=10.0,
        strengths=["Clear coverage of 11th Schedule", "Cited Article 243G and 243I"],
        weaknesses=["Did not elaborate on 3Fs (Funds, Functions, Functionaries)"],
        action_plan=["Emphasize devolution of 3Fs", "Cite Devolution Index by MoPR"],
    )

    # Q2: Blank / unattempted answer
    q2 = ParsedQuestion(
        q_num=2,
        max_marks=10.0,
        question="Analyze the role of the Competition Commission of India (CCI).",
        candidate_answer="",
        is_blank=True,
    )
    res2 = _mock_eval_result(
        score=0.0,
        max_marks=10.0,
        strengths=[],
        weaknesses=["Question left unattempted"],
        action_plan=["Attempt all questions in GS-2"],
    )

    eval_pairs = [(q1, res1), (q2, res2)]
    report = UnifiedEvaluationPipeline.distill_results("GS-II_Test.pdf", eval_pairs)

    # Check overall summary
    assert report.document == "GS-II_Test.pdf"
    assert report.summary.total_score == 6.0
    assert report.summary.max_marks == 20.0
    assert report.summary.percentage == 30.0
    assert len(report.summary.overall_feedback.key_strengths) > 0
    assert len(report.summary.overall_feedback.top_areas_to_improve) > 0

    # Check Q1
    q1_out = report.questions[0]
    assert q1_out.q_num == 1
    assert q1_out.score == 6.0
    assert q1_out.percentage == 60.0
    assert "Clear coverage of 11th Schedule" in q1_out.pros
    assert any("3Fs" in tip for tip in q1_out.what_to_do_better)

    # Check Q2 (Unattempted)
    q2_out = report.questions[1]
    assert q2_out.q_num == 2
    assert q2_out.score == 0.0
    assert q2_out.percentage == 0.0
    assert q2_out.pros == []
    assert len(q2_out.what_to_do_better) > 0

    # Check complete absence of performance band strings in model dump
    dump = report.model_dump()
    dump_str = str(dump)
    assert "performance_band" not in dump_str
    assert "overall_band" not in dump_str
    assert "trajectory_verdict" not in dump_str
    assert "Good Attempt" not in dump_str
