"""Unit tests for Decoupled AWS Lambda Workers: lambda_vision and lambda_eval."""
import json
from unittest.mock import patch

from src.lambda_eval import handler as eval_handler
from src.lambda_vision import handler as vision_handler
from src.models.api import JobStatus
from src.models.enums import DirectiveType, PresentationArchetype, UPSCPerformanceBand
from src.models.evaluation import CitationAudit, EvaluationResult, PresentationEvaluation
from src.models.parsing import ParsedDocument, ParsedQuestion


def _build_dummy_parsed_doc() -> ParsedDocument:
    """Creates a mock ParsedDocument for testing."""
    return ParsedDocument(
        source_file="test_booklet.pdf",
        total_pages=2,
        questions=[
            ParsedQuestion(
                q_num=1,
                page_numbers=[1],
                max_marks=10.0,
                question="Discuss the significance of judicial review.",
                candidate_answer="Judicial review is a basic structure of the Constitution.",
                word_count=10,
                is_blank=False,
            ),
            ParsedQuestion(
                q_num=2,
                page_numbers=[2],
                max_marks=15.0,
                question="Examine the role of the Finance Commission.",
                candidate_answer="Article 280 provides for the Finance Commission.",
                word_count=9,
                is_blank=False,
            ),
        ],
    )


def _build_mock_eval_result(score: float = 5.0, max_marks: float = 10.0) -> EvaluationResult:
    """Builds a stub EvaluationResult."""
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
        strengths=["Solid constitutional grounding"],
        weaknesses=["Need more current committee references"],
        topper_action_plan=["Add 2nd ARC recommendations"],
    )


# =====================================================================
# Lambda Vision Worker Tests
# =====================================================================


def test_lambda_vision_sqs_event():
    """Verifies that lambda_vision processes SQS records, runs OCR, and pushes to EvalQueue."""
    sqs_event = {
        "Records": [
            {
                "messageId": "vision_msg_101",
                "body": json.dumps({
                    "job_id": "job_vis_001",
                    "pdf_path": "s3://eval-bucket/uploads/test.pdf",
                    "email": "aspirant@test.com",
                    "start_page": 1,
                    "max_pages": 2,
                }),
            }
        ]
    }

    mock_parsed_doc = _build_dummy_parsed_doc()

    with (
        patch("src.lambda_vision.StorageService") as mock_storage_cls,
        patch("src.lambda_vision.JobStateService") as mock_state_cls,
        patch("src.lambda_vision.QueueService") as mock_queue_cls,
        patch("src.lambda_vision.DocumentParsingPipeline") as mock_pipeline_cls,
    ):
        mock_storage = mock_storage_cls.return_value
        mock_storage.get_local_path.return_value = "/tmp/test.pdf"

        mock_state = mock_state_cls.return_value
        mock_state.get_job.return_value = {"job_id": "job_vis_001"}

        mock_queue = mock_queue_cls.return_value

        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.parse_pdf.return_value = mock_parsed_doc

        resp = vision_handler(sqs_event)

        assert resp == {"batchItemFailures": []}
        mock_pipeline.parse_pdf.assert_called_once_with(
            pdf_path="/tmp/test.pdf",
            start_page=1,
            max_pages=2,
        )
        mock_queue.dispatch_eval.assert_called_once_with(
            job_id="job_vis_001",
            email="aspirant@test.com",
        )
        assert mock_state.save_job.call_count >= 2


def test_lambda_vision_direct_invocation():
    """Verifies direct invocation of the lambda_vision worker."""
    event = {
        "job_id": "job_vis_direct",
        "pdf_path": "local/test.pdf",
        "email": "student@upsc.org",
    }
    mock_parsed_doc = _build_dummy_parsed_doc()

    with (
        patch("src.lambda_vision.StorageService") as mock_storage_cls,
        patch("src.lambda_vision.JobStateService") as mock_state_cls,
        patch("src.lambda_vision.QueueService") as mock_queue_cls,
        patch("src.lambda_vision.DocumentParsingPipeline") as mock_pipeline_cls,
    ):
        mock_storage_cls.return_value.get_local_path.return_value = "/tmp/test.pdf"
        mock_state_cls.return_value.get_job.return_value = {"job_id": "job_vis_direct"}
        mock_pipeline_cls.return_value.parse_pdf.return_value = mock_parsed_doc

        resp = vision_handler(event)
        assert resp["status"] == "success"
        assert resp["job_id"] == "job_vis_direct"
        assert resp["questions_parsed"] == 2
        mock_queue_cls.return_value.dispatch_eval.assert_called_once()


def test_lambda_vision_batch_failure():
    """Verifies that an unhandled error inside a record appends the messageId to batchItemFailures."""
    sqs_event = {
        "Records": [
            {
                "messageId": "msg_fail_vis",
                "body": json.dumps({"job_id": "job_fail", "pdf_path": "bad.pdf"}),
            }
        ]
    }

    with patch("src.lambda_vision.StorageService") as mock_storage_cls:
        mock_storage_cls.return_value.get_local_path.side_effect = RuntimeError("S3 Download Corrupt")
        resp = vision_handler(sqs_event)

        assert len(resp["batchItemFailures"]) == 1
        assert resp["batchItemFailures"][0]["itemIdentifier"] == "msg_fail_vis"


# =====================================================================
# Lambda Evaluation Worker Tests
# =====================================================================


def test_lambda_eval_sqs_event():
    """Verifies that lambda_eval loads parsed_document from state, evaluates, renders PDF, and sends email."""
    sqs_event = {
        "Records": [
            {
                "messageId": "eval_msg_201",
                "body": json.dumps({
                    "job_id": "job_ev_001",
                    "email": "candidate@mains.in",
                }),
            }
        ]
    }

    parsed_doc = _build_dummy_parsed_doc()
    stored_state = {
        "job_id": "job_ev_001",
        "status": JobStatus.EVALUATING.value,
        "email": "candidate@mains.in",
        "parsed_document": parsed_doc.model_dump(),
    }

    mock_eval_res = _build_mock_eval_result()

    with (
        patch("src.lambda_eval.JobStateService") as mock_state_cls,
        patch("src.lambda_eval.UPSCEvaluationEngine") as mock_engine_cls,
        patch("src.lambda_eval.generate_scorecard_pdf") as mock_pdf_func,
        patch("src.lambda_eval.EmailService") as mock_email_cls,
    ):
        mock_state = mock_state_cls.return_value
        mock_state.get_job.return_value = stored_state

        mock_engine = mock_engine_cls.return_value
        mock_engine.evaluate_answer.return_value = mock_eval_res

        mock_pdf_func.return_value = b"%PDF-1.4 mock evaluation pdf bytes"

        mock_email = mock_email_cls.return_value
        mock_email.send_evaluation_email.return_value = True

        resp = eval_handler(sqs_event)

        assert resp == {"batchItemFailures": []}
        assert mock_engine.evaluate_answer.call_count == 2
        mock_pdf_func.assert_called_once()
        mock_email.send_evaluation_email.assert_called_once_with(
            to_email="candidate@mains.in",
            report=mock_email.send_evaluation_email.call_args[1]["report"],
            job_id="job_ev_001",
            pdf_bytes=b"%PDF-1.4 mock evaluation pdf bytes",
        )
        saved_call = mock_state.save_job.call_args[0][1]
        assert saved_call["status"] == JobStatus.COMPLETED.value
        assert "parsed_document" not in saved_call
        assert saved_call["email_sent"] is True


def test_lambda_eval_batch_failure():
    """Verifies that an error in lambda_eval correctly flags messageId for redrive."""
    sqs_event = {
        "Records": [
            {
                "messageId": "msg_fail_ev",
                "body": json.dumps({"job_id": "job_nonexistent"}),
            }
        ]
    }

    with patch("src.lambda_eval.JobStateService") as mock_state_cls:
        mock_state_cls.return_value.get_job.return_value = None  # Job not found

        resp = eval_handler(sqs_event)
        assert len(resp["batchItemFailures"]) == 1
        assert resp["batchItemFailures"][0]["itemIdentifier"] == "msg_fail_ev"
