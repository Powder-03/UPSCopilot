"""Unit tests for AWS Serverless services: StorageService, JobStateService, QueueService, and Lambda handlers."""
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.api import JobStatus
from src.services.job_state_service import (
    JobStateService,
    _decimals_to_floats,
    _floats_to_decimals,
)
from src.services.queue_service import QueueService
from src.services.storage_service import StorageService


def test_float_decimal_conversion():
    """Verifies recursive float <-> Decimal conversion for DynamoDB compatibility."""
    raw_data = {
        "score": 4.5,
        "max_marks": 10.0,
        "integer_val": 42,
        "nested": {"percent": 45.0, "notes": ["Clear", 8.5]},
    }
    converted = _floats_to_decimals(raw_data)
    assert isinstance(converted["score"], Decimal)
    assert converted["score"] == Decimal("4.5")
    assert isinstance(converted["nested"]["notes"][1], Decimal)
    assert converted["integer_val"] == 42

    restored = _decimals_to_floats(converted)
    assert restored["score"] == 4.5
    assert restored["nested"]["notes"][1] == 8.5
    assert restored["integer_val"] == 42


def test_storage_service_local_and_s3(tmp_path: Path):
    """Tests StorageService in local mode and mocked S3 mode."""
    # 1. Local mode
    storage_local = StorageService(local_dir=tmp_path / "uploads", s3_bucket=None)
    assert not storage_local.is_s3_enabled

    ref = storage_local.save_file(b"%PDF-1.4 mock", "test.pdf", "job_123")
    assert Path(ref).exists()
    assert storage_local.get_local_path(ref) == Path(ref)

    # 2. Mocked S3 mode
    storage_s3 = StorageService(local_dir=tmp_path / "uploads", s3_bucket="my-eval-bucket")
    assert storage_s3.is_s3_enabled

    with patch.object(storage_s3, "_s3_client", create=True) as mock_s3:
        mock_s3.put_object = MagicMock()
        mock_s3.download_file = MagicMock()

        s3_ref = storage_s3.save_file(b"%PDF-1.4 mock s3", "copy.pdf", "job_456")
        assert s3_ref == "s3://my-eval-bucket/uploads/job_456_copy.pdf"
        mock_s3.put_object.assert_called_once()

        # Download from S3
        temp_dest = tmp_path / "downloads"
        resolved = storage_s3.get_local_path("s3://my-eval-bucket/uploads/job_456_copy.pdf", temp_dir=temp_dest)
        assert resolved.name == "job_456_copy.pdf"
        mock_s3.download_file.assert_called_once_with(
            "my-eval-bucket", "uploads/job_456_copy.pdf", str(resolved)
        )


def test_job_state_service_local_and_dynamo(tmp_path: Path):
    """Tests JobStateService in local disk mode and mocked DynamoDB mode."""
    # 1. Local disk mode
    state_local = JobStateService(local_dir=tmp_path / "jobs", dynamodb_table=None)
    assert not state_local.is_dynamodb_enabled

    job_data = {
        "job_id": "job_local_1",
        "status": JobStatus.QUEUED.value,
        "progress_pct": 0,
        "score": 4.5,
    }
    state_local.save_job("job_local_1", job_data)
    loaded = state_local.get_job("job_local_1")
    assert loaded is not None
    assert loaded["job_id"] == "job_local_1"
    assert loaded["score"] == 4.5

    # 2. Mocked DynamoDB mode
    state_dynamo = JobStateService(local_dir=tmp_path / "jobs", dynamodb_table="eval-jobs-table")
    assert state_dynamo.is_dynamodb_enabled

    with patch.object(state_dynamo, "_table", create=True) as mock_table:
        mock_table.put_item = MagicMock()
        mock_table.get_item.return_value = {
            "Item": {
                "job_id": "job_dynamo_1",
                "status": JobStatus.COMPLETED.value,
                "score": Decimal("7.5"),
            }
        }

        state_dynamo.save_job("job_dynamo_1", job_data)
        mock_table.put_item.assert_called_once()

        fetched = state_dynamo.get_job("job_dynamo_1")
        assert fetched is not None
        assert fetched["score"] == 7.5


def test_queue_service_thread_and_sqs(tmp_path: Path):
    """Tests QueueService dispatching via local thread runner and SQS."""
    # 1. Local thread fallback
    queue_local = QueueService(queue_url=None)
    assert not queue_local.is_sqs_enabled

    runner_called = []

    def mock_runner(job_id, path, start_p, max_p):
        runner_called.append(job_id)

    dispatch_type = queue_local.dispatch(
        job_id="job_thread_1",
        pdf_path=tmp_path / "mock.pdf",
        local_thread_runner=mock_runner,
    )
    assert dispatch_type == "thread"

    # 2. SQS mode
    queue_sqs = QueueService(queue_url="https://sqs.us-east-1.amazonaws.com/123/eval-queue")
    assert queue_sqs.is_sqs_enabled

    with patch.object(queue_sqs, "_sqs_client", create=True) as mock_sqs:
        mock_sqs.send_message = MagicMock()
        dispatch_type = queue_sqs.dispatch(
            job_id="job_sqs_1",
            pdf_path="s3://bucket/test.pdf",
            start_page=3,
            max_pages=5,
        )
        assert dispatch_type == "sqs"
        mock_sqs.send_message.assert_called_once()
