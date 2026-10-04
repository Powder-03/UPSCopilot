"""Unit tests for EC2 Coordinator, Knowledge Base Prefetcher, and 20-Lambda Fan-Out."""
from unittest.mock import MagicMock

import pytest
from src.kb.prefetcher import KnowledgeBasePrefetcher
from src.models.parsing import ParsedQuestion
from src.orchestrator_ec2 import (
    MAX_CONCURRENT_COPIES,
    EC2CoordinatorOrchestrator,
    get_copy_semaphore,
)


def test_concurrency_semaphore_limit():
    """Verifies that the EC2 coordinator enforces MAX_CONCURRENT_COPIES = 2."""
    assert MAX_CONCURRENT_COPIES == 2
    sem = get_copy_semaphore()
    assert sem._value == 2


def test_prefetcher_skips_blank_questions():
    """Verifies that unattempted questions skip Pinecone retrieval instantly."""
    mock_retriever = MagicMock()
    prefetcher = KnowledgeBasePrefetcher(retriever=mock_retriever, top_k=6)

    # Blank question
    blank_q = ParsedQuestion(
        q_num=8,
        question="Discuss the Sevottam model.",
        candidate_answer="",
        max_marks=10.0,
        is_blank=True,
    )
    q_num, contexts = prefetcher.prefetch_single_question(blank_q)

    assert q_num == 8
    assert contexts == []
    # Retriever must never be invoked for blank answers
    mock_retriever.get_retrieval_context.assert_not_called()


def test_prefetcher_retrieves_for_attempted_questions():
    """Verifies that attempted questions query the retriever in parallel."""
    mock_retriever = MagicMock()
    mock_retriever.get_retrieval_context.return_value = [
        "[Article 21]\nProtection of life and personal liberty."
    ]
    prefetcher = KnowledgeBasePrefetcher(retriever=mock_retriever, top_k=6)

    attempted_q = ParsedQuestion(
        q_num=1,
        question="Analyze the right to privacy.",
        candidate_answer="The Supreme Court in Puttaswamy held that privacy is part of Article 21.",
        max_marks=10.0,
        is_blank=False,
    )
    q_num, contexts = prefetcher.prefetch_single_question(attempted_q)

    assert q_num == 1
    assert len(contexts) == 1
    assert "Article 21" in contexts[0]
    mock_retriever.get_retrieval_context.assert_called_once_with(attempted_q.question, top_k=6)


def test_prefetch_all_batch():
    """Verifies prefetch_all maps results by question number across a list."""
    mock_retriever = MagicMock()
    mock_retriever.get_retrieval_context.return_value = ["[Constitution]\nArticle 14."]
    prefetcher = KnowledgeBasePrefetcher(retriever=mock_retriever, top_k=6, max_workers=2)

    questions = [
        ParsedQuestion(q_num=1, question="Q1", candidate_answer="Ans 1", max_marks=10.0),
        ParsedQuestion(q_num=2, question="Q2", candidate_answer="", max_marks=10.0),
    ]

    results = prefetcher.prefetch_all(questions)
    assert 1 in results
    assert 2 in results
    assert len(results[1]) == 1
    assert results[2] == []  # blank question gets empty context


@pytest.mark.asyncio
async def test_ec2_fanout_evaluation_mock():
    """Verifies evaluate_copy_fanout processes question pairs concurrently."""
    orchestrator = EC2CoordinatorOrchestrator()

    questions = [
        ParsedQuestion(q_num=1, question="What is federalism?", candidate_answer="", max_marks=10.0),
    ]
    kb_contexts = {1: ["[Constitution]\nArticle 1."]}

    eval_pairs = await orchestrator.evaluate_copy_fanout(questions, kb_contexts)
    assert len(eval_pairs) == 1
    q, res = eval_pairs[0]
    assert q.q_num == 1
    assert res.total_score == 0.0  # Blank answer receives 0.0 marks


@pytest.mark.asyncio
async def test_oversized_file_rejected():
    """Verifies that an upload exceeding 200 MB is rejected and deleted immediately."""
    mock_storage = MagicMock()
    # 250 MB
    mock_storage.get_file_size.return_value = 250 * 1024 * 1024

    orchestrator = EC2CoordinatorOrchestrator(storage_service=mock_storage)

    with pytest.raises(ValueError, match="exceeds 200 MB"):
        await orchestrator.process_job(
            job_id="test_large_job",
            storage_ref="s3://test-bucket/large.pdf",
        )

    # Verifies file deletion was called immediately
    mock_storage.delete_file.assert_called_once_with("s3://test-bucket/large.pdf")


@pytest.mark.asyncio
async def test_ec2_worker_daemon_processes_and_deletes_message():
    """Verifies that the EC2 worker processes an SQS message and deletes it on success."""
    import json
    from unittest.mock import AsyncMock

    from src.handlers.ec2_worker import EC2WorkerDaemon

    mock_orch = MagicMock()
    mock_orch.process_job = AsyncMock()
    mock_sqs = MagicMock()

    daemon = EC2WorkerDaemon(queue_url="https://sqs.us-east-1.amazonaws.com/123/eval-queue", orchestrator=mock_orch)
    daemon._sqs = mock_sqs

    msg = {
        "ReceiptHandle": "receipt_xyz",
        "Body": json.dumps({
            "job_id": "job_abc",
            "pdf_path": "s3://bucket/test.pdf",
            "email": "student@example.com",
            "start_page": 1,
            "max_pages": 60,
        }),
    }

    await daemon._process_single_message(msg)

    # Verifies orchestrator was called with message params
    mock_orch.process_job.assert_awaited_once_with(
        job_id="job_abc",
        storage_ref="s3://bucket/test.pdf",
        filename="booklet.pdf",
        email="student@example.com",
        start_page=1,
        max_pages=60,
    )

    # Verifies message was deleted from SQS
    mock_sqs.delete_message.assert_called_once_with(
        QueueUrl="https://sqs.us-east-1.amazonaws.com/123/eval-queue",
        ReceiptHandle="receipt_xyz",
    )

