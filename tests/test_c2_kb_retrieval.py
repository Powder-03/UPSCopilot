"""Component-Level RAG Retrieval Evaluation Test Suite.

Tests:
1. Exact Statutory Guard (Direct Article numbers & Cases hit Rank 1)
2. Benchmark Recall@K (K=5) >= 90%
3. Mean Reciprocal Rank (MRR) >= 0.85
4. Typo and Alias Robustness
5. Hallucination / Negative Distractor Defense
"""
import pytest
from typing import List
from src.kb.retriever import HybridRetriever
from src.kb.benchmark import RETRIEVAL_GOLDEN_BENCHMARK


@pytest.fixture(scope="module")
def retriever():
    """Initializes and provides the HybridRetriever instance."""
    return HybridRetriever()


def test_kb_dataset_loaded(retriever):
    """Verifies that all curated knowledge base items are loaded properly."""
    assert len(retriever.items) >= 30, f"Expected >= 30 items, found {len(retriever.items)}"
    assert "art_163" in retriever.item_by_id
    assert "case_sr_bommai_1994" in retriever.item_by_id
    assert "comm_sarkaria_1988" in retriever.item_by_id


def test_exact_statutory_guard_articles(retriever):
    """Verifies that queries mentioning explicit Article numbers return that exact Article at Rank 1."""
    statutory_queries = [
        ("What are the powers under Article 163?", "art_163"),
        ("Examine the abuse of Article 356 in state governance", "art_356"),
        ("Discuss assent to bills under Article 200", "art_200"),
        ("Scope of Right to Life under Article 21", "art_21"),
        ("Significance of Article 32 as heart and soul of Constitution", "art_32"),
        ("Provisions of Article 239AA regarding governance of Delhi", "art_239aa"),
        ("Disqualification under Tenth Schedule anti-defection", "sched_10"),
    ]

    for query, expected_id in statutory_queries:
        results = retriever.retrieve_hybrid(query, top_k=3)
        assert len(results) > 0, f"No results for query: {query}"
        top_result = results[0]
        assert top_result.item.id == expected_id, (
            f"Exact statutory guard failed for '{query}'. "
            f"Expected {expected_id} at Rank 1, got {top_result.item.id}."
        )


def test_exact_case_law_guard(retriever):
    """Verifies that queries with landmark case names place that case at Rank 1."""
    case_queries = [
        ("Evaluate the guidelines laid down in S.R. Bommai case", "case_sr_bommai_1994"),
        ("Evolution of Basic Structure in Kesavananda Bharati", "case_kesavananda_1973"),
        ("Right to Privacy principles in Puttaswamy judgment", "case_puttaswamy_2017"),
        ("Governor summoning assembly in Nabam Rebia case", "case_nabam_rebia_2016"),
        ("Harmonious balance between FR and DPSP in Minerva Mills", "case_minerva_mills_1980"),
        ("Election Commission appointment in Anoop Baranwal case", "case_anoop_baranwal_2023"),
    ]

    for query, expected_id in case_queries:
        results = retriever.retrieve_hybrid(query, top_k=3)
        assert len(results) > 0, f"No results for query: {query}"
        top_result = results[0]
        assert top_result.item.id == expected_id, (
            f"Case law guard failed for '{query}'. "
            f"Expected {expected_id} at Rank 1, got {top_result.item.id}."
        )


def test_recall_at_5_and_mrr_golden_benchmark(retriever):
    """
    Evaluates the retriever on the 10 Golden Benchmark UPSC questions.
    Asserts:
    - Recall@5 >= 90%
    - MRR (Mean Reciprocal Rank) >= 0.85
    """
    total_expected = 0
    total_retrieved = 0
    reciprocal_ranks: List[float] = []

    for item in RETRIEVAL_GOLDEN_BENCHMARK:
        results = retriever.retrieve_hybrid(item.question_text, top_k=5)
        retrieved_ids = [r.item.id for r in results]

        # Calculate Recall@5 for this query
        matched_expected = [eid for eid in item.expected_ids if eid in retrieved_ids]
        total_expected += len(item.expected_ids)
        total_retrieved += len(matched_expected)

        # Calculate Reciprocal Rank for primary ID
        if item.primary_id in retrieved_ids:
            rank = retrieved_ids.index(item.primary_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_at_5 = total_retrieved / total_expected if total_expected > 0 else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    print(f"\n[RAG Benchmark Results] Recall@5: {recall_at_5:.2%}, MRR: {mrr:.4f}")

    # Assertions
    assert recall_at_5 >= 0.90, f"Recall@5 threshold failed: {recall_at_5:.2%} < 90%"
    assert mrr >= 0.85, f"MRR threshold failed: {mrr:.4f} < 0.85"


def test_typo_and_alias_robustness(retriever):
    """Verifies that common UPSC student typos and abbreviations resolve correctly."""
    typo_queries = [
        ("Discuss guidelines in Bomai case regarding President Rule", "case_sr_bommai_1994"),
        ("Recommendations of Sarkariya commission on Governor tenure", "comm_sarkaria_1988"),
        ("Punchi commission recommendations on Governor impeachment", "comm_punchhi_2010"),
        ("Scope of privacy in Putaswamy ruling", "case_puttaswamy_2017"),
        ("Basic structure doctrine in Keshavananda case", "case_kesavananda_1973"),
    ]

    for query, expected_id in typo_queries:
        results = retriever.retrieve_hybrid(query, top_k=5)
        retrieved_ids = [r.item.id for r in results]
        assert expected_id in retrieved_ids, (
            f"Typo resolution failed for '{query}'. Expected {expected_id} in {retrieved_ids}"
        )


def test_hallucination_negative_test(retriever):
    """
    Distractor test: queries with no relevant constitutional items
    must not match arbitrary high-confidence items.
    """
    irrelevant_query = "Quantum mechanics wave particle duality and semiconductor doping fabrication"
    results = retriever.retrieve_hybrid(irrelevant_query, top_k=5)
    # The retriever should either return empty or very low scores without false exact guards
    for r in results:
        assert r.score < 1.0, f"Unexpected high score for irrelevant query: {r.score} on {r.item.id}"
