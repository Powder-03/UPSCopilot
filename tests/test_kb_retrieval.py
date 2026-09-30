"""Component-level RAG retrieval evaluation suite.

Asserts against the live hybrid retriever:
1. Corpus load integrity (constitutional articles + case/act dossiers indexed)
2. Exact statutory guard (explicit Article numbers hit Rank 1)
3. Exact case-law guard (landmark case names hit Rank 1)
4. Benchmark Recall@5 >= 90% and MRR >= 0.85 over corpus-coverable expectations
5. Typo and alias robustness
6. Negative distractor defense (off-domain queries must not trigger the lexical guard)

Two adaptations to the current corpus:
- External dossiers are indexed as chunks, so their IDs carry a suffix
  (``case_sr_bommai_1994_chunk_12``); expected IDs are therefore matched by prefix.
- Expectations pointing at sources absent from the corpus (e.g. commission dossiers,
  schedule entries) are reported as a DATA GAP and excluded from scoring, because they
  measure missing source documents rather than retrieval quality.

Requires live AWS Bedrock credentials (the dense channel embeds every query), so the
module is marked ``bedrock`` and excluded from CI via ``-m "not bedrock"``.
"""
import json
import re
from pathlib import Path

import pytest
from src.kb.retriever import HybridRetriever, tokenize
from src.models.kb import RetrievalEvaluationItem

BENCHMARK_FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_benchmark.json"
IRRELEVANT_QUERY = "Quantum mechanics wave particle duality and semiconductor doping fabrication"

_CHUNK_SUFFIX_RE = re.compile(r"_chunk_\d+$")

pytestmark = pytest.mark.bedrock


def _load_benchmark() -> list[RetrievalEvaluationItem]:
    """Loads the golden retrieval benchmark from its JSON fixture."""
    with open(BENCHMARK_FIXTURE, encoding="utf-8") as f:
        return [RetrievalEvaluationItem(**item) for item in json.load(f)]


def _corpus_ids(retriever: HybridRetriever) -> set[str]:
    """Every chunk ID plus its parent source ID (chunk suffix stripped)."""
    ids: set[str] = set()
    for doc in retriever.documents:
        doc_id = doc.metadata.get("id")
        if not doc_id:
            continue
        ids.add(doc_id)
        ids.add(_CHUNK_SUFFIX_RE.sub("", doc_id))
    return ids


def _matches(expected_id: str, retrieved_ids: list[str]) -> bool:
    """True when any retrieved chunk belongs to (or is) the expected source document."""
    return any(rid == expected_id or _CHUNK_SUFFIX_RE.sub("", rid) == expected_id for rid in retrieved_ids)


def _rank_of(expected_id: str, retrieved_ids: list[str]) -> int:
    """1-based rank of the first chunk matching `expected_id`, or 0 when absent."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if _matches(expected_id, [rid]):
            return rank
    return 0


def _retrieve_ids(retriever: HybridRetriever, query: str, top_k: int) -> list[str]:
    return [doc.metadata.get("id", "") for doc in retriever.retrieve(query, top_k=top_k)]


def _coverable(queries: list[tuple[str, str]], corpus_ids: set[str], label: str) -> list[tuple[str, str]]:
    """Filters query/expected-ID pairs to those the corpus can satisfy, printing the gap."""
    available = [(q, e) for q, e in queries if e in corpus_ids]
    missing = [e for _, e in queries if e not in corpus_ids]
    if missing:
        print(f"\n[DATA GAP] {label}: expected IDs absent from corpus -> {missing}")
    return available


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    """Initializes the hybrid retriever once per module (corpus load + Chroma open)."""
    return HybridRetriever()


@pytest.fixture(scope="module")
def corpus_ids(retriever: HybridRetriever) -> set[str]:
    return _corpus_ids(retriever)


def test_corpus_loaded(retriever, corpus_ids):
    """Verifies constitutional articles and landmark case dossiers are indexed."""
    assert len(retriever.documents) >= 30, f"Expected >= 30 chunks, found {len(retriever.documents)}"
    assert "art_163" in corpus_ids
    assert "case_sr_bommai_1994" in corpus_ids
    assert retriever.bm25 is not None


def test_exact_statutory_guard_articles(retriever, corpus_ids):
    """Queries naming an explicit Article must return that Article at Rank 1."""
    statutory_queries = [
        ("What are the powers under Article 163?", "art_163"),
        ("Examine the abuse of Article 356 in state governance", "art_356"),
        ("Discuss assent to bills under Article 200", "art_200"),
        ("Scope of Right to Life under Article 21", "art_21"),
        ("Significance of Article 32 as heart and soul of Constitution", "art_32"),
        ("Provisions of Article 239AA regarding governance of Delhi", "art_239aa"),
        ("Disqualification under Tenth Schedule anti-defection", "sched_10"),
    ]

    available = _coverable(statutory_queries, corpus_ids, "statutory guard")
    assert available, "No statutory guard query is coverable by the current corpus"

    for query, expected_id in available:
        retrieved = _retrieve_ids(retriever, query, top_k=3)
        assert retrieved, f"No results for query: {query}"
        assert _rank_of(expected_id, retrieved) == 1, (
            f"Exact statutory guard failed for '{query}'. "
            f"Expected {expected_id} at Rank 1, got {retrieved}."
        )


def test_exact_case_law_guard(retriever, corpus_ids):
    """Queries naming a landmark case must place that case dossier at Rank 1."""
    case_queries = [
        ("Evaluate the guidelines laid down in S.R. Bommai case", "case_sr_bommai_1994"),
        ("Evolution of Basic Structure in Kesavananda Bharati", "case_kesavananda_1973"),
        ("Right to Privacy principles in Puttaswamy judgment", "case_puttaswamy_2017"),
        ("Governor summoning assembly in Nabam Rebia case", "case_nabam_rebia_2016"),
        ("Harmonious balance between FR and DPSP in Minerva Mills", "case_minerva_mills_1980"),
        ("Election Commission appointment in Anoop Baranwal case", "case_anoop_baranwal_2023"),
    ]

    available = _coverable(case_queries, corpus_ids, "case law guard")
    assert available, "No case law guard query is coverable by the current corpus"

    for query, expected_id in available:
        retrieved = _retrieve_ids(retriever, query, top_k=3)
        assert retrieved, f"No results for query: {query}"
        assert _rank_of(expected_id, retrieved) == 1, (
            f"Case law guard failed for '{query}'. Expected {expected_id} at Rank 1, got {retrieved}."
        )


def test_recall_at_5_and_mrr_golden_benchmark(retriever, corpus_ids):
    """Recall@5 >= 90% and MRR >= 0.85 over benchmark expectations present in the corpus."""
    benchmark = _load_benchmark()

    uncovered_queries = [
        item.query_id for item in benchmark if not any(eid in corpus_ids for eid in item.expected_ids)
    ]
    missing_ids: set[str] = set()
    total_expected = 0
    total_matched = 0
    reciprocal_ranks: list[float] = []

    for item in benchmark:
        coverable = [eid for eid in item.expected_ids if eid in corpus_ids]
        missing_ids.update(eid for eid in item.expected_ids if eid not in corpus_ids)
        if not coverable:
            continue

        retrieved = _retrieve_ids(retriever, item.question_text, top_k=5)
        total_expected += len(coverable)
        total_matched += sum(1 for eid in coverable if _matches(eid, retrieved))

        if item.primary_id in corpus_ids:
            primary_rank = _rank_of(item.primary_id, retrieved)
            reciprocal_ranks.append(1.0 / primary_rank if primary_rank else 0.0)

    if missing_ids:
        print(f"\n[DATA GAP] Expected IDs absent from corpus (excluded): {sorted(missing_ids)}")
    if uncovered_queries:
        print(f"[DATA GAP] Benchmark queries with zero corpus coverage (skipped): {uncovered_queries}")

    recall_at_5 = total_matched / total_expected if total_expected else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    print(f"[RAG Benchmark] Recall@5: {recall_at_5:.2%}, MRR: {mrr:.4f} over {len(reciprocal_ranks)} queries")

    assert total_expected > 0, "No benchmark expectation is coverable by the current corpus"
    assert recall_at_5 >= 0.90, f"Recall@5 threshold failed: {recall_at_5:.2%} < 90%"
    assert mrr >= 0.85, f"MRR threshold failed: {mrr:.4f} < 0.85"


def test_typo_and_alias_robustness(retriever, corpus_ids):
    """Common UPSC student typos and abbreviations must still resolve to the right source."""
    typo_queries = [
        ("Discuss guidelines in Bomai case regarding President Rule", "case_sr_bommai_1994"),
        ("Recommendations of Sarkariya commission on Governor tenure", "comm_sarkaria_1988"),
        ("Punchi commission recommendations on Governor impeachment", "comm_punchhi_2010"),
        ("Scope of privacy in Putaswamy ruling", "case_puttaswamy_2017"),
        ("Basic structure doctrine in Keshavananda case", "case_kesavananda_1973"),
    ]

    available = _coverable(typo_queries, corpus_ids, "typo robustness")
    assert available, "No typo robustness query is coverable by the current corpus"

    for query, expected_id in available:
        retrieved = _retrieve_ids(retriever, query, top_k=5)
        assert _matches(expected_id, retrieved), (
            f"Typo resolution failed for '{query}'. Expected {expected_id} in {retrieved}"
        )


def test_negative_distractor_does_not_trigger_lexical_guard(retriever):
    """An off-domain query must score zero on the sparse BM25 channel (no false statutory guard).

    The dense channel always returns its k nearest neighbours, so the meaningful
    hallucination guard is that lexical retrieval refuses to fire on unrelated text.
    """
    scores = retriever.bm25.get_scores(tokenize(IRRELEVANT_QUERY))
    assert float(scores.max()) == 0.0, (
        f"Lexical guard fired for an irrelevant query: max BM25 score {float(scores.max()):.4f}"
    )
