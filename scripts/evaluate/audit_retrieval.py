"""Audit retrieval across all 15 items in golden_dataset.json against live Pinecone index."""
import json
from src.kb.retriever import HybridRetriever

with open("tests/fixtures/golden_dataset.json", encoding="utf-8") as f:
    data = json.load(f)

r = HybridRetriever(top_k=8)

print("=" * 80)
print("AUDITING PINECONE RETRIEVAL ACROSS ALL 15 GOLDEN DATASET QUESTIONS")
print("=" * 80)

for item in data:
    q = item["query"]
    docs = r.retrieve(q, top_k=8)
    retrieved_titles = [d.metadata.get("title", "") for d in docs]
    all_text = " ".join([d.page_content for d in docs]).lower()

    # Check key entities
    hits = []
    misses = []
    for e in item["key_entities"]:
        clean_e = e.split("(")[0].strip().lower()
        if clean_e in all_text or any(part.strip().lower() in all_text for part in e.split("/")):
            hits.append(e)
        else:
            misses.append(e)

    pct = len(hits) / len(item["key_entities"]) * 100
    print(f"\n[{item['id']}] {item['category']}")
    print(f"  Q: {q[:75]}...")
    print(f"  Entity Recall: {len(hits)}/{len(item['key_entities'])} ({pct:.0f}%)")
    print(f"  Top Retrieved Sources: {retrieved_titles[:4]}")
    if misses:
        print(f"  MISSING ENTITIES: {misses}")
