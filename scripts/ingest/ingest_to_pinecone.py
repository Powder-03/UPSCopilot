"""Ingest authentic UPSC knowledge base into Pinecone serverless index with Vertex AI embeddings."""
import argparse
import os
import sys
import time
from typing import Any

from rich.console import Console
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import settings
from src.kb.corpus_loader import load_all_corpus_documents
from src.kb.vector_store import VertexEmbeddings


def ensure_pinecone_index(pc, index_name: str, dimension: int = 768, cloud: str = "aws", region: str = "us-east-1"):
    """Ensures the Pinecone serverless index exists and is ready."""
    from pinecone import ServerlessSpec

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        console.print(f"[bold cyan]Creating Pinecone serverless index:[/bold cyan] '{index_name}' (dim={dimension}, metric=cosine, region={region})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        console.print("[dim]Waiting for index to become ready...[/dim]")
        while True:
            desc = pc.describe_index(index_name)
            if desc.status.get("ready"):
                break
            time.sleep(2)
        console.print(f"[bold green]Index '{index_name}' is ready![/bold green]\n")
    else:
        console.print(f"[bold green]Connected to existing Pinecone index:[/bold green] '{index_name}'\n")

    return pc.Index(index_name)


console = Console()


def main():
    parser = argparse.ArgumentParser(description="Ingest authentic UPSC Knowledge Base into Pinecone.")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of chunks per embedding & upsert batch.")
    parser.add_argument("--dry-run", action="store_true", help="Load documents and print stats without calling APIs.")
    args = parser.parse_args()

    console.print("\n[bold magenta]==========================================================[/bold magenta]")
    console.print("[bold cyan] UPSC Evaluator: Pinecone Knowledge Base Ingestion Pipeline [/bold cyan]")
    console.print("[bold magenta]==========================================================[/bold magenta]\n")

    # 1. Check API credentials
    api_key = settings.pinecone_api_key or os.getenv("PINECONE_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/bold red] PINECONE_API_KEY is not set in .env. Please add your Pinecone API key.")
        sys.exit(1)

    gemini_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY is not set in .env. Please add your Gemini API key.")
        sys.exit(1)

    # 2. Load authentic corpus documents
    console.print("[bold yellow]Loading authentic corpus documents from data/raw/ and data/documents/...[/bold yellow]")
    docs = load_all_corpus_documents()
    total_docs = len(docs)
    console.print(f"[bold green]Loaded {total_docs} authentic knowledge chunks.[/bold green]\n")

    # Document breakdown table
    source_counts: dict[str, int] = {}
    for doc in docs:
        src = doc.metadata.get("source", doc.metadata.get("doc_type", "General"))
        source_counts[src] = source_counts.get(src, 0) + 1

    table = Table(title="Corpus Composition")
    table.add_column("Source / Document Type", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    for src, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        table.add_row(src, str(count))
    console.print(table)
    console.print()

    if args.dry_run:
        console.print("[bold yellow]--dry-run enabled. Exiting without making API calls.[/bold yellow]")
        return

    # 3. Connect to Pinecone & ensure index
    from pinecone import Pinecone

    pc = Pinecone(api_key=api_key)
    index = ensure_pinecone_index(
        pc=pc,
        index_name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )

    # 4. Initialize Vertex AI Embeddings (text-embedding-004 = 768 dim)
    console.print(f"[bold cyan]Initializing Vertex AI text-embedding-004[/bold cyan] (dim={settings.embedding_dimension})...\n")
    embedder = VertexEmbeddings(
        dimension=settings.embedding_dimension,
        batch_size=args.batch_size,
    )

    # 5. Batch embed and upsert
    batch_size = args.batch_size
    num_batches = (total_docs + batch_size - 1) // batch_size
    start_time = time.time()

    console.print(f"[bold yellow]Starting ingestion of {total_docs} chunks in {num_batches} batches (batch size={batch_size})...[/bold yellow]\n")

    for b in range(num_batches):
        batch_start = b * batch_size
        batch_end = min(batch_start + batch_size, total_docs)
        chunk_batch = docs[batch_start:batch_end]

        # Fast resume: check if all IDs in this batch are already indexed in Pinecone
        batch_ids = [doc.metadata.get("id", f"chunk_{batch_start + i}") for i, doc in enumerate(chunk_batch)]
        try:
            fetched = index.fetch(ids=batch_ids)
            vectors_dict = fetched.vectors if hasattr(fetched, "vectors") else fetched.get("vectors", {})
            if len(vectors_dict) == len(batch_ids):
                console.print(
                    f" [dim]⏭  Batch [cyan]{b + 1:2d}/{num_batches:2d}[/cyan] "
                    f"({len(batch_ids):2d} chunks, indices {batch_start:4d}-{batch_end:4d}) "
                    f"already in Pinecone (Skipped)[/dim]"
                )
                continue
        except Exception:
            pass

        texts = [doc.page_content for doc in chunk_batch]

        b_t0 = time.time()
        # Embed batch via Vertex AI
        vectors = embedder.embed_documents(texts)

        # Prepare Pinecone records
        records: list[dict[str, Any]] = []
        for doc, vec in zip(chunk_batch, vectors, strict=False):
            doc_id = doc.metadata.get("id", f"chunk_{batch_start + len(records)}")
            meta = {
                "id": doc_id,
                "title": str(doc.metadata.get("title", "")),
                "source": str(doc.metadata.get("source", "")),
                "doc_type": str(doc.metadata.get("doc_type", "")),
                "gs_paper": str(doc.metadata.get("gs_paper", "")),
                "text": doc.page_content[:30000],  # Pinecone 40KB metadata limit safeguard
            }
            records.append({
                "id": doc_id,
                "values": vec,
                "metadata": meta,
            })

        # Upsert into Pinecone
        index.upsert(vectors=records)
        b_elapsed = time.time() - b_t0

        console.print(
            f" [bold green]✓[/bold green] Batch [cyan]{b + 1:2d}/{num_batches:2d}[/cyan] "
            f"({len(records):2d} chunks, indices {batch_start:4d}-{batch_end:4d}) "
            f"upserted in [dim]{b_elapsed:.2f}s[/dim]"
        )

    total_time = time.time() - start_time
    console.print(f"\n[bold green] Ingestion Complete![/bold green] Successfully indexed [bold cyan]{total_docs}[/bold cyan] chunks into Pinecone index [bold cyan]'{settings.pinecone_index_name}'[/bold cyan] in [bold green]{total_time:.1f}s[/bold green].\n")

    # 6. Verify index stats
    stats = index.describe_index_stats()
    console.print(f"[bold cyan]Index Statistics:[/bold cyan] Total Vectors: [bold green]{stats.total_vector_count}[/bold green], Dimension: [bold green]{stats.dimension}[/bold green]\n")


if __name__ == "__main__":
    main()
