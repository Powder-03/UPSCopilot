"""CLI tool to ingest any scanned UPSC answer booklet PDF and export structured JSON for evaluation."""
import argparse
import sys
from pathlib import Path

from src.parsing.document_parser import DocumentParsingPipeline
from src.utils.cli import configure_console, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest and transcribe a scanned UPSC QCAB answer booklet PDF into evaluation-ready JSON."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the scanned answer booklet PDF file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="tests/outputs/parsed_copy.json",
        help="Path to save the resulting parsed JSON file (default: tests/outputs/parsed_copy.json).",
    )
    parser.add_argument(
        "--master",
        "-m",
        type=str,
        default=None,
        help="Optional path to master questions JSON for reconciliation and canonical ordering.",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of concurrent worker threads for parallel question transcription (default: 4).",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of questions to process (useful for dry runs / testing).",
    )
    parser.add_argument(
        "--max-pages",
        "-p",
        type=int,
        default=None,
        help="Limit number of pages to process from the PDF (e.g. 10 for testing).",
    )
    parser.add_argument(
        "--start-page",
        "-s",
        type=int,
        default=None,
        help="Optional 1-indexed starting page (default: auto-detects from page 1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Custom Bedrock multimodal vision model ID (overrides default in .env).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_console()
    setup_logging()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF file '{pdf_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("UPSC DOCUMENT PARSING & INGESTION PIPELINE")
    print(f"Input PDF:  {pdf_path.name}")
    print(f"Workers:    {args.workers}")
    if args.max_pages:
        print(f"Page Limit: {args.max_pages} pages")
    if args.limit:
        print(f"Question Limit: {args.limit}")
    print(f"Output:     {args.output}")
    print("=" * 80)

    pipeline = DocumentParsingPipeline(
        vision_model_id=args.model,
        max_workers=args.workers,
    )

    out_file = pipeline.parse_pdf_to_json_file(
        pdf_path=pdf_path,
        output_json_path=args.output,
        master_questions_path=args.master,
        limit_questions=args.limit,
        max_pages=args.max_pages,
        start_page=args.start_page,
    )

    print("\n" + "=" * 80)
    print(f"INGESTION COMPLETE: Saved evaluation-ready JSON to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
