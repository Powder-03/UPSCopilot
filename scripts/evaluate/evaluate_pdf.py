"""CLI tool for evaluating an answer booklet PDF end-to-end and producing a student scorecard."""
import argparse
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.models.api import JobStatus, StudentEvaluationReport
from src.orchestrator import UnifiedEvaluationPipeline
from src.services.email_service import EmailService

logger = logging.getLogger("evaluate_pdf")
console = Console()


def render_terminal_scorecard(report: StudentEvaluationReport) -> None:
    """Renders a beautiful summary table in the terminal."""
    console.print("\n[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]")
    console.print(f"[bold cyan]UPSC Mains Evaluation Report: {report.document}[/bold cyan]")
    console.print("[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]\n")

    summary = report.summary
    console.print(
        f"[bold]Total Score:[/bold] [bold green]{summary.total_score:g} / {summary.max_marks:g}[/bold green] "
        f"([bold yellow]{summary.percentage:.1f}%[/bold yellow])\n"
    )

    console.print("[bold green]🎯 Key Strengths Across Paper:[/bold green]")
    for s in summary.overall_feedback.key_strengths:
        console.print(f"  [green]•[/green] {s}")

    console.print("\n[bold yellow]🚀 Priority Areas to Improve:[/bold yellow]")
    for i in summary.overall_feedback.top_areas_to_improve:
        console.print(f"  [yellow]•[/yellow] {i}")

    # Question Breakdown Table
    table = Table(title="\nQuestion-by-Question Breakdown", show_header=True, header_style="bold magenta")
    table.add_column("Q#", justify="center", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("%", justify="center")
    table.add_column("Pros (Strengths)", style="green")
    table.add_column("What to Do Better", style="yellow")

    for q in report.questions:
        pros_str = "\n".join(f"• {p}" for p in q.pros) if q.pros else "[dim]None noted[/dim]"
        better_str = "\n".join(f"• {b}" for b in q.what_to_do_better) if q.what_to_do_better else "[dim]None noted[/dim]"
        score_color = "green" if q.percentage >= 50 else ("yellow" if q.percentage >= 35 else "red")

        table.add_row(
            f"Q{q.q_num:02d}",
            f"[{score_color}]{q.score:g} / {q.max_marks:g}[/{score_color}]",
            f"[{score_color}]{q.percentage:.1f}%[/{score_color}]",
            pros_str,
            better_str,
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a handwritten UPSC answer booklet PDF.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to answer booklet PDF")
    parser.add_argument("--start-page", type=int, default=None, help="1-indexed starting page (e.g. 3)")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to process")
    parser.add_argument("--workers", type=int, default=4, help="Parallel evaluation workers")
    parser.add_argument("--email", type=str, default=None, help="Email address to send scorecard to")
    parser.add_argument(
        "--output",
        type=str,
        default="tests/outputs/student_scorecard.json",
        help="Path to save output JSON",
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        console.print(f"[bold red]Error:[/bold red] PDF file '{pdf_path}' not found.")
        return

    console.print(f"[bold]Evaluating PDF:[/bold] {pdf_path.name}")
    if args.start_page:
        console.print(f"[dim]Starting from page {args.start_page}[/dim]")
    if args.max_pages:
        console.print(f"[dim]Max pages limited to {args.max_pages}[/dim]")

    pipeline = UnifiedEvaluationPipeline(workers=args.workers)

    def progress_callback(status: JobStatus, pct: int, step: str) -> None:
        console.print(f"[cyan][{pct:3d}%][/cyan] [{status.value}] {step}")

    report = pipeline.run_pipeline(
        pdf_path=pdf_path,
        start_page=args.start_page,
        max_pages=args.max_pages,
        progress_callback=progress_callback,
    )

    # Render summary table to terminal
    render_terminal_scorecard(report)

    # Save output JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
    console.print(f"\n[bold green]✓ Saved student scorecard JSON to:[/bold green] {output_path}")

    # Send Email if requested
    if args.email:
        console.print(f"[bold cyan]Dispatching scorecard email to {args.email}...[/bold cyan]")
        email_service = EmailService()
        success = email_service.send_evaluation_email(to_email=args.email, report=report)
        if success:
            console.print(f"[bold green]✓ Email successfully dispatched to {args.email}[/bold green]")
        else:
            console.print(f"[bold red]✗ Failed to send email to {args.email}[/bold red]")


if __name__ == "__main__":
    main()
