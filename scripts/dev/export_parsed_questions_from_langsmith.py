"""Extracts all parsed OCR questions from LangSmith and saves as complete parsed JSON."""
import json
from pathlib import Path
from src.config import settings
from langsmith import Client

def main():
    client = Client(api_key=settings.langsmith_api_key)
    print(f"[*] Querying LangSmith project '{settings.langsmith_project}' for OCR question spans...")
    
    # Query runs with name Process_Question_Pages
    runs = list(client.list_runs(
        project_name=settings.langsmith_project,
        execution_order=1,
        limit=100
    ))
    
    ocr_runs = [r for r in runs if r.name == "Process_Question_Pages" and r.outputs]
    print(f"[+] Found {len(ocr_runs)} OCR parsed question spans.")

    parsed_questions = []
    for r in ocr_runs:
        out = r.outputs
        # ParsedQuestion fields: q_num, question, candidate_answer, max_marks, is_blank, diagrams_found
        parsed_questions.append({
            "trace_id": str(r.id),
            "start_time": str(r.start_time),
            "latency": r.latency,
            "q_num": out.get("q_num"),
            "question": out.get("question"),
            "max_marks": out.get("max_marks"),
            "is_blank": out.get("is_blank", False),
            "candidate_answer": out.get("candidate_answer", ""),
            "diagrams_found": out.get("diagrams_found", []),
            "pages": out.get("pages", []),
            "word_count": len(out.get("candidate_answer", "").split())
        })

    # Sort canonically by question number
    parsed_questions.sort(key=lambda q: (q.get("q_num") or 999))

    out_file = Path("data/outputs/parsed_copy_vikas_air27.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(parsed_questions, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Successfully exported {len(parsed_questions)} questions to: {out_file}")
    for q in parsed_questions:
        print(f"  Q{q['q_num']}: [{q['word_count']} words] {str(q['question'])[:65]}...")

if __name__ == "__main__":
    main()
