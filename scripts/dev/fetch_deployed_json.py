"""Fetches the latest evaluated/parsed copy from the deployed DynamoDB state."""
import json
from decimal import Decimal
from pathlib import Path
import boto3

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main():
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    table = ddb.Table("upsc-evaluator-job-state")
    
    print("[*] Scanning 'upsc-evaluator-job-state' in us-east-1...")
    resp = table.scan()
    items = resp.get("Items", [])
    print(f"[+] Found {len(items)} total jobs in DynamoDB.")
    
    if not items:
        print("[!] No jobs found in DynamoDB table.")
        return

    # Sort by updated_at descending
    items.sort(key=lambda x: str(x.get("updated_at", "")), reverse=True)
    
    for idx, item in enumerate(items[:5]):
        print(f"  [{idx+1}] Job ID: {item.get('job_id')} | Status: {item.get('status')} | Updated: {item.get('updated_at')}")

    latest_job = items[0]
    job_id = latest_job.get("job_id")
    print(f"\n[*] Latest Job: {job_id}")
    print(f"    Status: {latest_job.get('status')}")
    print(f"    Progress: {latest_job.get('progress_pct')}%")
    print(f"    Current Step: {latest_job.get('current_step')}")
    print(f"    Email: {latest_job.get('email')}")
    print(f"    Available keys: {list(latest_job.keys())}")

    # Check for parsed_document or result
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"latest_deployed_job_{job_id}.json"

    # Save complete raw record
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(latest_job, f, default=decimal_default, indent=2, ensure_ascii=False)
    
    print(f"[+] Saved full job record to: {out_file}")

    if "parsed_document" in latest_job:
        parsed_doc_file = output_dir / f"parsed_document_{job_id}.json"
        with open(parsed_doc_file, "w", encoding="utf-8") as f:
            json.dump(latest_job["parsed_document"], f, default=decimal_default, indent=2, ensure_ascii=False)
        print(f"[+] Saved parsed document to: {parsed_doc_file}")
    elif "result" in latest_job:
        print(f"[+] 'result' found with {len(latest_job['result'].get('questions', []))} questions evaluated.")
        questions = latest_job["result"].get("questions", [])
        for q in questions:
            print(f"    Q{q.get('q_num', '?')}: {q.get('score')} / {q.get('max_marks')} - {str(q.get('question'))[:60]}...")

if __name__ == "__main__":
    main()
