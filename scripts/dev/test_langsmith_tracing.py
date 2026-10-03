"""Verification script for end-to-end LangSmith tracing in UPSCopilot."""
import os
import sys
import time

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath("."))

from src.config import settings
from src.evaluation.engine import UPSCEvaluationEngine
from src.utils.tracing import flush_traces, get_langsmith_client


def main() -> None:
    print("=" * 70)
    print("      UPSCopilot LangSmith Tracing Verification Probe")
    print("=" * 70)
    print(f"Tracing Enabled:      {settings.langsmith_tracing}")
    print(f"LangSmith Project:    {settings.langsmith_project}")
    print(f"LangSmith Endpoint:   {settings.langsmith_endpoint}")
    print(f"API Key Present:      {'Yes (masked)' if settings.langsmith_api_key else 'NO'}")
    print("-" * 70)

    # 1. Check LangSmith Client
    client = get_langsmith_client()
    if not client:
        print("[!] ERROR: Could not initialize LangSmith Client. Check LANGSMITH_API_KEY in .env.")
        sys.exit(1)

    print("[+] LangSmith Client initialized successfully.")

    # 2. Run a targeted single-question evaluation to trigger full waterfall trace
    sample_question = (
        "Discuss the significance of the 44th Constitutional Amendment Act, 1978 in "
        "safeguarding civil liberties and restoring constitutional balance post-Emergency."
    )
    sample_answer = """
The 44th Constitutional Amendment Act, 1978 enacted by the Janata Party government was a landmark watershed in Indian constitutionalism, primarily aimed at dismantling the distortions introduced by the 42nd Amendment and fortifying fundamental rights against executive arbitrariness.

1. Restoration of Civil Liberties and Fundamental Rights:
- Article 21 & Article 20: Explicitly made non-suspendable even during a National Emergency under Article 352, neutralizing the dark shadow of ADM Jabalpur (1976).
- Right to Property: Omitted from Part III (Article 19(1)(f) and Article 31) and relocated to Article 300A as a constitutional/legal right, preventing misuse while protecting socialist land reforms.
- Article 19: Safeguards for freedom of speech and press to report parliamentary proceedings without censorship (Article 361A).

2. Rationalization of Emergency Provisions (Article 352 & 356):
- Internal Disturbance: Replaced by 'Armed Rebellion', eliminating vague political pretexts.
- Cabinet Concurrence: Mandated written recommendation from the Union Cabinet, not merely unilateral Prime Ministerial advice.
- Approval Threshold: Raised Parliamentary approval to a special majority (2/3rd present and voting + majority of total membership) within one month.
- Periodic Review: Resolution for continuance required every six months.

3. Restoring Checks and Balances:
- Curtailed executive hegemony by strengthening judicial review over emergency proclamations.
- Re-established the independence of the higher judiciary after executive supersession controversies.

Way Forward & Conclusion:
The 44th Amendment reinforced the Basic Structure doctrine (Kesavananda Bharati, 1973) and established procedural due process (Maneka Gandhi, 1978). It stands as a vital democratic bulwark ensuring that executive supremacy never eclipses constitutional morality and citizen liberties.
""".strip()

    print("[*] Running live evaluation with LangSmith tracing enabled...")
    start_time = time.time()

    engine = UPSCEvaluationEngine()
    result = engine.evaluate_answer(
        question=sample_question,
        candidate_answer=sample_answer,
        max_marks=10.0,
    )

    elapsed = round(time.time() - start_time, 2)
    print(f"[+] Evaluation finished in {elapsed}s.")
    print(f"    - Score: {result.total_score} / {result.max_marks} ({result.percentage}%)")
    print(f"    - Performance Band: {result.performance_band.value}")
    print(f"    - Directive: {result.directive_detected.value if result.directive_detected else 'None'}")
    print(f"    - Pillars Scored: {len(result.pillars)}")

    # 3. Flush traces immediately
    print("[*] Flushing buffered traces to LangSmith...")
    flush_traces(timeout=5.0)
    print("[+] Traces flushed successfully.")

    # 4. Display LangSmith project dashboard link
    project_url = f"https://smith.langchain.com/o/projects/p/{settings.langsmith_project}"
    print("-" * 70)
    print("[SUCCESS] All spans traced successfully!")
    print("View your live traces on LangSmith Dashboard:")
    print(f"-> Project: {settings.langsmith_project}")
    print(f"-> Dashboard URL: {project_url}")
    print("=" * 70)


if __name__ == "__main__":
    main()
