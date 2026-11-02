# UPSC Evaluation Rubrics & Empirical Benchmarks

[← Back to Main README](README.md)

This document details the **6-Pillar UPSC Evaluation Rubric**, the **2-Call Continuous Logprob Scoring Protocol**, and the **Empirical Multi-Run Benchmark Results** for **UPSCopilot**.

---

## 1. The 6-Pillar UPSC Marking Rubric

Every candidate answer is evaluated across six weighted pillars (`P1`–`P6`) aligned with authentic UPSC Civil Services Mains evaluation standards:

| Pillar | Name | Weight | Core Evaluation Criteria |
| :--- | :--- | :---: | :--- |
| **P1** | **Directive & Demand Fulfillment** | **25%** | Addresses all explicit sub-parts and obeys the specific UPSC directive (*Discuss*, *Examine*, *Critically Analyze*, *Evaluate*, *Elucidate*). |
| **P2** | **Structural Architecture & Presentation** | **15%** | Logical flow, sub-headings, legibility, and effective use of presentation archetypes (bullet points, flowcharts, diagrams, or comparative tables). |
| **P3** | **Multi-Dimensional Breadth (GS Governance)** | **20%** | Coverage across constitutional, administrative, socio-economic, ethical, technological, and federal dimensions. |
| **P4** | **Grounded Legal Citations & Accuracy** | **20%** | Accurate use of Constitutional Articles, statutory provisions, Supreme Court landmark rulings, and committee reports (2nd ARC, Law Commission). |
| **P5** | **Conclusion & Constructive Way Forward** | **15%** | Actionable, balanced, and forward-looking reforms rather than repetitive summaries. |
| **P6** | **Introduction & Context Setting** | **5%** | Concise definition, constitutional/statutory anchor, or relevant current affairs context within the opening lines. |

---

## 2. The 2-Call Evaluation & Calibration Protocol

### Call 1: Deep Diagnostic Chain-of-Thought (CoT)
1. **Micro-Demand Decomposition**: Splits the question into weighted micro-demands based on its directive word.
2. **Citation Fact-Checking**: Audits every legal, constitutional, and judicial reference against the Knowledge Base and classifies each as:
   - `VERIFIED_ACCURATE`: Confirmed accurate against ground-truth sources.
   - `VALID_OPEN_WORLD`: Accurate contemporary example or committee reference outside static statutes.
   - `MISATTRIBUTED`: Real Article/Case cited for the wrong legal principle.
   - `HALLUCINATED_INVENTED`: Fabricated Article, statute, or judgment (triggers strict P4 penalty).
3. **Presentation Archetype Analysis**: Evaluates structure fairly across paragraph-heavy, diagrammatic, and tabular styles.

### Call 2: Continuous Probability-Weighted G-Eval Scoring
Instead of prompting the LLM for an arbitrary integer score, Call 2 extracts the token log-probability distribution over discrete rating tokens $r \in \{1, 2, 3, 4, 5\}$ for each pillar:

1. **Expected Pillar Rating ($E[R]$)**:
   $$E[R] = \sum_{r=1}^{5} P(\text{token} = r) \cdot r$$

2. **Authentic UPSC Linear Calibration**:
   Maps the continuous rating $E[R] \in [1.0, 5.0]$ into realistic UPSC CSE Mains mark percentages:
   $$\text{UPSC Mark \%} = 0.12 + (E[R] \times 0.08)$$

3. **Safety Gates & Fast-Paths**:
   - **Unattempted / Blank Fast-Path**: Blank questions bypass LLM scoring and immediately receive `0.00` marks in `0.00ms`.
   - **Hard Demand Relevance Gatekeeper**: Off-topic essays that fail to address the core demand of the question are strictly capped at $\le 2.5\% - 25\%$ of maximum marks regardless of language fluency.

### Calibrated UPSC CSE Performance Bands

| Performance Band | Percentage Range | Marks (/ 250) | Real UPSC CSE Trajectory |
| :--- | :---: | :---: | :--- |
| **Topper Benchmark** | `48% – 55%+` | `120 – 138+` | All-India Rank 1–50 Trajectory |
| **Good / Selection Zone** | `41% – 47%` | `102 – 119` | All-India Rank 50–300 Trajectory |
| **Average** | `32% – 40%` | `80 – 101` | Mains Interview Cutoff Zone |
| **Needs Foundation** | `< 32%` | `< 80` | Below Cutoff / Needs Fundamental Value Add |

---

## 3. Empirical Benchmark Data & Multi-Run Consistency

UPSCopilot was evaluated across complete 20-question UPSC General Studies answer copies under repeated independent runs with zero cross-question CoT contamination.

### A. High-Quality / Selection-Trajectory Copy (5 Consecutive Runs)

#### Run-by-Run Total Score Stability

| Run # | Total Marks Awarded | Percentage |
| :--- | :--- | :--- |
| **Run 1** | **104.33** / 250.00 marks | **41.7%** |
| **Run 2** | **104.26** / 250.00 marks | **41.7%** |
| **Run 3** | **104.44** / 250.00 marks | **41.8%** |
| **Run 4** | **106.12** / 250.00 marks | **42.4%** |
| **Run 5** | **105.41** / 250.00 marks | **42.2%** |

#### Statistical Spread Metrics
- **Mean Copy Score**: `104.91 / 250.00 marks` (Authentic UPSC CSE Selection Zone)
- **Standard Deviation ($\sigma$)**: `0.82 marks`
- **Score Range ($\text{Max} - \text{Min}$)**: `1.86 marks` (Min: 104.26, Max: 106.12)
- **Spread as % of Total Marks**: `0.74%`
- **Coefficient of Variation ($CV$)**: `0.78%`
- **Consistency Verdict**: **Near-Deterministic Stability**

#### Per-Question Marking Comparison Sample (Q01–Q06)

| Q# | Max | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | StdDev ($\sigma$) | Range |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q01** | 10 | 5.00 | 5.20 | 5.14 | 5.00 | 5.20 | **5.11** | 0.10 | 0.20 |
| **Q02** | 10 | 4.15 | 4.51 | 4.15 | 4.60 | 4.56 | **4.39** | 0.23 | 0.45 |
| **Q03** | 10 | 3.55 | 3.55 | 3.78 | 4.21 | 4.20 | **3.86** | 0.33 | 0.66 |
| **Q04** | 10 | 3.75 | 4.30 | 4.60 | 4.30 | 4.60 | **4.31** | 0.35 | 0.85 |
| **Q05** | 10 | 4.35 | 4.40 | 4.57 | 4.25 | 4.80 | **4.47** | 0.22 | 0.55 |
| **Q06** | 10 | 4.48 | 4.40 | 4.90 | 4.18 | 4.56 | **4.50** | 0.26 | 0.72 |

---

### B. Poor-Quality & Stress-Test Copy (4 Consecutive Runs)

Stress-testing with gross factual hallucinations (e.g., citing Article 500 for tribunals), off-topic digressions (writing about Chandrayaan-3 for a Sevottam question), and unattempted blank questions:

#### Run-by-Run Total Score Stability

| Run # | Total Marks Awarded | Percentage |
| :--- | :--- | :--- |
| **Run 1** | **35.63** / 250.00 marks | **14.3%** |
| **Run 2** | **35.64** / 250.00 marks | **14.3%** |
| **Run 3** | **29.48** / 250.00 marks | **11.8%** |
| **Run 4** | **31.99** / 250.00 marks | **12.8%** |

#### Statistical Spread Metrics
- **Mean Copy Score**: `33.19 / 250.00 marks` (`13.3%`)
- **Standard Deviation ($\sigma$)**: `3.01 marks`
- **Score Range ($\text{Max} - \text{Min}$)**: `6.16 marks` (Min: 29.48, Max: 35.64)
- **Consistency Verdict**: **Strict Anti-Sycophantic Enforcement**

#### Per-Question Stress Breakdown Sample

| Q# | Max | Run 1 | Run 2 | Run 3 | Run 4 | Mean | StdDev ($\sigma$) | Evaluation Behavior |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q01** | 10 | 2.50 | 0.25 | 0.40 | 0.25 | **0.85** | 1.10 | Severe penalty for constitutional errors |
| **Q02** | 10 | 2.50 | 2.50 | 2.50 | 2.50 | **2.50** | 0.00 | Exact deterministic floor score |
| **Q03** | 10 | 2.67 | 2.50 | 2.50 | 2.50 | **2.54** | 0.08 | Floor stability |
| **Q04** | 10 | 2.55 | 2.55 | 2.80 | 3.30 | **2.80** | 0.35 | Marginal attempt scored strictly |
| **Q05** | 10 | 0.00 | 0.25 | 0.26 | 0.27 | **0.20** | 0.13 | **Off-Topic Gate Triggered** (Hard penalty cap) |
| **Q07** | 10 | 0.38 | 0.27 | 0.25 | 0.25 | **0.29** | 0.06 | Gatekeeper floor |
| **Q08** | 10 | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **Unattempted Fast-Path** (0.00ms, zero-mark lock) |

---

### C. 6-Pillar G-Eval Stability Across Runs (Whole-Copy Aggregates)

| Pillar Name | Weight | Mean Score | StdDev ($\sigma$) | Stability |
| :--- | :---: | :---: | :---: | :---: |
| **P1: Directive & Demand Fulfillment** | 25% | 14.23 marks | 0.25 | High |
| **P2: Structural Architecture & Presentation** | 15% | 9.42 marks | 0.18 | High |
| **P3: Multi-Dimensional Breadth (GS Governance)** | 20% | 11.34 marks | 0.11 | High |
| **P4: Grounded Legal Citations & Accuracy** | 20% | 11.35 marks | 0.11 | High |
| **P5: Conclusion & Constructive Way Forward** | 15% | 8.61 marks | 0.11 | High |
| **P6: Introduction & Context Setting** | 5% | 2.88 marks | 0.05 | High |

---

## 4. Observability & Automated Evaluation Suites

### End-to-End LangSmith Tracing
Every booklet evaluation is captured as a single hierarchical trace tree in **LangSmith**:

```
UPSC_Answer_Booklet_Evaluation (Root Trace)
├── DocumentParsing_OCR (Dynamic Page-by-Page Vision OCR & Stitching)
├── KB_Ground_Truth_Prefetch (Parallel Knowledge Base Prefetch)
├── EC2_FanOut_20_Lambda_Evaluation
│   └── LambdaSingleQuestionEval [Q1..Q20]
│       ├── _run_diagnostic (Call 1: Deep Diagnostic CoT & Citation Audit)
│       └── _apply_marking_policy (Call 2: G-Eval Continuous Logprob Scoring)
├── Student_Scorecard_Distillation
├── Scorecard_PDF_Generation
└── SMTP_Scorecard_Email_Delivery
```

> 🔗 **[View Live Production Evaluation Trace on LangSmith](https://smith.langchain.com/public/f553aa87-076d-4431-944a-cb8aecad7743/r/01a10363-900f-7be1-976d-17257b4deca1?start_time=2026-10-03T20%3A10%3A19.023878Z)**

### Reproducing Benchmarks via CLI

```bash
# 1. Run 72 offline unit & integration tests
uv run pytest -m "not bedrock"

# 2. Run multi-run score consistency benchmark
uv run python scripts/evaluate/topper_copy_evaluation.py \
  --copy tests/fixtures/gs2_copy_parsed.json \
  --runs 5 \
  --workers 4 \
  --report-out data/outputs/consistency_benchmark.json

# 3. Run DeepEval Ground-Truth Retrieval Benchmark (Contextual Recall = 1.00, Precision = 0.84)
uv run python scripts/evaluate/retrieval.py --top-k 8
```
