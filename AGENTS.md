# AGENTS.md - Project Context & Agent Memory

## 1. Project Overview
- **Project**: `upsc-evaluator` (UPSCopilot)
- **Purpose**: Calibrated, grounded UPSC Civil Services Mains answer evaluation engine.
- **Repository**: `https://github.com/Powder-03/UPSCopilot.git`
- **Environment**: Python 3.11+ managed with `uv` (`uv run ...`)

---

## 2. Core Model & AWS Bedrock Architecture
- **Evaluation & Core LLM**: Moonshot Kimi 2.5 (`moonshotai.kimi-k2.5`)
- **Embeddings**: Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`, 1024-dim)
- **Gateway**: AWS Bedrock Converse API via `langchain_aws.ChatBedrockConverse`
- **Region**: `us-east-1` (configurable in `.env`)
- **Authentication Pattern**:
  - Uses standard `boto3.client("bedrock-runtime", region_name=settings.aws_region)` with AWS credentials.
  - **Important**: Do NOT pass `bedrock_api_key` directly to `ChatBedrockConverse` for third-party marketplace models like Kimi 2.5, as Bedrock rejects HTTP bearer tokens with `ValidationException: Operation not allowed`. Always pass the authenticated `client=boto3.client(...)`.
- **Model Factory**: `src.evaluation.model_factory.get_eval_llm()`
- **Test Script**: `scripts/dev/test_llm.py` (`uv run python scripts/dev/test_llm.py` verified working)

---

## 3. Real Raw Knowledge Base Ingestion Architecture (Zero Hardcoding)
- **Strictly No Hardcoded Python Arrays / Snippets**: All hardcoded python arrays have been deleted.
- **Data lives outside the importable package**: raw sources in `data/raw/`, documents in `data/documents/`, Chroma store in `data/storage/chroma_db/` (paths configurable via `DATA_DIR` / `KB_STORAGE_DIR` in `.env`, defaults in `src/config.py`).
- **Primary Raw Sources**:
  1. **Complete Constitution**: All 465 authentic articles + Preamble loaded from `data/raw/constitution_raw.json` by `src/kb/corpus_loader.py`.
  2. **Official Legislative Central Acts Downloader**: `scripts/ingest/download_acts.py` downloads official government gazette/statute PDFs (RTI Act 2005, DPDP Act 2023, RPA 1951, Lokpal Act 2013, CVC Act 2003, PMLA 2002, Disaster Management Act 2005) into `data/documents/`.
  3. **Master Ingestion Pipeline**: `scripts/ingest/all.py` batches every document in `data/documents/` plus the constitution JSON into the local Chroma vector store (`data/storage/chroma_db`) with Bedrock Titan Embeddings (PyMuPDF + `RecursiveCharacterTextSplitter`, `chunk_size=1000`, `chunk_overlap=200`).
  4. **Hybrid Retriever**: `src/kb/retriever.py` provides BM25 + dense Chroma retrieval with Reciprocal Rank Fusion (`RRF_K=60`) and FlashRank cross-encoder reranking over the real ingested documents.

---

## 4. Repository Layout
```
evaluator/
├── .env / .env.example   # Local secrets & paths (DATA_DIR, KB_STORAGE_DIR, AWS, model IDs)
├── AGENTS.md             # This file: agent context, memory & rules
├── pyproject.toml        # Dependencies, pytest config (bedrock marker), ruff config, dev group
├── conftest.py           # Repo-wide pytest fixtures + --run-bedrock opt-in for live tests
├── .pre-commit-config.yaml # Hygiene hooks + ruff --fix
├── .github/workflows/ci.yml # CI: ruff check + offline pytest on every push/PR
├── data/                 # NOT importable; raw corpus + vector store (gitignored artifacts)
│   ├── raw/constitution_raw.json       # 465 authentic articles
│   ├── documents/                      # Official Act PDFs, sc_cases/, central_acts/
│   └── storage/chroma_db/              # Persisted Chroma vector store
├── scripts/
│   ├── dev/              # test_llm.py, check_kimi_logprobs.py (developer probes)
│   ├── ingest/           # all.py (master pipeline), download_acts.py (govt PDFs)
│   └── evaluate/         # answer_copy.py (fixture-driven copy CLI), sample.py, retrieval.py
├── src/
│   ├── config.py         # Settings loading from .env (pydantic-settings)
│   ├── evaluation/
│   │   ├── engine.py         # 2-call orchestrator + marking-policy constants
│   │   ├── geval_scorer.py   # G-Eval logprob continuous scoring
│   │   ├── model_factory.py  # Factory returning ChatBedrockConverse (Kimi 2.5)
│   │   └── prompt_templates.py
│   ├── kb/
│   │   ├── corpus_loader.py  # Constitution JSON + document loaders (single source of truth)
│   │   ├── retriever.py      # Hybrid retriever (BM25 + Chroma RRF + FlashRank)
│   │   └── vector_store.py   # Chroma DB with Bedrock Titan Embeddings
│   ├── models/           # ALL schemas live here (single ownership)
│   │   ├── enums.py          # UPSC performance bands, directives, pillars
│   │   ├── evaluation.py     # EvaluationResult, PillarGEvalScore, citation audit models
│   │   └── kb.py             # RetrievalEvaluationItem
│   └── utils/
│       ├── json.py           # Shared LLM-output JSON extraction (extract_json_dict, clean_json_text)
│       └── cli.py            # Shared console/logging setup for scripts
└── tests/
    ├── fixtures/         # golden_dataset.json, topper_copy.json, sample_answers.json, retrieval_benchmark.json
    ├── outputs/          # Evaluation run artifacts (gitignored)
    ├── test_evaluation_engine.py # Offline unit tests (13)
    └── test_kb_retrieval.py      # Live retrieval suite, marked `bedrock` (6)
```

---

## 5. Agent Working Rules (Strict)
1. **No Code Bloat or Hardcoded Snippets**: Never hardcode snippet dictionaries in Python files. Always work with authentic raw files (JSON, PDF, TXT, MD) under `data/`.
2. **Context Persistence**: Keep `AGENTS.md` updated whenever changes are made.
3. **Execution Environment**: Always run commands using `uv run python ...` within the workspace.
4. **Thin Scripts, Fat Library**: Scripts under `scripts/` are small CLIs; reusable logic belongs in `src/` (e.g. `src/utils/`). Never name a script after a stdlib module (`copy.py`, `json.py`, ...) — it shadows the stdlib for every sibling script.
5. **Schemas Live in `src/models/`**: Do not create new `schema.py` files inside `kb/` or `evaluation/`.
6. **Lint Before Commit**: `uv run ruff check .` must pass; pre-commit hooks run ruff automatically.
7. **Live-Credential Tests**: Retrieval tests are marked `bedrock` and skipped by default. Run them with `uv run pytest --run-bedrock`. CI only runs the offline suite.

---

## 6. Current Progress & Status
- [x] Configured AWS Bedrock with Moonshot Kimi 2.5 (`moonshotai.kimi-k2.5`).
- [x] Verified inferencing with `scripts/dev/test_llm.py` (`"I'm active and ready to help."`).
- [x] Verified Bedrock Titan Embeddings (`amazon.titan-embed-text-v2:0`).
- [x] Deleted all hardcoded snippet files (`ingest_central_acts.py`, `ingest_sc_cases.py`, mock JSONs).
- [x] Built real raw ingestion pipeline with zero hardcoding (`scripts/ingest/download_acts.py`, `scripts/ingest/all.py`, `src/kb/corpus_loader.py`).
- [x] Populated all 14 authentic Supreme Court landmark case dossiers in `data/documents/sc_cases/`.
- [x] Populated comprehensive statutory dossiers for all 7 key Central Acts in `data/documents/central_acts/` (RTI 2005, DPDP 2023, RPA 1951, Lokpal 2013, CVC 2003, PMLA 2002, Disaster Management 2005).
- [x] Created 15-item Golden Dataset (`tests/fixtures/golden_dataset.json`) with canonical Ideal Answers spanning the GS-2 syllabus.
- [x] Configured chunking parameters to `chunk_size=1000` and `chunk_overlap=200` in `src/kb/corpus_loader.py`.
- [x] Integrated FlashRank cross-encoder reranker (`flashrank`) into `src/kb/retriever.py` with expanded candidate pool (top 40) and relaxed source diversity ceiling (`max_per_source=5`) to ensure complete statutory and landmark case coverage without premature truncation.
- [x] Refined `case_puttaswamy_2017.md` with explicit 4-fold proportionality test and decisional autonomy facets.
- [x] Implemented Core UPSC Answer Evaluation Engine (`src/evaluation/engine.py`) with 2-call architecture:
  - Deep CoT Diagnostic Analysis (`prompt_templates.py`)
  - G-Eval continuous probability-weighted scoring using native Moonshot Kimi 2.5 logprobs (`geval_scorer.py`)
  - 6 canonical pillars (P1-P6): Demand Fulfillment 25%, Structure & Presentation 15%, Multi-Dimensional Breadth 20%, Grounded Legal Citations 20%, Conclusion & Way Forward 15%, Introduction & Context Setting 5%
  - Multi-archetype presentation handling (paragraphs vs diagrams vs tables)
  - Dual Grounding (mandatory KB anchors vs valid open-world insights)
  - Hard Demand Relevance Gatekeeper against off-topic essays
- [x] Created evaluation demo script (`scripts/evaluate/sample.py`) and unit tests (`tests/test_evaluation_engine.py`).
- [x] Full structural refactor (2026-09): dead code purged (shim `kb/schema.py`, broken `kb/embeddings/`, orphan `kb/ingestion/chunker.py`, empty `intelligence/`, duplicate ingest scripts); schemas consolidated into `src/models/` (`evaluation.py`, `kb.py`, `enums.py`); data moved out of the package to root `data/` (Chroma store relocated intact, 769 docs, no re-ingestion); scripts regrouped into `dev/`, `ingest/`, `evaluate/` with the copy-evaluation trio merged into fixture-driven `scripts/evaluate/answer_copy.py`; shared helpers extracted to `src/utils/` (`json.py`, `cli.py`); magic numbers promoted to named constants in `engine.py` (`TOPPER_CEILING_PCT`, `MIN_SCORE_FLOOR`, `OFF_TOPIC_*`, ...) and `retriever.py` (`RRF_K`, `CHANNEL_DEPTH`, `RERANK_POOL_*`, `MAX_CHUNKS_PER_SOURCE`); `evaluate_answer` decomposed into `_retrieve_ground_truth` / `_run_diagnostic` / `_apply_marking_policy`; hardcoded benchmark arrays moved to `tests/fixtures/retrieval_benchmark.json`; stale `test_c2_kb_retrieval.py` replaced by API-correct `test_kb_retrieval.py`; tooling added (ruff config + 182 lint findings fixed, GitHub Actions CI, pre-commit, root `conftest.py` with `--run-bedrock` opt-in). Verified: `ruff check .` clean, 13 offline tests pass, all module imports and script CLIs smoke-tested.
- [ ] Active: Ready for User Acceptance & Testing.
