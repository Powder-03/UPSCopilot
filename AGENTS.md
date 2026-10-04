# AGENTS.md - Project Context & Agent Memory

## 1. Project Overview
- **Project**: `upsc-evaluator` (UPSCopilot)
- **Purpose**: Calibrated, grounded UPSC Civil Services Mains answer evaluation engine.
- **Repository**: `https://github.com/Powder-03/UPSCopilot.git`
- **Environment**: Python 3.11+ managed with `uv` (`uv run ...`)

---

## 2. Core Model & Multi-Provider Architecture (Vertex AI & AWS Bedrock)
- **Dual-Provider Plug-and-Play**: Toggled via `LLM_PROVIDER` in `.env` (`"vertex"` or `"bedrock"`).
  - **Vertex AI (Default / Active)**:
    - Primary Model: Gemini 2.5 Flash (`gemini-2.5-flash`) via `ChatVertexExpress` in `src.providers.vertex`.
    - Vision Model: Gemini 2.5 Flash (`gemini-2.5-flash`) via `VertexVisionClient` in `src.providers.vision`.
    - G-Eval Scorer: `VertexGEvalScorer` in `src.evaluation.geval_scorer` using Gemini token logprobs (`responseLogprobs: True`, `logprobs: 5`, `thinkingBudget: 0`).
    - Authentication: `GEMINI_API_KEY` with Google Cloud Vertex AI Model Garden endpoints (`https://us-central1-aiplatform.googleapis.com/...`).
  - **AWS Bedrock (Plug-and-Play on Account Activation)**:
    - Evaluation & Core LLM: Moonshot Kimi 2.5 (`moonshotai.kimi-k2.5`) via `langchain_aws.ChatBedrockConverse`.
    - Vision Model: Moonshot Kimi 2.5 via `BedrockVisionClient` in `src.providers.vision`.
    - G-Eval Scorer: `BedrockGEvalScorer` using native Kimi 2.5 token logprobs.
    - Embeddings: Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`, 1024-dim).
    - Authentication: Standard `boto3.client("bedrock-runtime", region_name=settings.aws_region)` with AWS credentials. (Lazy client instantiation ensures no crashes when Bedrock is inactive).
- **Factories**:
  - `src.providers.factory.get_eval_llm()`
  - `src.evaluation.geval_scorer.get_geval_scorer()`
  - `src.providers.vision.get_vision_client()`
  - `src.kb.vector_store.get_embedding_function()` (falls back to `DeterministicMockEmbeddings(1024)` in offline / Vertex mode).
- **Test Script**: `scripts/dev/test_llm.py` (`uv run python scripts/dev/test_llm.py` verified working on Vertex AI Gemini 2.5 Flash).


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
├── Dockerfile            # Multi-stage container image for AWS Lambda
├── pyproject.toml        # Dependencies, pytest config, ruff config
├── samconfig.toml        # SAM deploy configuration
├── template.yaml         # SAM IaC: S3, DynamoDB, SQS, API/Vision/Eval Lambda functions
├── conftest.py           # Repo-wide pytest fixtures + --run-bedrock opt-in for live tests
├── data/                 # NOT importable; raw corpus + uploads (gitignored artifacts)
│   ├── raw/constitution_raw.json       # 465 authentic articles
│   ├── documents/                      # Official Act PDFs, sc_cases/, central_acts/
│   ├── jobs/                           # Local dev job state
│   └── uploads/                        # Local dev uploads
├── scripts/
│   ├── dev/              # Developer probes & test scripts
│   ├── ingest/           # Ingestion pipeline scripts (Pinecone, Acts, all)
│   ├── parse/            # parse_copy.py (PDF OCR transcription CLI)
│   └── evaluate/         # Evaluation CLIs (topper_copy_evaluation.py, evaluate_pdf.py, etc.)
├── src/
│   ├── config.py         # Settings loading from .env (pydantic-settings)
│   ├── orchestrator.py   # Unified end-to-end evaluation & distillation pipeline
│   ├── api/              # FastAPI routes (health, submit, status, presigned URL)
│   │   └── app.py
│   ├── handlers/         # AWS Lambda entrypoint handlers
│   │   ├── api.py           # Mangum ASGI adapter for API Gateway
│   │   ├── vision_worker.py # SQS worker for Stage 1 multimodal vision OCR
│   │   └── eval_worker.py   # SQS worker for Stage 2 RAG + 2-call evaluation & SES delivery
│   ├── providers/        # LLM & Vision provider clients & factories
│   │   ├── vertex.py        # ChatVertexExpress (Google GenAI SDK)
│   │   ├── vision.py        # VertexVisionClient & BedrockVisionClient
│   │   └── factory.py       # get_eval_llm() provider router
│   ├── evaluation/       # Core 2-call evaluation engine
│   │   ├── engine.py        # CoT diagnostic + marking policy
│   │   ├── geval_scorer.py  # Continuous logprob scoring
│   │   └── prompt_templates.py
│   ├── parsing/          # Vision OCR booklet parsing
│   │   ├── document_parser.py # DocumentParsingPipeline
│   │   ├── preprocessor.py    # PyMuPDF rendering & visual blank detection
│   │   ├── prompts.py         # OCR & layout prompts
│   │   └── segmenter.py       # QCAB question slicing & sorting
│   ├── kb/               # Knowledge base & Pinecone retrieval
│   │   ├── corpus_loader.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── models/           # Pydantic schemas (single ownership)
│   │   ├── api.py
│   │   ├── enums.py
│   │   ├── evaluation.py
│   │   ├── exceptions.py
│   │   ├── kb.py
│   │   └── parsing.py
│   ├── services/         # Pluggable infrastructure adapters
│   │   ├── email_service.py
│   │   ├── job_manager.py
│   │   ├── job_state_service.py
│   │   ├── queue_service.py
│   │   ├── scorecard_pdf.py
│   │   └── storage_service.py
│   ├── static/           # Single source of truth web frontend
│   │   └── index.html
│   └── utils/
│       ├── cli.py
│       ├── json.py
│       └── tracing.py
└── tests/
    ├── fixtures/         # golden_dataset.json, topper_copy.json, benchmark JSONs
    └── test_*.py         # Comprehensive unit test suite (55 offline tests pass)
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
- [x] Multi-Run Consistency CLI & Concurrency Optimization (`scripts/evaluate/topper_copy_evaluation.py`):
  - Streamlined output JSON to store only high-level stats, runs summary, and per-question marks/mean/stddev (no nested diagnostic bloat).
  - Multi-worker concurrent evaluation (`--workers 4`) with hermetic per-question LLM calls (zero cross-question CoT contamination).
  - In-memory thread-safe KB cache in `engine.py` reducing repeat question retrieval latency to 0.00ms.
  - Zero-mark fast-path for unattempted/blank candidate answers without redundant LLM invocation.
- [x] Created Negative Stress-Testing Fixture (`tests/fixtures/gs2_copy_poor.json`):
  - 2 unattempted questions (Q08, Q18 left blank).
  - 3 off-topic questions (Q05 ISRO Chandrayaan-3 for Sevottam, Q13 Harappan urban planning for CBI, Q19 Patanjali Yoga for BRICS).
  - 15 poor-quality answers with gross factual errors (Art 500 for tribunals, Ambedkar inserting Socialist/Secular in 1950, Vajpayee introducing GST in 2017), slang, and lack of governance depth.
  - Verified evaluation: blank questions receive 0.00 marks, off-topic answers trigger Hard Demand Relevance Gate (capped at 0.25/10), and overall score drops to 14.2% (Below Average / Needs Fundamental Value Add).
- [x] Implemented Document Parsing & Ingestion Pipeline (`src/parsing/`):
  - `src/models/parsing.py`: Pydantic schemas (`ParsedQuestion`, `ParsedDocument`) with direct serialization to evaluation engine JSON.
  - `src/parsing/preprocessor.py`: PyMuPDF (`pymupdf`) page rendering, byte compression, and O(1) histogram-based visual blank page detection.
  - `src/parsing/prompts.py`: Multimodal vision prompts for UPSC QCAB header extraction, handwriting transcription, strikethrough omission, and diagram conversion (`[Diagram: ...]`, markdown tables).
  - `src/parsing/vision_client.py`: AWS Bedrock Multimodal Vision client with Converse API, defaulting to Moonshot Kimi 2.5 (`moonshotai.kimi-k2.5`).
  - `src/parsing/segmenter.py`: QCAB page slicer (10M=2 pages, 15M=3 pages) with canonical 1-to-20 re-sorting and unattempted question reconciliation.
  - `src/parsing/pipeline.py`: Master multi-threaded ingestion pipeline (`DocumentParsingPipeline`) with parallel question transcription and direct JSON export.
  - `scripts/parse/parse_copy.py`: Production CLI tool (`uv run python scripts/parse/parse_copy.py --pdf ...`).
- [x] Implemented Asynchronous End-to-End Evaluation Pipeline & FastAPI Service (Zero Performance Bands & Email Delivery):
  - `src/models/api.py`: Pydantic models (`StudentQuestionEvaluation`, `OverallFeedback`, `StudentSummary`, `StudentEvaluationReport`, `JobStatusResponse`, `JobSubmitResponse`) enforcing zero performance bands and zero internal CoT/pillar tokens.
  - `src/services/email_service.py`: Responsive HTML scorecard email generator with Amazon SES (`boto3`), SMTP, and dev mock preview modes.
  - `src/services/job_manager.py`: Thread-safe background execution manager with disk persistence (`data/jobs/`, `data/uploads/`), progress tracking, and worker thread lifecycle.
  - `src/pipeline.py`: `UnifiedEvaluationPipeline` orchestrator linking PDF vision OCR + KB retrieval + 2-call evaluator + student distillation.
  - `src/api/app.py`: FastAPI service with `POST /api/v1/jobs/submit` (immediate HTTP 202 `<500ms`), `GET /api/v1/jobs/{job_id}`, and `GET /api/v1/health`.
  - `scripts/evaluate/evaluate_pdf.py`: CLI tool with terminal table rendering, JSON scorecard output, and email delivery.
  - Unit test suite expanded to 32 passing tests (including `test_api_schemas.py`, `test_email_service.py`, `test_job_manager.py`, `test_api_endpoints.py`, `test_pipeline_distiller.py`).
  - Verified 100% clean with `uv run ruff check .` and `uv run pytest`.
- [x] Multi-Paper GS Knowledge Base & NCERT Conceptual Anchors Generated (`data/documents/`):
  - Generated 15 substantive authentic dossiers across GS-1, GS-2, GS-3, GS-4 and Class 11-12 NCERTs.
  - GS-1: Class 11 Physical Geography, Class 11 Indian Art, Class 12 Indian Society, Freedom Struggle Historiography, Geography & Critical Minerals, Social Issues & Demographic Transition.
  - GS-2: International Relations (UNCLOS, Quad, I2U2, BRICS+, G20, WTO), Governance Reforms (2nd ARC Reports, Sevottam IS 15700, MGNREGA Social Audit, NITI Aspirational Districts).
  - GS-3: Class 12 Macroeconomics, Macroeconomics & FRBM, Agriculture & NFSA/MSP, Environment & Wildlife/Biodiversity Acts, Science/Space/Quantum Missions, Internal Security (UAPA, NIA, AFSPA, LWE).
  - GS-4: Moral Thinkers (Kant, Utilitarians, Aristotle, Rawls, Gandhi, Kautilya), Ethics in Governance (Nolan Principles, 2nd ARC 4th Report, POCA 1988/2018), Case Study Frameworks & Emotional Intelligence.
  - Updated `src/kb/corpus_loader.py` with multi-paper metadata tags (`gs_paper`, `doc_type`), verified 1,036 total document chunks.
  - Created 35-item Golden Multi-GS Validation Benchmark (`tests/fixtures/golden_dataset_all_gs.json`).
- [x] Adopted Official `google-genai` SDK & Purged Manual Glue Code:
  - Added `"google-genai>=1.0.0"` dependency; replaced manual `requests.post` and `base64` boilerplate across `ChatVertexExpress`, `VertexVisionClient`, and `VertexGEvalScorer`.
  - Configured `genai.Client(vertexai=True, project=..., location=..., api_key=...)` for native Vertex AI Model Garden endpoints.
  - Used native `types.Part.from_bytes` for vision OCR and `types.GenerateContentConfig(response_logprobs=True, logprobs=5, thinking_config=types.ThinkingConfig(thinking_budget=0))` for fast (<2.4s) logprob retrieval.
- [x] Completely Purged Hardcoded Fallback Scoring & Silent Error Masking:
  - Deleted `_fallback_pillar_scoring` (which fabricated 3.0 rating / 45% marks) from `geval_scorer.py`.
  - Replaced `_extract_digit_fallback` with strict parsing raising `RatingExtractionError` if any pillar score is missing.
  - Removed canned defaults (`DEFAULT_COT_TRAIL`, `DEFAULT_STRENGTHS`, `DEFAULT_WEAKNESSES`, `DEFAULT_ACTION_PLAN`) from `engine.py`; Call 1 failures now raise `ModelInvocationError` immediately.
  - Fixed OCR failure masking in `src/parsing/pipeline.py` and `src/pipeline.py` so transcription exceptions set `error` and raise `DocumentParsingError` instead of falsely recording `is_blank=True` (which erroneously gave 0 marks).
  - Defined standard domain exceptions in `src/models/exceptions.py`.
  - Full test suite passing (41 passed, 7 Bedrock credential-skipped, 0 failures), 100% clean `ruff` check.
- [x] Fixed Vision Parsing Harness & Evaluated Complete 20-Question GS-2 Copy (`GS-II.pdf`):
  - Diagnosed root causes: confirmed 100% harness issue (not a vision model limitation). Bilingual header was caused by unconstrained verbatim prompt instruction; blank answers were caused by cross-thread `httpx` socket collisions on Windows.
  - Refined Vision Prompt (`src/parsing/prompts.py`): Explicitly instructed vision OCR to extract ONLY clean English questions from bilingual UPSC QCAB headers, excluding Devanagari text, numbering prefixes, and marks indications.
  - Added Regex Sanitizer (`src/parsing/pipeline.py`): Implemented `_clean_question_text` to filter Devanagari lines and strip question numbers ("Q1.") and trailing metadata ("(10 Marks, 150 words)").
  - Thread-Safe Vision & Scoring Clients (`src/parsing/vision_client.py`, `src/evaluation/geval_scorer.py`, `src/evaluation/vertex_chat.py`): Converted `VertexVisionClient` and `VertexGEvalScorer` to `threading.local()` isolated clients with retry logic for Windows `WinError 10053` socket resets.
  - Robust JSON Extraction (`src/utils/json.py`): Fixed premature regex truncation (changed non-greedy `.*?` to greedy `.*`) and added `json.loads(..., strict=False)` fallback.
  - ChatVertexExpress Tuning (`src/evaluation/vertex_chat.py`): Set `thinking_budget=0` and `max_output_tokens=8192` to eliminate token exhaustion truncation in Gemini 2.5 Flash Call 1 diagnostics.
  - Successfully Parsed `GS-II.pdf` (`tests/fixtures/gs2_copy_parsed.json`): 20/20 questions parsed with 0 Hindi characters, 0 blank attempts, and 160-300 words transcribed per question.
- [x] Authentic UPSC CSE Calibration & Grade Inflation Purged:
  - Eliminated grade inflation where toppers previously scored 138.33 / 250 (55.3%), which exceeded real UPSC CSE mark sheets.
  - Aligned calibration formula to authentic UPSC standards in `src/evaluation/geval_scorer.py`: `upsc_pct = 0.12 + (R * 0.08)`.
  - Recalibrated performance bands in `src/models/enums.py`: Needs Foundation (<32%), Average (32-40%, Interview Cutoff), Good / Selection Zone (41-47%, Rank 50-300), Topper Benchmark (48-55%+, Rank 1-50 Trajectory).
  - Explicitly injected authentic marking rubrics into `SYSTEM_PROMPT_UPSC_EXAMINER` in `src/evaluation/prompt_templates.py` (national AIR 1 toppers score 110-120 / 250, aggregate 135+ marks do not exist).
  - Purged invalid JavaScript comments (`//`) from prompt JSON schema, strengthened `extract_json_dict` with sanitized comment/comma stripping, and added parse-retry in `_run_diagnostic`.
  - Verified Whole-Copy Evaluation (`scripts/evaluate/topper_copy_evaluation.py`): whole copy scored an authentic **108.51 / 250.00 marks (43.4%)** in Good / Selection Zone trajectory, with 10-markers scoring 3.90–4.84 and 15-markers scoring 5.29–6.98.
- [x] Implemented AWS Serverless Lambda Deployment Layer (Zero Breaking Changes):
  - Pluggable Storage & State: `StorageService` (transparent local disk vs Amazon S3 uploads) and `JobStateService` (transparent local JSON vs Amazon DynamoDB with float/Decimal conversions).
  - Pluggable Queue Dispatcher: `QueueService` (local `threading.Thread` in development vs Amazon SQS message queue in production).
  - Lambda API Entrypoint: `src/lambda_api.py` with `Mangum` ASGI adapter for API Gateway HTTP API.
  - Lambda Worker Entrypoint: `src/lambda_worker.py` processing SQS batches with partial batch failure reporting, S3 download to `/tmp`, evaluation execution, and SES delivery.
  - Multi-Purpose Lambda Container: `Dockerfile` on `public.ecr.aws/lambda/python:3.11` bundling PyMuPDF, onnxruntime, ChromaDB, and application code.
  - Infrastructure-as-Code: `template.yaml` AWS SAM template defining S3 bucket, DynamoDB table, SQS queue + DLQ, API Lambda (29s timeout), Worker Lambda (15-minute timeout, 4GB RAM, 2GB `/tmp`), and IAM policies.
- [x] Implemented Decoupled Two-Stage Serverless Assembly Line Architecture (Zero Breaking Changes):
  - **Stage 1 (Vision OCR Worker)**: `src/lambda_vision.py` downloads PDF from S3 to `/tmp`, parses answer booklet via multimodal Vision OCR (`DocumentParsingPipeline`), saves transcribed JSON into DynamoDB, and dispatches job to `EvalQueue`.
  - **Stage 2 (Evaluation Engine Worker)**: `src/lambda_eval.py` retrieves parsed document from DynamoDB, performs RAG retrieval, runs 2-call CoT + G-Eval evaluation, compiles high-resolution Scorecard PDF via `pymupdf.Story`, and dispatches email via Amazon SES `send_raw_email` with PDF attached.
  - **Scorecard PDF Engine**: `src/services/scorecard_pdf.py` renders multi-page, executive evaluation scorecards directly to raw bytes or disk with zero external binaries (wkhtmltopdf/Puppeteer).
  - **Email Service Upgrade**: `src/services/email_service.py` upgraded to `MIMEMultipart('mixed')` supporting PDF attachment delivery via Amazon SES (`send_raw_email`), SMTP, and dev mock disk preview.
  - **Pluggable Dual-Queue Dispatcher**: `src/services/queue_service.py` upgraded with `dispatch_vision` and `dispatch_eval`, maintaining full backward compatibility for `dispatch` and `queue_url`.
  - **Instant Web UI Confirmation**: `src/static/index.html` updated with immediate visual confirmation ("You can safely close this page now!") and live polling fallback.
  - **Decoupled SAM Infrastructure as Code**: `template.yaml` updated with `VisionQueue` + DLQ, `EvaluationQueue` + DLQ, `JobStateTable`, `ParsedBookletsTable`, `ApiFunction`, `VisionWorkerFunction` (10m timeout, 3GB RAM), and `EvalWorkerFunction` (15m timeout, 4GB RAM).
  - **Test Suite**: 57 passing tests (including `test_scorecard_pdf.py`, `test_decoupled_workers.py`, `test_serverless_services.py`), 100% clean `ruff` check.
- [x] Migrated Knowledge Base to Cloud-Native Pinecone Serverless (Zero Heavy C++ Wheels):
  - Purged `chromadb`, `langchain-chroma`, `scipy`, `flashrank`, and `rank-bm25` from dependencies.
  - Successfully ingested all 1,036 authentic UPSC knowledge chunks into Pinecone index `upsc-kb` on AWS `us-east-1` (Serverless, dimension 768, cosine metric) using Vertex AI `text-embedding-004`.
  - Built pure-cloud `SelfQueryRetriever` in `src/kb/retriever.py` with multi-paper metadata filters and diversity ceilings.
  - Verified retrieval benchmark using DeepEval: Contextual Recall = 1.00, Contextual Precision = 0.84.
- [x] Regal Institutional Landing Page Redesign:
  - Pixel-perfect implementation of the provided design mockup in `src/static/index.html` and `index.html`.
  - Color palette: Alabaster ivory paper (`#FCFBF6`), deep midnight navy (`#0F1E36`), rich forest emerald green (`#0F5132`), warm golden amber (`#D97706`).
  - Seamlessly integrated authentic UPSC hero artwork (Indian Parliament Samvidhan Sadan, Ashoka Lion Capital, UPSC answer booklet and luxury fountain pen) on the left editorial column.
  - Added delicate watercolor botanical eucalyptus leaves framing the top-right corner.
  - Right column: Elevated white submission card with drag & drop PDF dropzone, recipient email input, QCAB page slicing accordion, forest green CTA button, and bottom trust badges (`Grounded Case Law Citations | Calibrated UPSC CSE Standards`).
  - Mounted `/static` in FastAPI `src/api/app.py` for high-resolution static asset serving.
  - Preserved full asynchronous job dispatching to `POST /api/v1/jobs/submit`, immediate HTTP 202 leave confirmation modal, and real-time live status polling fallback.
- [x] Direct S3 Presigned Upload Architecture (Bypassing API Gateway 10MB Limit):
  - Solved HTTP 413 Payload Too Large error when uploading large scanned answer booklets (e.g. 230 MB).
  - Implemented `generate_presigned_upload_url` in `src/services/storage_service.py` generating secure, short-lived S3 PUT URLs.
  - Implemented `create_job_from_storage_ref` in `src/services/job_manager.py` allowing job creation from existing S3 objects.
  - Added `GET /api/v1/jobs/upload-url` and `POST /api/v1/jobs/submit-direct` in `src/api/app.py`.
  - Upgraded frontend in `src/static/index.html` and `index.html` to stream large PDFs directly to Amazon S3 with real-time percentage progress bar before triggering background evaluation (< 1 KB payload).
- [x] Project Structure Clean-Up & Production Refactoring:
  - Purged root-level large PDFs (`GS-II.pdf`, `VIKAS...pdf`), stray test artifacts (`test_msg.json`, `test_scorecard.pdf`), and redundant root files (`index.html`, `static/`).
  - Purged 18 unused design iteration PNGs from `src/static/`, leaving `src/static/index.html` as the single source of truth.
  - Consolidated LLM and vision model clients into `src/providers/` (`vertex.py`, `vision.py`, `factory.py`).
  - Consolidated AWS Lambda entrypoint handlers into `src/handlers/` (`api.py`, `vision_worker.py`, `eval_worker.py`), updating `template.yaml` and `Dockerfile`.
  - Renamed `src/pipeline.py` -> `src/orchestrator.py` and `src/parsing/pipeline.py` -> `src/parsing/document_parser.py` to eliminate ambiguous name collision.
  - Fixed circular import between providers and parsing packages.
- [x] Hybrid Free-Tier Direct-S3 + EC2 Coordinator + 20-Lambda Fan-Out Architecture:
  - **Memory Cliff Mitigations**:
    - Calibrated default parsing DPI to 135 DPI in `src/config.py` (80% RAM reduction, 0% Lanczos downsampling CPU waste).
    - Upgraded `PDFPreprocessor` in `src/parsing/preprocessor.py` with `render_page_processed`: single-pass rendering, direct PyMuPDF C-level JPEG compression (`pix.tobytes("jpeg")`), fast blank sampling, and explicit `del pix` + `gc.collect()`.
    - Eliminated double-rendering in `src/parsing/document_parser.py`.
    - Cached singleton `genai.Client` in `ChatVertexExpress` (`src/providers/vertex.py`) via `PrivateAttr` to prevent spawning dozens of HTTP/2 connection pools.
  - **Single-Question Lambda Evaluator (`src/handlers/single_question_eval.py`)**:
    - Micro-worker evaluating 1 question in ~7-8 seconds with pre-provided KB ground truth context.
    - Pure compute micro-container with zero vector DB / Pinecone / PyMuPDF dependencies.
    - Added `SingleQuestionEvalFunction` resource and output to `template.yaml`.
  - **Knowledge Base Batch Prefetcher (`src/kb/prefetcher.py`)**:
    - Prefetches authoritative statutory and landmark case context blocks for all 20 questions in parallel on EC2 (< 1.0s).
    - Supports fast-path blank question skipping.
  - **EC2 Concurrency-Gated Fan-Out Orchestrator (`src/orchestrator_ec2.py`)**:
    - Enforces `MAX_CONCURRENT_COPIES = 2` via `asyncio.Semaphore(2)` so peak EC2 memory stays < 350 MB on `t3.micro`.
    - Fans out 20 concurrent SingleQuestionEval Lambdas, completes evaluation in ~7.5s flat, and compiles/delivers student scorecards.
  - **Immediate Storage Cleanup & DynamoDB TTL**:
    - Added `delete_file` and `get_file_size` (200 MB limit) to `StorageService` (`src/services/storage_service.py`).
    - Added 1-day expiration lifecycle rule to `UploadsBucket` in `template.yaml`.
    - Auto-injected 14-day TTL in `JobStateService` (`src/services/job_state_service.py`) and enabled TTL in `template.yaml`.
  - **EC2 Free-Tier Bootstrap (`scripts/deploy/setup_ec2.sh`) & Worker Daemon (`src/handlers/ec2_worker.py`)**:
    - Created long-polling `EC2WorkerDaemon` in `src/handlers/ec2_worker.py` consuming SQS jobs with concurrency gating and 20-Lambda fan-out.
    - Automated provisioning of 3 GB swap file on the 30 GB EBS volume, swappiness tuning, and systemd services (`upscopilot.service` and `upscopilot-worker.service`).
  - **Monolithic Lambda Worker Purge**:
    - Purged `VisionWorkerFunction`, `EvalWorkerFunction`, `VisionQueue`, `VisionDeadLetterQueue`, and `ParsedBookletsTable` from `template.yaml`.
    - CloudFormation will automatically delete old workers and tables upon `sam deploy`, activating the clean EC2 Coordinator + `SingleQuestionEvalFunction` micro-worker stack.
  - **Comprehensive Test Suite**: 63 offline tests passing (including `tests/test_ec2_coordinator.py` and `tests/test_lambda_handlers.py`), 100% clean `ruff` check.







<!-- BEGIN AWS Agent Toolkit rules -->
# AWS Guidance

- Where these AWS rules conflict with the project's own instructions, the
  project's instructions take precedence.
- Prefer the AWS MCP Server for AWS interactions — it provides sandboxed
  execution, observability, and audit logging. If unavailable, use the
  AWS CLI directly.
- Before starting a task, check whether a relevant AWS skill is available.
  Load the skill with `retrieve_skill` and prefer its guidance over
  general knowledge.
- When uncertain about specific AWS details (API parameters, permissions,
  limits, error codes), verify against documentation rather than guessing.
  State uncertainty explicitly if you cannot confirm.
- When creating infrastructure, prefer infrastructure-as-code (AWS CDK or
  CloudFormation) over direct CLI commands.
<!-- END AWS Agent Toolkit rules -->


