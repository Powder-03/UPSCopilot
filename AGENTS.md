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
- **Test Script**: `scripts/test_llm.py` (`uv run python scripts/test_llm.py` verified working)

---

## 3. Real Raw Knowledge Base Ingestion Architecture (Zero Hardcoding)
- **Strictly No Hardcoded Python Arrays / Snippets**: All hardcoded python arrays have been deleted.
- **Primary Raw Sources**:
  1. **Complete Constitution**: All 465 authentic articles + Preamble loaded from `src/kb/data/raw/constitution_raw.json` (`scripts/ingest_constitution.py`).
  2. **Official Legislative Central Acts Downloader**: `scripts/populate_central_acts.py` downloads official government gazette/statute PDFs (RTI Act 2005, DPDP Act 2023, RPA 1951, Lokpal Act 2013, CVC Act 2003, PMLA 2002, Disaster Management Act 2005) into `src/kb/data/documents/`.
  3. **Official PDF & Textbook Loader**: `scripts/ingest_documents.py` processes all `.pdf`, `.txt`, and `.md` files in `src/kb/data/documents/` using PyMuPDF and `RecursiveCharacterTextSplitter`.
  4. **Master Ingestion Pipeline**: `scripts/ingest_all.py` batches all authentic documents into the local Chroma vector store (`src/kb/storage/chroma_db`) with Bedrock Titan Embeddings.
  5. **Hybrid Retriever**: `src/kb/retriever.py` provides BM25 + dense Chroma retrieval with Reciprocal Rank Fusion over the real ingested documents.

---

## 4. Repository Layout
```
evaluator/
├── .env                  # Local secrets (AWS credentials, model IDs, paths)
├── .env.example          # Environment template
├── AGENTS.md             # This file: agent context, memory & rules
├── pyproject.toml        # Dependencies (langchain, langchain-aws, chromadb, etc.)
├── scripts/
│   ├── test_llm.py       # Minimal verification script for Moonshot Kimi 2.5
│   ├── populate_central_acts.py # Downloads official Central Act PDFs from govt portals
│   ├── ingest_all.py     # Master ingestion pipeline for all authentic raw sources
│   ├── ingest_constitution.py  # All 465 articles from raw constitution data
│   └── ingest_documents.py     # Generic PDF / textbook / act loader
├── src/
│   ├── config.py         # Settings loading from .env
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── model_factory.py  # Factory returning ChatBedrockConverse (Kimi 2.5)
│   ├── kb/
│   │   ├── retriever.py  # Hybrid retriever (BM25 + Chroma RRF)
│   │   ├── vector_store.py # Chroma DB with Bedrock Titan Embeddings
│   │   └── data/
│   │       ├── raw/
│   │       │   └── constitution_raw.json # 465 authentic articles
│   │       └── documents/                # Official Act PDFs, textbooks, reports
│   └── models/
│       └── enums.py      # UPSC performance bands and directive types
└── tests/
```

---

## 5. Agent Working Rules (Strict)
1. **No Code Bloat or Hardcoded Snippets**: Never hardcode snippet dictionaries in Python files. Always work with authentic raw files (JSON, PDF, TXT, MD).
2. **Context Persistence**: Keep `AGENTS.md` updated whenever changes are made.
3. **Execution Environment**: Always run commands using `uv run python ...` within the workspace.

---

## 6. Current Progress & Status
- [x] Configured AWS Bedrock with Moonshot Kimi 2.5 (`moonshotai.kimi-k2.5`).
- [x] Verified inferencing with `scripts/test_llm.py` (`"I'm active and ready to help."`).
- [x] Verified Bedrock Titan Embeddings (`amazon.titan-embed-text-v2:0`).
- [x] Deleted all hardcoded snippet files (`ingest_central_acts.py`, `ingest_sc_cases.py`, mock JSONs).
- [x] Built real raw ingestion pipeline with zero hardcoding (`populate_central_acts.py`, `ingest_constitution.py`, `ingest_documents.py`, `ingest_all.py`).
- [x] Populated all 14 authentic Supreme Court landmark case dossiers in `src/kb/data/documents/sc_cases/`.
- [x] Populated comprehensive statutory dossiers for all 7 key Central Acts in `src/kb/data/documents/central_acts/` (RTI 2005, DPDP 2023, RPA 1951, Lokpal 2013, CVC 2003, PMLA 2002, Disaster Management 2005).
- [x] Created 15-item Golden Dataset (`tests/data/golden_dataset.json`) with canonical Ideal Answers spanning the GS-2 syllabus.
- [x] Configured chunking parameters to `chunk_size=1000` and `chunk_overlap=200` across `corpus_loader.py` and `ingest_documents.py`.
- [x] Integrated FlashRank cross-encoder reranker (`flashrank`) into `src/kb/retriever.py` with expanded candidate pool (top 40) and relaxed source diversity ceiling (`max_per_source=5`) to ensure complete statutory and landmark case coverage without premature truncation.
- [x] Refined `case_puttaswamy_2017.md` with explicit 4-fold proportionality test and decisional autonomy facets.
- [ ] Active: Ready for Step 4 — Core UPSC Answer Evaluation Engine with Kimi 2.5.
