
ResearchOps Agent

An evaluated AI research assistant for ML papers and project docs.

## Current MVP

The project currently supports a fully local document ingestion foundation:

- Load local PDF, Markdown, and text documents.
- Read PDFs page by page with `pypdf`.
- Normalize basic whitespace while preserving document text.
- Create deterministic character chunks with source path, page number, and chunk index metadata.
- Use the CLI to inspect ingestion output:

```bash
researchops ingest data/raw/example.md
```

## Current Retrieval MVP

The retrieval layer now fits a local `TfidfVectorizer` over deterministic chunks and returns
citation-ready results with source path, source type, page number, chunk index, text, and score.
This proves that retrieval and citations work before adding any LLM-based answering.

```bash
researchops retrieve data/raw/example.md "What is the paper about?"
```

## Current Answering MVP

The system can now build citation-ready evidence packs from retrieval results and answer from
the top retrieved evidence. Answering is extractive, deterministic, and abstains when retrieved
evidence is insufficient. This intentionally comes before LLM-based answering so citations and
evidence handling are trustworthy first.

```bash
researchops ask data/raw/example.md "What is this document about?"
```

## Current ResearchOps MVP

The project now supports the first end-to-end local ResearchOps flow:

- Ingest local research documents.
- Retrieve citation-ready evidence.
- Answer extractively from retrieved evidence.
- Extract experiment-like claims with deterministic heuristics.
- Suggest a heuristic experiment config.
- Export a JSON research report.

```bash
researchops claims data/raw/example.md "What models and metrics are used?"
researchops report data/raw/example.md "What experiment does this document describe?" --out reports/example_report.json
```

Honesty note: the current system is deterministic and heuristic. It does not yet use LLM
reasoning, external tools, or real experiment execution.

The system will:

ingest PDFs, Markdown files, and repo docs
answer questions with retrieved evidence and citations
extract experiment claims into structured schemas
generate validated experiment configs
run bounded experiments instead of arbitrary code
produce auditable research memos
Planned Stack
Python
FastAPI
Streamlit or simple frontend
local vector store
Pydantic schemas
pytest
Docker
GitHub Actions
Core Modules

src/researchops_agent/
ingestion/ PDF/Markdown loading, chunking, metadata
retrieval/ embeddings, vector search, citation retrieval
agents/ research assistant logic and tool orchestration
runner/ safe config-based experiment runner
evaluation/ retrieval, grounding, abstention, latency evals
api/ FastAPI endpoints
schemas/ Pydantic models and config schemas
utils/ shared helpers

First Milestone

Build a local MVP:

Load one ML paper/report.
Chunk it with source metadata.
Retrieve relevant chunks for a question.
Answer only from retrieved evidence.
Cite source chunks.
Refuse when evidence is missing.
