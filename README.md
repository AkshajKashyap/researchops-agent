
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
