# Portfolio Summary

ResearchOps Agent is a local, citation-grounded assistant for research papers, project notes, and
bounded experiment demos. It can ingest documents, retrieve evidence, answer with citations, extract
experiment-like claims, suggest configs, validate whether a config is runnable, and optionally run a
small safe forecasting experiment while writing auditable artifacts.

## Strongest Technical Points

- Local ingestion and deterministic chunking for text, Markdown, and PDF files.
- Citation-ready retrieval over single documents and persisted local corpora.
- Extractive answering baseline with abstention when evidence is weak.
- Optional LLM provider abstraction with fake offline provider and citation validation.
- End-to-end workflow orchestration with JSON/Markdown audit artifacts.
- Bounded runner that validates configs before execution.
- Evaluation harnesses for retrieval, answers, corpus retrieval, and LLM answer paths.
- FastAPI, Streamlit, Docker, Makefile, and CI surfaces around the same core modules.

## What Is Deliberately Bounded

- No web search.
- No LangChain or LangGraph.
- No database.
- No arbitrary shell or Python execution.
- No real API calls in tests.
- No required embedding model downloads in tests.
- Only a small supported time-series forecasting runner.

## Tradeoffs

- Corpus indexes are JSON files and retrievers are re-fit at query time. This keeps the MVP simple
  and inspectable but is not optimized for large corpora.
- Claim extraction is heuristic. It is deterministic and easy to test, but not a robust scientific
  information extraction system.
- The fake LLM provider is intentionally simple. It gives offline demo coverage without implying
  real generative reasoning.

## What I Would Improve Next

- Add richer document parsing for tables, captions, and experiment sections.
- Expand bounded runners to a few more safe task families.
- Add stronger benchmark fixtures and historical eval trend reporting.
- Improve index performance while preserving local inspectability.
- Add workflow comparison and artifact diff views.

## Resume Bullet Options

- Built a local ResearchOps assistant that ingests PDFs/Markdown/text, retrieves citation-ready
  evidence, and answers with deterministic abstention behavior.
- Implemented multi-document corpus indexing and retrieval with persisted JSON manifests, chunk
  metadata, and corpus-level evaluation.
- Designed an optional grounded LLM layer with fake/OpenAI providers, strict citation validation,
  and metadata-only trace logging.
- Built a bounded experiment runner that validates supported configs before running deterministic
  synthetic time-series forecasting experiments.
- Added an end-to-end workflow orchestrator that turns corpus evidence into answers, claims,
  suggested configs, validation results, optional runs, and auditable JSON/Markdown artifacts.
