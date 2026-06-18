
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

## Current Evaluation MVP

The project now includes a small deterministic evaluation harness:

- A local demo corpus in `examples/docs/`.
- Retrieval evaluation cases.
- Answer evaluation cases.
- Hit-rate and pass-rate reporting.
- JSON and Markdown evaluation report export.

```bash
researchops eval --retrieval-cases examples/eval/retrieval_cases.json --answer-cases examples/eval/answer_cases.json --out-json reports/evaluation.json --out-md reports/evaluation.md
```

Honesty note: the current evaluation harness is small and deterministic. It measures basic
evidence retrieval and extractive answer grounding, not full natural-language answer quality.

## Current Retrieval Comparison MVP

TF-IDF retrieval remains the deterministic baseline, and local embedding retrieval is now
available through `sentence-transformers`. Retrieval backends are selectable through the CLI,
and the evaluation harness can compare backend hit rates and pass rates. Tests avoid external
model downloads by using fake embedding models.

```bash
researchops retrieve examples/docs/time_series_note.md "What models were compared?" --retriever tfidf
researchops retrieve examples/docs/time_series_note.md "What models were compared?" --retriever embedding
researchops eval --retriever tfidf --retrieval-cases examples/eval/retrieval_cases.json --answer-cases examples/eval/answer_cases.json --out-json reports/evaluation.json --out-md reports/evaluation.md
researchops compare-retrievers --retrievers tfidf,embedding --retrieval-cases examples/eval/retrieval_cases.json --answer-cases examples/eval/answer_cases.json --out-dir reports/retriever_comparison
```

Honesty note: embedding retrieval can improve semantic matching, but it is not guaranteed to
beat TF-IDF on a tiny demo corpus. The evaluation harness exists to measure this instead of
assuming it. The first real embedding run may need to download the local sentence-transformers
model.

## Current Bounded Experiment Runner MVP

The project can now close a safe local ResearchOps loop:

- Suggest experiment configs from document evidence.
- Validate whether configs are runnable by the bounded local runner.
- Run a toy time-series forecasting experiment.
- Compare `mean_baseline` and `ridge`.
- Write metrics and artifacts to a run directory.
- Avoid arbitrary shell command or Python code execution.

```bash
researchops suggest-config examples/docs/time_series_note.md "What experiment is described?" --out configs/suggested.yaml --retriever tfidf
researchops run-config configs/time_series_demo.yaml --out-dir reports/runs
```

Honesty note: the runner only supports a small local time-series forecasting task right now. It
is designed to demonstrate safe ResearchOps orchestration, not to reproduce arbitrary papers.

## Current API and Dashboard MVP

The project now supports a local demo surface around the same deterministic core:

- CLI usage for ingestion, retrieval, answering, reports, evaluation, and bounded runs.
- FastAPI endpoints for document queries, reports, config suggestion, runs, and evaluation.
- A simple Streamlit dashboard for local demos.
- A local Dockerized API.
- Bounded experiment runs with metrics/artifacts.
- Local JSON and Markdown evaluation reports.

```bash
make api
make dashboard
make demo-eval
make demo-run
docker build -t researchops-agent .
docker run -p 8000:8000 researchops-agent
```

API examples:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"path":"examples/docs/time_series_note.md","query":"What experiment is described?","retriever":"tfidf"}'

curl -X POST http://localhost:8000/run-config \
  -H "Content-Type: application/json" \
  -d '{"config_path":"configs/time_series_demo.yaml","out_dir":"reports/runs","seed":42}'
```

Honesty note: the API and dashboard wrap deterministic local functionality. This is not yet an
LLM agent, and the bounded runner only supports a small explicit set of experiment configs.

## Current Optional LLM Grounded Answering MVP

Deterministic extractive answering remains the safe baseline. The optional LLM layer adds a
provider abstraction with a fake provider for offline tests/demos and an OpenAI provider that
requires environment configuration. LLM answers are validated against retrieved citations:
unsupported, uncited, or empty answers are forced to abstain. Trace logs store metadata only,
not full evidence text, prompts, or secrets.

```bash
researchops llm-ask examples/docs/time_series_note.md "What models were compared?" --provider fake --retriever tfidf
researchops llm-report examples/docs/time_series_note.md "What experiment is described?" --provider fake --out reports/llm_report.json
researchops eval-llm --answer-cases examples/eval/answer_cases.json --provider fake --out-json reports/llm_answer_eval.json --out-md reports/llm_answer_eval.md
OPENAI_MODEL=<your_model> researchops llm-ask examples/docs/time_series_note.md "What models were compared?" --provider openai --retriever tfidf
```

Honesty note: the LLM layer is optional and grounded by retrieved evidence. It is not allowed
to browse the web, execute code, or cite sources outside the evidence pack.

## Current Multi-Document Corpus MVP

The project now supports a small local research workspace:

- Discover local research documents.
- Build a persistent local corpus index.
- Search across multiple documents.
- Answer from corpus evidence.
- Run grounded LLM answering over corpus evidence.
- Evaluate corpus retrieval.
- Use a multi-document demo corpus in `examples/docs`.

```bash
make demo-index
researchops search-corpus data/indexes/demo_corpus "Which models are compared across the notes?"
researchops ask-corpus data/indexes/demo_corpus "What experiment limitations are mentioned?"
researchops llm-ask-corpus data/indexes/demo_corpus "What datasets are used?" --provider fake
make demo-corpus-eval
```

Honesty note: the persistent index stores chunk metadata and corpus manifests. Retriever models
are re-fit from stored chunks at query time for simplicity. This is acceptable for a local MVP
but not optimized for large corpora.

## Current End-to-End Workflow MVP

The project can now run one coherent local ResearchOps workflow:

- Query a local corpus index.
- Retrieve citation-ready evidence.
- Answer with citations.
- Extract experiment-like claims.
- Suggest an experiment config.
- Validate whether the config is runnable.
- Optionally run a bounded local experiment.
- Produce JSON and Markdown workflow artifacts.

```bash
make demo-index
make demo-workflow
researchops workflow data/indexes/demo_corpus "What experiment is described and can we run a local version?" --run-if-runnable --out-dir reports/workflows
```

Honesty note: the workflow orchestrator coordinates deterministic and optional LLM components,
but it is still bounded. It does not browse the web, execute arbitrary code, or reproduce
arbitrary papers.

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
