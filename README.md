# ResearchOps Agent

Citation-grounded local ResearchOps for documents, evidence, configs, and bounded experiment demos.

ResearchOps Agent is a Python project for working with small research-paper and project-note
corpora. It can ingest local documents, retrieve citation-ready evidence, answer from that
evidence, extract experiment-like claims, suggest configs, validate whether a config is runnable,
and optionally run a safe toy forecasting experiment.

The project is intentionally bounded. It does not browse the web, execute arbitrary code, use a
database, or claim to reproduce arbitrary papers.

## Why It Matters

Many research assistants jump directly from retrieval to generated prose. This project keeps the
intermediate steps visible: chunks, citations, evidence packs, abstention, config validation, run
artifacts, and evaluation reports. The goal is an auditable local workflow before broad automation.

## Key Features

- Local ingestion for `.txt`, `.md`, and `.pdf`.
- Deterministic character chunking with source metadata.
- TF-IDF retrieval baseline and optional local embedding retrieval.
- Single-document and multi-document corpus search.
- Citation-ready evidence packs.
- Deterministic extractive answering with abstention.
- Optional grounded LLM answering with fake/OpenAI providers and citation validation.
- Heuristic experiment claim extraction and config suggestion.
- Safe bounded runner for one explicit local time-series forecasting task.
- Evaluation harnesses for retrieval, answers, LLM answers, and corpus retrieval.
- End-to-end workflow orchestration with JSON and Markdown artifacts.
- CLI, FastAPI, Streamlit, Docker, Makefile, and GitHub Actions surfaces.

## Architecture Summary

```text
documents -> loaders -> pages -> chunks -> retriever -> evidence pack
          -> extractive or grounded LLM answer -> citations
          -> claim extraction -> config suggestion -> validation -> optional bounded run
          -> JSON/Markdown reports and workflow artifacts
```

For details, see [docs/architecture.md](docs/architecture.md) and
[docs/safety_and_scope.md](docs/safety_and_scope.md).

## Quickstart

```bash
python -m pip install -e .
researchops health
researchops version
```

Build the demo corpus index:

```bash
make demo-index
```

Search and ask across the corpus:

```bash
researchops search-corpus data/indexes/demo_corpus "Which graph kernels were compared?"
researchops ask-corpus data/indexes/demo_corpus "What experiment limitations are mentioned?"
```

Run the end-to-end workflow:

```bash
researchops workflow data/indexes/demo_corpus \
  "What experiment is described and can we run a local version?" \
  --run-if-runnable \
  --out-dir reports/workflows
```

Generate the final local demo artifacts:

```bash
make demo-all
```

## Demo Commands

```bash
make full-check
make demo-index
make demo-corpus-search
make demo-corpus-eval
make demo-run
make demo-workflow
make demo-all
```

`make demo-all` writes:

- `reports/final_demo/evaluation.json`
- `reports/final_demo/evaluation.md`
- `reports/final_demo/corpus_eval.json`
- `reports/final_demo/corpus_eval.md`
- `reports/final_demo/runs/...`
- `reports/final_demo/workflows/...`

## CLI Overview

Core examples:

```bash
researchops ingest examples/docs/time_series_note.md
researchops retrieve examples/docs/time_series_note.md "What models are compared?"
researchops ask examples/docs/time_series_note.md "What metrics are used?"
researchops report examples/docs/time_series_note.md "What experiment is described?" --out reports/report.json
```

Corpus examples:

```bash
researchops index-corpus examples/docs --index-dir data/indexes/demo_corpus
researchops search-corpus data/indexes/demo_corpus "Which models are compared?"
researchops llm-ask-corpus data/indexes/demo_corpus "What datasets are used?" --provider fake
```

Runner and workflow examples:

```bash
researchops suggest-config examples/docs/time_series_note.md "What experiment is described?" --out configs/suggested.yaml
researchops run-config configs/time_series_demo.yaml --out-dir reports/runs
researchops workflow data/indexes/demo_corpus "What experiment is described?" --out-dir reports/workflows
```

See [docs/cli_reference.md](docs/cli_reference.md) for a command inventory.

## API and Dashboard

Start the FastAPI service:

```bash
make api
```

Start the Streamlit dashboard:

```bash
make dashboard
```

Example API calls:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"path":"examples/docs/time_series_note.md","query":"What metrics are used?","retriever":"tfidf"}'

curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{"index_dir":"data/indexes/demo_corpus","query":"What experiment is described?","run_if_runnable":true}'
```

See [docs/api_reference.md](docs/api_reference.md) for endpoint examples.

## Current Demo Results

These are from the included tiny local fixtures generated by
`scripts/generate_demo_artifacts.py`:

- Single-document evaluation: 4 retrieval cases, 4 hits, hit rate `1.00`.
- Single-document answer evaluation: 4 answer cases, 4 passes, pass rate `1.00`.
- Corpus retrieval evaluation: 5 retrieval cases, 5 hits, hit rate `1.00`.
- Bounded runner demo:
  - `mean_baseline_rmse`: `0.721089`
  - `mean_baseline_mae`: `0.633234`
  - `ridge_rmse`: `0.122892`
  - `ridge_mae`: `0.100187`

These results are not benchmark claims. They show that the local demo fixtures and audit paths are
working.

## Safety and Scope Limits

- No arbitrary shell execution.
- No arbitrary Python execution from configs.
- No web search.
- No database.
- No LangChain or LangGraph.
- Real OpenAI calls are optional and never required for tests.
- The fake LLM provider is deterministic and offline.
- LLM citations are validated against retrieved evidence, but LLM output is not guaranteed correct.
- Embedding retrieval is optional and may require a local model download outside tests.
- The corpus index stores JSON metadata and chunks, then re-fits retrievers at query time.
- Claim extraction and config suggestion are heuristic.
- The bounded runner only supports a small local `time_series_forecasting` demo task.

## Project Structure

```text
src/researchops_agent/
  ingestion/       loaders and chunking
  retrieval/       TF-IDF, optional embeddings, citations, evidence packs
  agents/          extractive answers, LLM-grounded answers, claim/report helpers
  corpus/          discovery, manifest/index persistence, corpus search
  runner/          config validation, synthetic data, forecasting runner
  evaluation/      retrieval, answer, LLM, and corpus eval harnesses
  workflows/       end-to-end ResearchOps orchestration
  api/             FastAPI app
  schemas/         Pydantic models
  utils/           JSON/YAML/text helpers
app/               Streamlit dashboard
docs/              architecture, safety, demo, CLI, API, portfolio notes
examples/          demo docs and eval cases
scripts/           demo artifact generator
```

## Future Work

- Add better parsing for tables and experiment sections.
- Expand the bounded runner to more safe task families.
- Add larger evaluation fixtures and trend reports.
- Improve corpus index performance while staying local and inspectable.
- Add workflow comparison and artifact diffing.

## More Docs

- [Architecture](docs/architecture.md)
- [Safety and Scope](docs/safety_and_scope.md)
- [Demo Script](docs/demo_script.md)
- [Technical Project Report](docs/project_report.md)
- [CLI Reference](docs/cli_reference.md)
- [API Reference](docs/api_reference.md)
- [Portfolio Summary](docs/portfolio_summary.md)
