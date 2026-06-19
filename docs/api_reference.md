# API Reference

The FastAPI service wraps the same local deterministic and optional grounded LLM functionality as
the CLI.

Run locally:

```bash
make api
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health check. |
| `POST /retrieve` | Retrieve chunks from one document. |
| `POST /ask` | Extractive answer from one document. |
| `POST /claims` | Extract experiment-like claims from one document. |
| `POST /report` | Build a single-document report. |
| `POST /suggest-config` | Suggest an experiment config from document evidence. |
| `POST /run-config` | Run a supported bounded experiment config. |
| `POST /eval` | Run local retrieval/answer evaluation. |
| `POST /llm/ask` | Optional grounded LLM answer from one document. |
| `POST /llm/report` | Optional grounded LLM report from one document. |
| `POST /corpus/index` | Build a local corpus index. |
| `POST /corpus/search` | Search a corpus index. |
| `POST /corpus/ask` | Extractive corpus answer. |
| `POST /corpus/report` | Deterministic corpus report. |
| `POST /llm/corpus/ask` | Optional grounded LLM answer over corpus evidence. |
| `POST /llm/corpus/report` | Optional grounded LLM corpus report. |
| `POST /workflow` | Run the end-to-end workflow orchestrator. |

## Examples

Health:

```bash
curl http://localhost:8000/health
```

Single-document ask:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"path":"examples/docs/time_series_note.md","query":"What metrics are used?","retriever":"tfidf","top_k":5}'
```

Corpus search:

```bash
curl -X POST http://localhost:8000/corpus/search \
  -H "Content-Type: application/json" \
  -d '{"index_dir":"data/indexes/demo_corpus","query":"Which graph kernels were compared?","top_k":5}'
```

Workflow:

```bash
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{"index_dir":"data/indexes/demo_corpus","query":"What experiment is described?","run_if_runnable":true,"out_dir":"reports/workflows"}'
```

Run config:

```bash
curl -X POST http://localhost:8000/run-config \
  -H "Content-Type: application/json" \
  -d '{"config_path":"configs/time_series_demo.yaml","out_dir":"reports/runs","seed":42}'
```

Provider configuration errors and validation errors are returned as HTTP 400 responses. Real OpenAI
usage is optional and requires environment configuration.
