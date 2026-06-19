# Demo Script

This is a 2 to 3 minute walkthrough for a local demo.

## 1. Install and Check

```bash
python -m pip install -e .
researchops health
researchops version
```

Talking point: the project is local-first. The default demo path uses TF-IDF and deterministic
logic, so no API key is needed.

## 2. Build a Corpus Index

```bash
make demo-index
```

Talking point: the corpus index stores document metadata and chunks, not a database or heavy model.
Retriever state is re-fit from stored chunks for simplicity.

## 3. Search and Ask the Corpus

```bash
researchops search-corpus data/indexes/demo_corpus "Which graph kernels were compared?"
researchops ask-corpus data/indexes/demo_corpus "What experiment limitations are mentioned?"
```

Talking point: results carry source paths and chunk citations, so answers can be inspected.

## 4. Run the End-to-End Workflow

```bash
researchops workflow data/indexes/demo_corpus \
  "What experiment is described and can we run a local version?" \
  --run-if-runnable \
  --out-dir reports/workflows
```

Talking point: the workflow retrieves evidence, answers with citations, extracts claims, suggests
a config, validates it, and only runs if the config is supported.

## 5. Show Workflow Artifacts

```bash
ls reports/workflows
```

Open the generated `workflow_summary.md` and `workflow_result.json`.

Talking point: the artifact trail is meant to be auditable. It records steps, citations, config
validation, optional run results, and honesty notes.

## 6. Run the Bounded Demo Config

```bash
researchops run-config configs/time_series_demo.yaml --out-dir reports/runs
```

Talking point: this is a deliberately small local forecasting demo comparing `mean_baseline` and
`ridge` on synthetic data.

## 7. Mention API and Dashboard

```bash
make api
make dashboard
```

Talking point: FastAPI and Streamlit wrap the same local functionality. They do not add web search,
arbitrary code execution, or required LLM calls.
