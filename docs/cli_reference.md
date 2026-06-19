# CLI Reference

| Command | Area | What it does | Minimal example |
| --- | --- | --- | --- |
| `health` | Core | Checks that the CLI is wired correctly. | `researchops health` |
| `version` | Core | Prints the package version. | `researchops version` |
| `ingest` | Single-document | Loads and chunks a document. | `researchops ingest examples/docs/time_series_note.md` |
| `retrieve` | Single-document | Retrieves chunks from one document. | `researchops retrieve examples/docs/time_series_note.md "What models are compared?"` |
| `ask` | Single-document | Answers extractively from one document. | `researchops ask examples/docs/time_series_note.md "What metrics are used?"` |
| `claims` | Single-document | Extracts experiment-like claims. | `researchops claims examples/docs/time_series_note.md "What experiment is described?"` |
| `report` | Single-document | Writes a JSON research report. | `researchops report examples/docs/time_series_note.md "What experiment is described?" --out reports/report.json` |
| `llm-ask` | Optional LLM | Answers with fake/OpenAI grounded provider. | `researchops llm-ask examples/docs/time_series_note.md "What models were compared?" --provider fake` |
| `llm-report` | Optional LLM | Builds an LLM-grounded report. | `researchops llm-report examples/docs/time_series_note.md "What experiment is described?" --provider fake` |
| `index-corpus` | Corpus | Builds a persisted local corpus index. | `researchops index-corpus examples/docs --index-dir data/indexes/demo_corpus` |
| `search-corpus` | Corpus | Searches a persisted corpus index. | `researchops search-corpus data/indexes/demo_corpus "Which graph kernels were compared?"` |
| `ask-corpus` | Corpus | Answers extractively from corpus evidence. | `researchops ask-corpus data/indexes/demo_corpus "What limitations are mentioned?"` |
| `llm-ask-corpus` | Corpus + LLM | Uses grounded LLM answering over corpus evidence. | `researchops llm-ask-corpus data/indexes/demo_corpus "What models are compared?" --provider fake` |
| `corpus-report` | Corpus | Writes a deterministic corpus report. | `researchops corpus-report data/indexes/demo_corpus "What datasets are used?"` |
| `llm-corpus-report` | Corpus + LLM | Writes a grounded LLM corpus report. | `researchops llm-corpus-report data/indexes/demo_corpus "What experiment is described?" --provider fake` |
| `suggest-config` | Runner | Suggests a heuristic experiment config. | `researchops suggest-config examples/docs/time_series_note.md "What experiment is described?" --out configs/suggested.yaml` |
| `run-config` | Runner | Runs a supported bounded config. | `researchops run-config configs/time_series_demo.yaml --out-dir reports/runs` |
| `workflow` | Workflow | Runs corpus search, answering, claims, config validation, optional bounded run, and artifacts. | `researchops workflow data/indexes/demo_corpus "What experiment is described?" --out-dir reports/workflows` |
| `eval` | Evaluation | Runs single-document retrieval and answer evals. | `researchops eval --retriever tfidf` |
| `eval-llm` | Evaluation | Evaluates grounded LLM answers with fake/OpenAI provider. | `researchops eval-llm --provider fake` |
| `eval-corpus` | Evaluation | Evaluates retrieval against a corpus index. | `researchops eval-corpus --index-dir data/indexes/demo_corpus` |
| `compare-retrievers` | Evaluation | Compares configured retrievers on eval cases. | `researchops compare-retrievers --retrievers tfidf,embedding` |

All examples can run with TF-IDF and the fake provider. Real OpenAI calls and embedding model
downloads are optional and not used in tests.
