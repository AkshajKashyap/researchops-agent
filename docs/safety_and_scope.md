# Safety and Scope

ResearchOps Agent is designed around bounded, inspectable local behavior.

## Safety Properties

- No arbitrary shell execution is exposed through configs or workflows.
- No arbitrary Python execution is allowed from suggested configs.
- The bounded runner only accepts explicitly validated `ExperimentConfig` values.
- Unsupported tasks, datasets, models, metrics, and validation strategies fail before running.
- LLM answers are validated against retrieved evidence citations.
- The fake LLM provider supports deterministic offline demos and tests.
- The OpenAI provider is optional and isolated behind a provider interface.
- Trace logs store metadata only: query, provider, model, citations, abstention, latency, errors.
- Traces do not store full evidence text, prompts, API keys, or secrets.

## Current Scope

- Document loading supports local text, Markdown, and PDF files.
- Corpus indexes persist manifests, metadata, and chunks as JSON.
- Retriever models are re-fit from stored chunks at query time.
- The local index is not optimized for huge corpora.
- Claim extraction is heuristic and should be reviewed.
- Config suggestion is heuristic and may produce non-runnable configs.
- The bounded runner currently supports only `time_series_forecasting` on `synthetic_sine`.
- Optional embedding retrieval may require a local model download outside tests.
- OpenAI usage requires environment configuration and is never required for tests.

## What It Does Not Claim

The project does not reproduce arbitrary papers, provide production-scale search, guarantee LLM
correctness, or replace careful human review. Its purpose is to demonstrate a safe ResearchOps
loop: evidence, citations, validation, and bounded execution before broader automation.
