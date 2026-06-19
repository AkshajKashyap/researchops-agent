# Project Report

## Problem

Research workflows often mix papers, notes, experiments, and loosely tracked decisions. A useful
assistant for this space should preserve evidence, cite sources, validate what it can run, and
avoid pretending to reproduce arbitrary work.

## Design Goals

- Keep the core local, deterministic, and testable.
- Require citations for answers.
- Make abstention a normal outcome when evidence is weak.
- Separate retrieval, evidence, answering, claims, configs, and execution.
- Add optional LLM generation only after citation validation exists.
- Restrict execution to small explicitly supported tasks.

## Implementation Summary

The project loads text, Markdown, and PDF documents into page schemas, chunks them
deterministically, and retrieves chunks with TF-IDF or optional local embeddings. Retrieved chunks
become evidence packs with citation strings. The extractive answerer selects evidence text directly;
the optional LLM path uses fake/OpenAI providers and strict citation containment validation.

For ResearchOps tasks, heuristic claim extraction feeds config suggestion. A bounded runner validates
configs before running a synthetic time-series forecasting experiment. Corpus indexing persists local
manifests, metadata, and chunk JSON. The workflow orchestrator ties these pieces together and writes
JSON/Markdown audit artifacts.

## Evaluation Summary

The demo evaluation fixtures cover retrieval hit rate and extractive answer grounding over a small
local corpus. The final demo generator writes current evaluation artifacts under
`reports/final_demo/`. These tests are intentionally small and deterministic; they demonstrate
grounding behavior rather than broad benchmark performance.

## Bounded Runner Results

The demo config `configs/time_series_demo.yaml` runs `mean_baseline` and `ridge` on a deterministic
synthetic sine series using chronological splitting. It writes config, metrics, predictions, and a
Markdown summary under the selected run directory.

## Key Limitations

- Claim extraction and config suggestion are heuristic.
- The bounded runner supports only one small forecasting task.
- Corpus indexes are simple JSON artifacts and not optimized for large collections.
- Optional embedding retrieval may require a local model download outside tests.
- LLM output is citation-validated but not guaranteed correct.

## Future Work

- Add richer parsers for tables and structured experiment sections.
- Expand the bounded runner to a few more safe task families.
- Add stronger evaluation datasets and regression reports.
- Improve corpus indexing performance without adding hidden external services.
- Add richer workflow diffing and run comparison views.

## Portfolio Positioning

This project demonstrates practical AI engineering restraint: grounded retrieval before generation,
bounded execution before automation, and audit artifacts before claims of intelligence. It is best
presented as a local ResearchOps foundation rather than a production-scale paper reproduction system.
