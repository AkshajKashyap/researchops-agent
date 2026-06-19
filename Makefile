.PHONY: install test lint api dashboard demo-eval demo-run demo-index demo-corpus-search demo-corpus-eval demo-workflow demo-all full-check ci-check health

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)
RUFF ?= $(shell if [ -x .venv/bin/ruff ]; then echo .venv/bin/ruff; else echo ruff; fi)
PYTEST ?= $(shell if [ -x .venv/bin/pytest ]; then echo .venv/bin/pytest; else echo pytest; fi)
RESEARCHOPS ?= $(shell if [ -x .venv/bin/researchops ]; then echo .venv/bin/researchops; else echo researchops; fi)

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTEST) -q

lint:
	$(RUFF) check src tests

api:
	uvicorn researchops_agent.api.main:app --reload

dashboard:
	streamlit run app/streamlit_app.py

demo-eval:
	$(RESEARCHOPS) eval --retriever tfidf --retrieval-cases examples/eval/retrieval_cases.json --answer-cases examples/eval/answer_cases.json --out-json reports/evaluation.json --out-md reports/evaluation.md

demo-run:
	$(RESEARCHOPS) run-config configs/time_series_demo.yaml --out-dir reports/runs

demo-index:
	$(RESEARCHOPS) index-corpus examples/docs --index-dir data/indexes/demo_corpus --retriever tfidf

demo-corpus-search:
	$(RESEARCHOPS) search-corpus data/indexes/demo_corpus "Which graph kernels were compared?"

demo-corpus-eval:
	$(RESEARCHOPS) eval-corpus --index-dir data/indexes/demo_corpus --cases examples/eval/corpus_retrieval_cases.json --out-json reports/corpus_eval.json --out-md reports/corpus_eval.md

demo-workflow: demo-index
	$(RESEARCHOPS) workflow data/indexes/demo_corpus "What experiment is described and can we run a local version?" --run-if-runnable --out-dir reports/workflows

full-check:
	$(RUFF) check src tests
	$(PYTEST) -q

demo-all:
	$(PYTHON) scripts/generate_demo_artifacts.py --out-dir reports/final_demo

ci-check: full-check demo-index demo-eval demo-run

health:
	$(RESEARCHOPS) health
