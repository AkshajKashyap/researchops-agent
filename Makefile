.PHONY: install test lint api dashboard demo-eval demo-run demo-index demo-corpus-search demo-corpus-eval demo-workflow health

RESEARCHOPS ?= $(shell if [ -x .venv/bin/researchops ]; then echo .venv/bin/researchops; else echo researchops; fi)

install:
	python -m pip install -e .

test:
	pytest -q

lint:
	ruff check src tests

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

health:
	$(RESEARCHOPS) health
