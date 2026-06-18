.PHONY: install test lint api dashboard demo-eval demo-run health

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
	researchops eval --retriever tfidf --retrieval-cases examples/eval/retrieval_cases.json --answer-cases examples/eval/answer_cases.json --out-json reports/evaluation.json --out-md reports/evaluation.md

demo-run:
	researchops run-config configs/time_series_demo.yaml --out-dir reports/runs

health:
	researchops health
