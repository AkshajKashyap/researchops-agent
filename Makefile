.PHONY: install test lint api health

install:
python -m pip install -e .

test:
pytest -q

lint:
ruff check src tests

api:
uvicorn researchops_agent.api.main:app --reload

health:
researchops health
