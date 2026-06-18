import json
from pathlib import Path

from researchops_agent.schemas.evaluation import AnswerEvalCase, RetrievalEvalCase


def load_retrieval_cases(path: str) -> list[RetrievalEvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("retrieval cases must be a JSON list")
    return [RetrievalEvalCase.model_validate(item) for item in data]


def load_answer_cases(path: str) -> list[AnswerEvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("answer cases must be a JSON list")
    return [AnswerEvalCase.model_validate(item) for item in data]
