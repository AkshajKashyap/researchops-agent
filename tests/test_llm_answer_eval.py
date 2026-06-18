from researchops_agent.evaluation.llm_answer_eval import evaluate_llm_answers
from researchops_agent.schemas.evaluation import AnswerEvalCase


def test_llm_answer_eval_works_with_fake_provider(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("Ridge and LSTM are compared using RMSE.", encoding="utf-8")
    case = AnswerEvalCase(
        case_id="answer",
        document_path=str(document),
        query="What models were compared?",
        expected_answer_substrings=["Ridge"],
    )

    results = evaluate_llm_answers([case], provider="fake")

    assert results[0].passed is True
    assert results[0].citations
