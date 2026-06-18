from researchops_agent.evaluation.answer_eval import evaluate_answers
from researchops_agent.schemas.evaluation import AnswerEvalCase


def _document(tmp_path, text: str) -> str:
    path = tmp_path / "note.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_answer_eval_passes_when_substring_is_supported_by_answer_or_evidence(tmp_path) -> None:
    document_path = _document(
        tmp_path,
        "We compare Ridge and LSTM. Results show the LSTM improves RMSE.",
    )
    case = AnswerEvalCase(
        case_id="supported",
        document_path=document_path,
        query="Ridge LSTM RMSE",
        expected_answer_substrings=["Ridge", "RMSE"],
    )

    results = evaluate_answers([case])

    assert results[0].passed is True
    assert results[0].matched_substrings == ["Ridge", "RMSE"]


def test_answer_eval_passes_abstention_case_when_answer_abstains(tmp_path) -> None:
    document_path = _document(tmp_path, "We compare Ridge using RMSE.")
    case = AnswerEvalCase(
        case_id="abstain",
        document_path=document_path,
        query="What optimizer learning rate schedule was used?",
        should_abstain=True,
    )

    results = evaluate_answers([case])

    assert results[0].passed is True
    assert results[0].abstained is True


def test_answer_eval_fails_when_expected_substring_is_unsupported(tmp_path) -> None:
    document_path = _document(tmp_path, "We compare Ridge using RMSE.")
    case = AnswerEvalCase(
        case_id="unsupported",
        document_path=document_path,
        query="What does Ridge use?",
        expected_answer_substrings=["Transformer"],
    )

    results = evaluate_answers([case])

    assert results[0].passed is False
    assert results[0].matched_substrings == []
