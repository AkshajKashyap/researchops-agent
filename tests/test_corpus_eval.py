from researchops_agent.corpus.search import build_index_for_corpus
from researchops_agent.evaluation.corpus_eval import evaluate_corpus_retrieval
from researchops_agent.evaluation.report import build_evaluation_report
from researchops_agent.schemas.evaluation import RetrievalEvalCase


def _index(tmp_path) -> str:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "forecast.md").write_text("Ridge and LSTM use RMSE.", encoding="utf-8")
    (docs / "graph.md").write_text("WL graph kernels use accuracy.", encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index_for_corpus(str(docs), str(index_dir))
    return str(index_dir)


def test_corpus_retrieval_eval_hits_expected_substrings(tmp_path) -> None:
    case = RetrievalEvalCase(
        case_id="hit",
        document_path="ignored.md",
        query="graph kernels accuracy",
        expected_substrings=["WL graph kernels"],
    )

    result = evaluate_corpus_retrieval(_index(tmp_path), [case], top_k=1)[0]

    assert result.hit is True
    assert result.matched_substrings == ["WL graph kernels"]


def test_corpus_retrieval_eval_misses_unsupported_substrings(tmp_path) -> None:
    case = RetrievalEvalCase(
        case_id="miss",
        document_path="ignored.md",
        query="Ridge RMSE",
        expected_substrings=["Transformer"],
    )

    result = evaluate_corpus_retrieval(_index(tmp_path), [case], top_k=1)[0]

    assert result.hit is False


def test_corpus_eval_report_summary_is_correct(tmp_path) -> None:
    cases = [
        RetrievalEvalCase(
            case_id="hit",
            document_path="ignored.md",
            query="Ridge RMSE",
            expected_substrings=["Ridge"],
        )
    ]

    results = evaluate_corpus_retrieval(_index(tmp_path), cases, top_k=1)
    report = build_evaluation_report(results, [])

    assert report.summary.retrieval_cases == 1
    assert report.summary.retrieval_hits == 1
    assert report.summary.answer_cases == 0
