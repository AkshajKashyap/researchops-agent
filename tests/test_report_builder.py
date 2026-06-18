from researchops_agent.agents.report_builder import build_research_report
from researchops_agent.schemas.answer import ExtractiveAnswer
from researchops_agent.schemas.experiment import ExperimentClaim


def test_builds_report_from_answer_and_claims() -> None:
    answer = ExtractiveAnswer(
        query="What experiment is described?",
        answer="We compare Ridge and LSTM.",
        citations=["data/raw/paper.md#chunk=0"],
        abstained=False,
    )
    claims = [
        ExperimentClaim(
            task="classification",
            models=["Ridge", "LSTM"],
            metrics=["accuracy"],
            main_claim="We compare Ridge and LSTM.",
        )
    ]

    report = build_research_report(answer.query, answer, claims)

    assert report.query == answer.query
    assert report.answer == "We compare Ridge and LSTM."
    assert report.citations == ["data/raw/paper.md#chunk=0"]
    assert report.claims == claims


def test_includes_suggested_config_when_claims_exist() -> None:
    answer = ExtractiveAnswer(
        query="query",
        answer="Results show Ridge improves RMSE.",
        citations=["citation"],
        abstained=False,
    )
    claim = ExperimentClaim(main_claim="Results show Ridge improves RMSE.", models=["Ridge"])

    report = build_research_report("query", answer, [claim])

    assert report.suggested_config is not None
    assert report.suggested_config.models == ["Ridge"]


def test_produces_no_suggested_config_when_answer_abstained() -> None:
    answer = ExtractiveAnswer(
        query="query",
        answer="I do not have enough retrieved evidence to answer this question.",
        citations=[],
        abstained=True,
        reason="Insufficient retrieved evidence.",
    )
    claim = ExperimentClaim(main_claim="Results show Ridge improves RMSE.")

    report = build_research_report("query", answer, [claim])

    assert report.claims == []
    assert report.suggested_config is None
    assert report.abstained is True
    assert report.reason == "Insufficient retrieved evidence."
