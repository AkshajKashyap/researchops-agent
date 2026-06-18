from researchops_agent.runner.config_builder import suggest_experiment_config
from researchops_agent.schemas.answer import ExtractiveAnswer
from researchops_agent.schemas.experiment import ExperimentClaim, ResearchReport


def build_research_report(
    query: str, answer: ExtractiveAnswer, claims: list[ExperimentClaim]
) -> ResearchReport:
    report_claims = [] if answer.abstained else claims
    suggested_config = None if answer.abstained else suggest_experiment_config(report_claims)

    return ResearchReport(
        query=query,
        answer=answer.answer,
        citations=answer.citations,
        claims=report_claims,
        suggested_config=suggested_config,
        abstained=answer.abstained,
        reason=answer.reason,
    )
