from researchops_agent.schemas.experiment import ExperimentClaim

def test_experiment_claim_schema() -> None:
claim = ExperimentClaim(
task="time-series forecasting",
dataset="synthetic benchmark",
models=["Ridge", "LSTM"],
metrics=["RMSE", "MAE"],
main_claim="Chronological validation is more realistic than shuffled validation.",
)

assert claim.task == "time-series forecasting"
assert "RMSE" in claim.metrics

