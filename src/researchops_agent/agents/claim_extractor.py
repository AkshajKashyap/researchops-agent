import re

from researchops_agent.schemas.answer import EvidencePack
from researchops_agent.schemas.experiment import ExperimentClaim

CLAIM_CUES = (
    "we compare",
    "we evaluate",
    "we train",
    "results show",
    "outperforms",
    "achieves",
    "accuracy",
    "f1",
    "rmse",
    "mae",
    "auc",
    "baseline",
    "dataset",
    "validation",
)
METRIC_PATTERNS = {
    "accuracy": r"\baccuracy\b",
    "F1": r"\bf1(?:-score)?\b",
    "RMSE": r"\brmse\b",
    "MAE": r"\bmae\b",
    "AUC": r"\bauc\b",
}
MODEL_PATTERNS = (
    "Random Forest",
    "Logistic Regression",
    "Linear Regression",
    "Ridge",
    "LSTM",
    "Transformer",
    "BERT",
    "SVM",
    "XGBoost",
    "baseline",
)
RISK_CUES = ("not reproducible", "missing seed", "no seed", "not specified", "unclear")


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _has_claim_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in CLAIM_CUES)


def _extract_metrics(text: str) -> list[str]:
    metrics: list[str] = []
    for metric, pattern in METRIC_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            metrics.append(metric)
    return metrics


def _extract_models(text: str) -> list[str]:
    models: list[str] = []
    for model in MODEL_PATTERNS:
        if re.search(rf"\b{re.escape(model)}\b", text, flags=re.IGNORECASE):
            models.append(model)
    return models


def _extract_dataset(text: str) -> str | None:
    patterns = (
        r"\b(?:on|using|from)\s+(?:the\s+)?([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,3})\s+dataset\b",
        r"\bdataset\s+(?:is|was|:)?\s*(?:the\s+)?([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,3})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _extract_risk(text: str) -> str | None:
    lowered = text.lower()
    if any(cue in lowered for cue in RISK_CUES):
        return "Reproducibility details may be incomplete in the retrieved evidence."
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def extract_experiment_claims(pack: EvidencePack, max_claims: int = 5) -> list[ExperimentClaim]:
    if max_claims <= 0:
        raise ValueError("max_claims must be positive")

    claims: list[ExperimentClaim] = []
    for item in pack.items:
        candidates = _sentences(item.text) or [item.text.strip()]
        for candidate in candidates:
            if not candidate or not _has_claim_cue(candidate):
                continue

            claims.append(
                ExperimentClaim(
                    dataset=_extract_dataset(candidate),
                    models=_dedupe(_extract_models(candidate)),
                    metrics=_dedupe(_extract_metrics(candidate)),
                    main_claim=candidate,
                    evidence_citations=[item.citation],
                    reproducibility_risk=_extract_risk(candidate),
                )
            )
            if len(claims) >= max_claims:
                return claims

    return claims
