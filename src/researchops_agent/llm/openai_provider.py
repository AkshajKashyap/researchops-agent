import json
import os
from typing import Any

from researchops_agent.schemas.answer import EvidencePack
from researchops_agent.schemas.llm import GroundedLLMAnswer, LLMProviderConfig


class OpenAIProvider:
    def generate_grounded_answer(
        self,
        query: str,
        evidence: EvidencePack,
        config: LLMProviderConfig,
    ) -> GroundedLLMAnswer:
        model = config.model or os.getenv("OPENAI_MODEL")
        if not model:
            raise ValueError("OpenAI model is required. Set config.model or OPENAI_MODEL.")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise ValueError("OpenAI SDK is not installed. Install with the llm extra.") from exc

        prompt = self._build_prompt(query, evidence)
        try:
            client = OpenAI()
            response = client.responses.create(
                model=model,
                input=prompt,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            )
            payload = self._parse_response(response)
        except Exception as exc:
            raise RuntimeError(f"OpenAI provider failed: {exc.__class__.__name__}") from exc

        return GroundedLLMAnswer(
            query=query,
            answer=str(payload.get("answer", "")),
            citations=list(payload.get("citations", [])),
            abstained=bool(payload.get("abstained", False)),
            reason=payload.get("reason"),
            provider="openai",
            model=model,
        )

    @staticmethod
    def _build_prompt(query: str, evidence: EvidencePack) -> str:
        evidence_lines = [
            f"CITATION: {item.citation}\nTEXT: {item.text}"
            for item in evidence.items
        ]
        citations = [item.citation for item in evidence.items]
        return (
            "You answer questions using only the evidence below.\n"
            "You may only cite citations listed in the evidence.\n"
            "If the evidence is insufficient, abstain.\n"
            "Return strict JSON with keys: answer, citations, abstained, reason.\n\n"
            f"Allowed citations: {json.dumps(citations)}\n"
            f"Question: {query}\n\n"
            "Evidence:\n"
            + "\n\n".join(evidence_lines)
        )

    @staticmethod
    def _parse_response(response: Any) -> dict[str, Any]:
        text = getattr(response, "output_text", None)
        if not text:
            try:
                text = response.output[0].content[0].text
            except Exception as exc:
                raise ValueError("OpenAI response did not contain text output") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("OpenAI response JSON must be an object")
        return payload
