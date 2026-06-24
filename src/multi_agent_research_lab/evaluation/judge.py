"""Independent quality evaluator shared by every benchmark candidate."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

DIMENSION_WEIGHTS = {
    "relevance": 0.25,
    "completeness": 0.20,
    "groundedness": 0.35,
    "citation_accuracy": 0.20,
}


@dataclass(frozen=True)
class QualityEvaluation:
    dimensions: dict[str, float]
    overall_score: float
    findings: list[str]
    evaluator_cost_usd: float
    evaluator_tokens: int


def _clamp_score(value: object) -> float:
    try:
        return round(min(max(float(value), 0.0), 10.0), 1)
    except (TypeError, ValueError):
        return 0.0


def _extract_json(content: str) -> dict[str, object]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Evaluator did not return a JSON object")
    return json.loads(cleaned[start : end + 1])


def evaluate_quality(query: str, state: ResearchState) -> QualityEvaluation:
    """Score one answer independently against its query and available evidence."""
    sources_text = "\n\n".join(
        [
            f"Source [{index + 1}]\n"
            f"Title: {source.title}\n"
            f"URL: {source.url or 'N/A'}\n"
            f"Evidence: {source.snippet}"
            for index, source in enumerate(state.sources)
        ]
    )
    system_prompt = (
        "You are an independent benchmark judge. Evaluate only the supplied answer; "
        "do not compare it with another system and do not reward agent count, verbosity, "
        "or internal self-critique. Use the same rubric for every candidate.\n"
        "Score each dimension from 0 to 10:\n"
        "- relevance: directly answers the requested task without distraction.\n"
        "- completeness: covers the important requested components at useful depth.\n"
        "- groundedness: factual claims are supported by the supplied evidence; "
        "unsupported specificity lowers the score.\n"
        "- citation_accuracy: citations point to valid sources and support nearby claims.\n"
        "Return JSON only with keys relevance, completeness, groundedness, "
        "citation_accuracy, and findings. Findings must be an array of at most five "
        "short evidence-based observations."
    )
    user_prompt = (
        f"Query:\n{query}\n\n"
        f"Available evidence:\n{sources_text or 'No sources provided.'}\n\n"
        f"Candidate answer:\n{state.final_answer or 'No answer generated.'}"
    )

    response = LLMClient().complete(system_prompt, user_prompt)
    payload = _extract_json(response.content)
    dimensions = {name: _clamp_score(payload.get(name)) for name in DIMENSION_WEIGHTS}
    overall_score = round(
        sum(dimensions[name] * weight for name, weight in DIMENSION_WEIGHTS.items()),
        1,
    )
    raw_findings = payload.get("findings", [])
    findings = (
        [str(item) for item in raw_findings[:5]]
        if isinstance(raw_findings, list)
        else [str(raw_findings)]
    )
    return QualityEvaluation(
        dimensions=dimensions,
        overall_score=overall_score,
        findings=findings,
        evaluator_cost_usd=response.cost_usd or 0.0,
        evaluator_tokens=(response.input_tokens or 0) + (response.output_tokens or 0),
    )
