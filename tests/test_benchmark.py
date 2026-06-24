from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.judge import QualityEvaluation


def test_benchmark_uses_independent_evaluator_without_penalizing_state_errors() -> None:
    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        state.final_answer = "Structured answer [1]"
        state.errors = ["Internal critic finding"]
        return state

    def evaluator(query: str, state: ResearchState) -> QualityEvaluation:
        assert query == state.request.query
        return QualityEvaluation(
            dimensions={
                "relevance": 8.0,
                "completeness": 7.0,
                "groundedness": 9.0,
                "citation_accuracy": 8.0,
            },
            overall_score=8.2,
            findings=["Same rubric applied"],
            evaluator_cost_usd=0.0001,
            evaluator_tokens=50,
        )

    _, metrics = run_benchmark("candidate", "Explain the workflow", runner, evaluator)

    assert metrics.quality_score == 8.2
    assert metrics.quality_dimensions["groundedness"] == 9.0
    assert metrics.evaluator_cost_usd == 0.0001
    assert "Internal critic findings: 1" in metrics.notes
