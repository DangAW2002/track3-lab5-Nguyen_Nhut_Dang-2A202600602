from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_correctly() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    new_state = SupervisorAgent().run(state)
    assert new_state.iteration == 1
    assert "researcher" in new_state.route_history


def _completed_draft_state(latest_agent: AgentName) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.sources = [SourceDocument(title="Source", url="https://example.com", snippet="Evidence")]
    state.research_notes = "Research notes [1]"
    state.analysis_notes = "Analysis [1]"
    state.final_answer = "Draft [1]"
    state.agent_results.append(AgentResult(agent=latest_agent, content="result"))
    return state


def test_supervisor_routes_failed_critic_to_writer_revision() -> None:
    state = _completed_draft_state(AgentName.CRITIC)
    state.errors = ["Unsupported claim"]

    new_state = SupervisorAgent().run(state)

    assert new_state.route_history[-1] == "writer"
    assert new_state.final_answer == "Draft [1]"


def test_supervisor_rechecks_revised_writer_draft() -> None:
    state = _completed_draft_state(AgentName.WRITER)
    state.errors = ["Feedback awaiting re-review"]

    new_state = SupervisorAgent().run(state)

    assert new_state.route_history[-1] == "critic"


def test_supervisor_finishes_after_clean_critic_review() -> None:
    state = _completed_draft_state(AgentName.CRITIC)

    new_state = SupervisorAgent().run(state)

    assert new_state.route_history[-1] == "done"
