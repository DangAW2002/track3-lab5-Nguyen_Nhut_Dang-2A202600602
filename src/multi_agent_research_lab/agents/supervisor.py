import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        settings = get_settings()
        max_iters = settings.max_iterations

        logger.info(
            "Supervisor running. Iteration: %s/%s. Errors: %s",
            state.iteration,
            max_iters,
            len(state.errors),
        )

        if state.iteration >= max_iters:
            logger.warning("Max iterations reached. Ending workflow.")
            state.record_route("done")
            state.add_trace_event(
                "supervisor",
                {"decision": "done", "reason": "Max iterations reached"},
            )
            return state

        latest_agent = state.agent_results[-1].agent.value if state.agent_results else None

        # Routing decision logic
        if not state.sources or not state.research_notes:
            next_agent = "researcher"
            reason = "Missing search sources or research notes"
        elif not state.analysis_notes:
            next_agent = "analyst"
            reason = "Missing analysis notes"
        elif not state.final_answer:
            next_agent = "writer"
            reason = "Missing final answer report"
        elif latest_agent == "writer":
            next_agent = "critic"
            reason = "Reviewing the latest writer draft for grounding and citations"
        elif latest_agent == "critic" and state.errors:
            next_agent = "writer"
            reason = "Revising the draft using the latest critic feedback"
        else:
            next_agent = "done"
            reason = "Latest critic review passed with no outstanding errors"

        state.record_route(next_agent)
        state.add_trace_event(
            "supervisor",
            {
                "decision": next_agent,
                "reason": reason,
                "iteration": state.iteration,
                "latest_agent": latest_agent,
                "outstanding_errors": len(state.errors),
            },
        )
        logger.info(f"Supervisor decided to route to: {next_agent} because: {reason}")
        return state
