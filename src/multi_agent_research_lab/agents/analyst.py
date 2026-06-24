"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


import logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.core.schemas import AgentName, AgentResult

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        logger.info("AnalystAgent starting...")
        llm = LLMClient()

        system_prompt = (
            "You are a critical research analyst. Examine the provided research notes for the query. "
            "Your task is to: \n"
            "1. Extract the key claims made.\n"
            "2. Identify any divergent viewpoints or potential bias.\n"
            "3. Assess the strength of evidence provided by the sources.\n"
            "4. Structure your response clearly using bullet points and preserve relevant source citations, e.g. [1]."
        )
        user_prompt = f"Query: {state.request.query}\n\nResearch Notes:\n{state.research_notes or 'No research notes found.'}"

        response = llm.complete(system_prompt, user_prompt)
        state.analysis_notes = response.content

        # Record AgentResult
        result = AgentResult(
            agent=AgentName.ANALYST,
            content=state.analysis_notes,
            metadata={
                "input_tokens": response.input_tokens or 0,
                "output_tokens": response.output_tokens or 0,
                "cost_usd": response.cost_usd or 0.0,
            },
        )
        state.agent_results.append(result)
        state.add_trace_event(
            "analyst",
            {
                "analysis": state.analysis_notes,
                "input_tokens": response.input_tokens or 0,
                "output_tokens": response.output_tokens or 0,
                "cost_usd": response.cost_usd or 0.0,
            },
        )

        logger.info("AnalystAgent complete.")
        return state
