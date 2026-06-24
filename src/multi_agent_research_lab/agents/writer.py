import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        logger.info("WriterAgent starting...")
        llm = LLMClient()

        sources_list = "\n".join(
            [
                f"Source [{idx + 1}]: {doc.title} - {doc.url}\nEvidence: {doc.snippet}"
                for idx, doc in enumerate(state.sources)
            ]
        )
        previous_draft = state.final_answer
        critic_feedback = "\n\n".join(state.errors)
        is_revision = bool(previous_draft and critic_feedback)

        system_prompt = (
            "You are an expert technical writer. Synthesize a comprehensive "
            f"research report for the audience: '{state.request.audience}'.\n"
            "Your output MUST be well-structured in markdown format with headings, "
            "bullet points, and paragraphs.\n"
            "You MUST integrate the research notes and analytical insights cleanly.\n"
            "You MUST use inline citations to back up claims (e.g., '[1]', '[2]') "
            "referencing the source index numbers provided below.\n"
            "Every factual claim must be directly supported by the supplied source evidence. "
            "Do not add plausible background knowledge that is absent from those snippets.\n"
            "Include a 'References' section at the very end of your report, listing "
            "all cited sources by their index numbers."
        )
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Research Notes:\n{state.research_notes or 'None'}\n\n"
            f"Analysis Notes:\n{state.analysis_notes or 'None'}\n\n"
            f"Sources:\n{sources_list}\n\n"
            f"Previous Draft:\n{previous_draft or 'None - create the first draft.'}\n\n"
            f"Critic Feedback:\n{critic_feedback or 'None - this is the first draft.'}\n\n"
            "If critic feedback is present, revise the previous draft point-by-point. "
            "Remove or qualify unsupported claims instead of inventing new evidence."
        )

        response = llm.complete(system_prompt, user_prompt)
        state.final_answer = response.content

        # Record AgentResult
        result = AgentResult(
            agent=AgentName.WRITER,
            content=state.final_answer,
            metadata={
                "input_tokens": response.input_tokens or 0,
                "output_tokens": response.output_tokens or 0,
                "cost_usd": response.cost_usd or 0.0,
                "revision": is_revision,
            },
        )
        state.agent_results.append(result)
        state.add_trace_event(
            "writer",
            {
                "answer": state.final_answer,
                "input_tokens": response.input_tokens or 0,
                "output_tokens": response.output_tokens or 0,
                "cost_usd": response.cost_usd or 0.0,
                "revision": is_revision,
                "critic_feedback_used": critic_feedback if is_revision else None,
            },
        )

        logger.info("WriterAgent complete.")
        return state
