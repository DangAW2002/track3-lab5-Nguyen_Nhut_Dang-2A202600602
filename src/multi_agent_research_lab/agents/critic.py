import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Fact-checking and citation-validation agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        logger.info("CriticAgent starting...")
        llm = LLMClient()

        final_answer = state.final_answer or ""
        source_count = len(state.sources)
        errors = []

        # 1. Regex check for citation boundary consistency
        citations = re.findall(r"\[(\d+)\]", final_answer)
        for citation in citations:
            idx = int(citation)
            if idx < 1 or idx > source_count:
                err_msg = (
                    f"Citation [{idx}] is out of bounds (only {source_count} sources available)."
                )
                logger.warning(err_msg)
                errors.append(err_msg)

        # 2. LLM check for factual consistency and hallucinations
        sources_text = "\n\n".join(
            [
                f"Source [{idx + 1}]: {doc.title}\n"
                f"URL: {doc.url or 'N/A'}\n"
                f"Content: {doc.snippet}"
                for idx, doc in enumerate(state.sources)
            ]
        )

        system_prompt = (
            "You are a strict verification and fact-checking agent. Review the final "
            "report against the provided sources. Identify any claims, numbers, or "
            "facts that are unsupported or hallucinated. If everything is verified "
            "and supported, output 'PASSED'. Otherwise, list the errors clearly."
        )
        user_prompt = f"Sources:\n{sources_text}\n\nReport:\n{final_answer}"

        response = llm.complete(system_prompt, user_prompt)
        llm_feedback = response.content.strip()

        if "PASSED" not in llm_feedback.upper():
            logger.warning(f"LLM fact-checking failed: {llm_feedback}")
            errors.append(f"Fact-checking findings: {llm_feedback}")

        # Errors represent the latest review, not an append-only history. This lets
        # a successful re-review clear findings from the previous draft.
        state.errors = errors
        if errors:
            logger.warning(f"CriticAgent found {len(errors)} errors.")
        else:
            logger.info("CriticAgent passed successfully.")

        # Record AgentResult
        result = AgentResult(
            agent=AgentName.CRITIC,
            content=llm_feedback,
            metadata={
                "input_tokens": response.input_tokens or 0,
                "output_tokens": response.output_tokens or 0,
                "cost_usd": response.cost_usd or 0.0,
                "errors_found": len(errors),
            },
        )
        state.agent_results.append(result)
        state.add_trace_event(
            "critic",
            {
                "feedback": llm_feedback,
                "errors": errors,
                "input_tokens": response.input_tokens or 0,
                "output_tokens": response.output_tokens or 0,
                "cost_usd": response.cost_usd or 0.0,
            },
        )

        logger.info("CriticAgent complete.")
        return state
