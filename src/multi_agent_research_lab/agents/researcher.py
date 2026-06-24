"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


import logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.core.schemas import AgentName, AgentResult

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        logger.info("ResearcherAgent starting...")
        llm = LLMClient()
        searcher = SearchClient()

        # Step 1: Query LLM to generate a search term from the user's research query
        system_prompt = "You are a research assistant. Based on the user's query, return 1-2 distinct search terms separated by newlines to retrieve relevant data. Do not include any other text."
        user_prompt = f"Query: {state.request.query}"

        response = llm.complete(system_prompt, user_prompt)
        search_terms = [line.strip() for line in response.content.split("\n") if line.strip()]
        if not search_terms:
            search_terms = [state.request.query]

        logger.info(f"Generated search terms: {search_terms}")

        # Step 2: Run search for each term
        all_sources = list(state.sources)
        urls = {src.url for src in all_sources if src.url}

        for term in search_terms[:2]:
            docs = searcher.search(term, max_results=state.request.max_sources)
            for doc in docs:
                if doc.url not in urls:
                    urls.add(doc.url)
                    all_sources.append(doc)

        state.sources = all_sources[: state.request.max_sources]
        logger.info(f"Retrieved {len(state.sources)} sources.")

        # Step 3: Synthesize research notes
        sources_text = "\n\n".join(
            [
                f"Source [{idx + 1}]: {doc.title}\nURL: {doc.url}\nContent: {doc.snippet}"
                for idx, doc in enumerate(state.sources)
            ]
        )

        notes_system_prompt = (
            "You are a meticulous researcher. Summarize the provided sources into structured research notes "
            "related to the query. For every fact, claim, or summary, you MUST explicitly cite the source number using "
            "square brackets, e.g., [1] or [2]. Keep notes highly objective and detailed."
        )
        notes_user_prompt = f"Query: {state.request.query}\n\nSources:\n{sources_text}"

        notes_response = llm.complete(notes_system_prompt, notes_user_prompt)
        state.research_notes = notes_response.content

        # Record AgentResult
        total_input_tokens = (response.input_tokens or 0) + (notes_response.input_tokens or 0)
        total_output_tokens = (response.output_tokens or 0) + (notes_response.output_tokens or 0)
        total_cost = (response.cost_usd or 0.0) + (notes_response.cost_usd or 0.0)

        result = AgentResult(
            agent=AgentName.RESEARCHER,
            content=state.research_notes,
            metadata={
                "search_queries": search_terms,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "cost_usd": total_cost,
                "source_count": len(state.sources),
            },
        )
        state.agent_results.append(result)
        state.add_trace_event(
            "researcher",
            {
                "notes": state.research_notes,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "cost_usd": total_cost,
            },
        )

        logger.info("ResearcherAgent complete.")
        return state
