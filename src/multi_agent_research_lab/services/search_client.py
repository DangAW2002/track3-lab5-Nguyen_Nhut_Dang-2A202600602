"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument


import requests
import logging
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client using Tavily or high-quality local mocks."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        settings = get_settings()
        api_key = settings.tavily_api_key

        if not api_key:
            logger.warning("TAVILY_API_KEY not configured. Falling back to local mock search.")
            return self._mock_search(query, max_results)

        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append(
                    SourceDocument(
                        title=item.get("title", "No Title"),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                        metadata={"score": item.get("score", 0.0)},
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Tavily search API failed: {e}. Falling back to mock search.")
            return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        query_lower = query.lower()
        if "graphrag" in query_lower:
            return [
                SourceDocument(
                    title="GraphRAG: Graph-based Retrieval-Augmented Generation",
                    url="https://arxiv.org/abs/2404.16130",
                    snippet="GraphRAG combines vector search with knowledge graphs to capture both local and global semantics. It structures unstructured text into an entity-relation graph and uses LLMs to generate community summaries.",
                ),
                SourceDocument(
                    title="Microsoft GraphRAG Query Engine",
                    url="https://github.com/microsoft/graphrag",
                    snippet="Microsoft's GraphRAG utilizes hierarchical community detection (Leiden algorithm) and local/global search query engines to answer complex multi-hop questions across the corpus.",
                ),
            ][:max_results]
        elif "guardrail" in query_lower:
            return [
                SourceDocument(
                    title="Guardrails in AI Systems and Workflows",
                    url="https://guardrails.ai",
                    snippet="Guardrails are safety mechanisms implemented around LLM inputs and outputs to enforce constraints, check for hallucinations, validate JSON schemas, and avoid undesirable behavior.",
                ),
                SourceDocument(
                    title="LLM Safety and Alignment Practices",
                    url="https://openai.com/safety",
                    snippet="Best practices for LLM safety suggest adding semantic monitors, rate limiters, iteration bounds, and fallback options to avoid runaway agent behaviors.",
                ),
            ][:max_results]
        elif "multi-agent" in query_lower:
            return [
                SourceDocument(
                    title="Multi-Agent Workflows: Supervisor and Router Patterns",
                    url="https://langchain.com/blog/multi-agent",
                    snippet="Multi-agent workflows split complex tasks among specialized agents. A Supervisor agent orchestrates the workflow, routing tasks sequentially to Researcher, Analyst, and Writer agents.",
                ),
                SourceDocument(
                    title="Building Effective Agents by Anthropic",
                    url="https://anthropic.com/research/agents",
                    snippet="Anthropic discusses patterns for agentic workflows including orchestrator-workers, chains, and routing loops, emphasizing state checking, validation, and loop prevention.",
                ),
            ][:max_results]
        else:
            return [
                SourceDocument(
                    title=f"General Web Search: {query}",
                    url="https://example.com/search",
                    snippet=f"Mock search result snippet containing relevant context for: '{query}'. This covers general aspects of multi-agent and single-agent performance.",
                )
            ][:max_results]
