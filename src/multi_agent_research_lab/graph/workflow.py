"""LangGraph workflow skeleton."""

from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import ResearchQuery


import logging
from time import perf_counter
from langgraph.graph import StateGraph, END
from multi_agent_research_lab.agents import (
    SupervisorAgent,
    ResearcherAgent,
    AnalystAgent,
    WriterAgent,
    CriticAgent,
)

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

    def build(self) -> object:
        """Create a LangGraph graph."""
        workflow = StateGraph(ResearchState)

        # Wrap agent executions to measure latency and record tracing data
        def run_node(agent_name: str, agent_run_fn):
            def node_wrapper(state) -> ResearchState:
                logger.info(f"===> Node starting: {agent_name}")
                start_time = perf_counter()

                # Coerce state to ResearchState if it is passed as a dict
                if isinstance(state, dict):
                    state_obj = ResearchState(**state)
                else:
                    state_obj = state

                trace_count_before = len(state_obj.trace)
                new_state = agent_run_fn(state_obj)

                duration = perf_counter() - start_time
                logger.info(f"<=== Node finished: {agent_name} in {duration:.2f}s")

                # Append detailed trace log to trace list
                # Each trace event logs step name, iteration, duration, and key inputs/outputs
                trace_payload = {
                    "duration_seconds": duration,
                    "iteration": new_state.iteration,
                    "status": "completed",
                }

                # Capture snapshot data for visual trace in dashboard
                if agent_name == "researcher":
                    trace_payload["output_notes"] = new_state.research_notes
                    trace_payload["source_count"] = len(new_state.sources)
                elif agent_name == "analyst":
                    trace_payload["output_notes"] = new_state.analysis_notes
                elif agent_name == "writer":
                    trace_payload["output_notes"] = new_state.final_answer
                elif agent_name == "critic":
                    trace_payload["errors"] = list(new_state.errors)
                elif agent_name == "supervisor":
                    trace_payload["decision"] = (
                        new_state.route_history[-1] if new_state.route_history else "none"
                    )

                matching_event = next(
                    (
                        event
                        for event in reversed(new_state.trace[trace_count_before:])
                        if event.get("name") == agent_name
                    ),
                    None,
                )
                if matching_event is None:
                    new_state.add_trace_event(agent_name, trace_payload)
                else:
                    matching_event.setdefault("payload", {}).update(trace_payload)
                return new_state

            return node_wrapper

        # Add nodes
        workflow.add_node("supervisor", run_node("supervisor", self.supervisor.run))
        workflow.add_node("researcher", run_node("researcher", self.researcher.run))
        workflow.add_node("analyst", run_node("analyst", self.analyst.run))
        workflow.add_node("writer", run_node("writer", self.writer.run))
        workflow.add_node("critic", run_node("critic", self.critic.run))

        # Set entrypoint
        workflow.set_entry_point("supervisor")

        # Define conditional router with state coercion
        def router(state) -> str:
            if isinstance(state, dict):
                state_obj = ResearchState(**state)
            else:
                state_obj = state

            if not state_obj.route_history:
                return "supervisor"
            next_step = state_obj.route_history[-1]
            if next_step == "done":
                return END
            return next_step

        workflow.add_conditional_edges(
            "supervisor",
            router,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )

        # Route workers back to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        return workflow.compile()

    def run(self, state_or_query: ResearchState | str) -> ResearchState:
        """Execute the graph and return final state."""
        if isinstance(state_or_query, str):
            state = ResearchState(request=ResearchQuery(query=state_or_query))
        else:
            state = state_or_query

        compiled_graph = self.build()
        result = compiled_graph.invoke(state)

        # In case LangGraph invokes returned a dictionary
        if isinstance(result, dict):
            return ResearchState(**result)
        return result
