"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


import logging
import os
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery, AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report, render_html_report
from multi_agent_research_lab.services.storage import LocalArtifactStore
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()
logger = logging.getLogger(__name__)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def run_baseline_logic(query: str) -> ResearchState:
    """Execute real single-agent baseline logic."""
    llm = LLMClient()
    searcher = SearchClient()

    # 1. Fetch sources
    docs = searcher.search(query, max_results=5)

    # 2. Synthesize answer in one model call
    sources_text = "\n\n".join(
        [f"Source [{idx + 1}]: {doc.title}\nContent: {doc.snippet}" for idx, doc in enumerate(docs)]
    )

    system_prompt = (
        "You are a technical research assistant. Summarize and answer the user query in a structured report. "
        "Format using markdown and cite sources using square brackets, e.g. [1], [2]."
    )
    user_prompt = f"Query: {query}\n\nSources:\n{sources_text}"

    response = llm.complete(system_prompt, user_prompt)

    # 3. Build state
    state = ResearchState(request=ResearchQuery(query=query))
    state.sources = docs
    state.final_answer = response.content

    result = AgentResult(
        agent=AgentName.WRITER,
        content=response.content,
        metadata={
            "input_tokens": response.input_tokens or 0,
            "output_tokens": response.output_tokens or 0,
            "cost_usd": response.cost_usd or 0.0,
        },
    )
    state.agent_results.append(result)
    state.add_trace_event(
        "baseline_run",
        {
            "duration_seconds": 0.0,
            "input_tokens": response.input_tokens or 0,
            "output_tokens": response.output_tokens or 0,
            "cost_usd": response.cost_usd or 0.0,
        },
    )
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a real single-agent baseline and record latency/cost/quality metrics."""
    _init()
    console.print(Panel(f"Running Single-Agent Baseline for query: [cyan]'{query}'[/cyan]..."))

    state, metrics = run_benchmark("baseline", query, run_baseline_logic)

    console.print(
        Panel.fit(
            f"[green]SUCCESS![/green]\n\n"
            f"Latency: {metrics.latency_seconds:.2f}s\n"
            f"Estimated Cost: ${metrics.estimated_cost_usd or 0.0:.5f}\n"
            f"Quality Score: {metrics.quality_score or 0.0}/10\n"
            f"Details: {metrics.notes}",
            title="Single-Agent Baseline Summary",
        )
    )
    console.print(Panel(state.final_answer or "", title="Baseline Final Answer"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow and single-agent baseline side-by-side to generate reports."""
    _init()
    console.print(
        Panel(
            f"Starting Multi-Agent Research System Benchmarking for query: [cyan]'{query}'[/cyan]..."
        )
    )

    # 1. Run Baseline
    console.print("[yellow]Executing Single-Agent Baseline...[/yellow]")
    baseline_state, baseline_metrics = run_benchmark("baseline", query, run_baseline_logic)

    # 2. Run Multi-Agent
    console.print("[yellow]Executing Multi-Agent LangGraph Workflow...[/yellow]")
    workflow = MultiAgentWorkflow()
    multi_agent_state, multi_agent_metrics = run_benchmark("multi-agent", query, workflow.run)

    # 3. Generate Reports
    metrics_list = [baseline_metrics, multi_agent_metrics]

    md_report = render_markdown_report(metrics_list)
    html_report = render_html_report(metrics_list, multi_agent_state, baseline_state)

    store = LocalArtifactStore()
    md_path = store.write_text("benchmark_report.md", md_report)
    html_path = store.write_text("benchmark_report.html", html_report)

    # Make absolute paths clickable
    abs_md_path = os.path.abspath(md_path)
    abs_html_path = os.path.abspath(html_path)

    console.print(
        Panel.fit(
            f"[green]BENCHMARK COMPLETE![/green]\n\n"
            f"• Markdown Report saved to: [cyan]file:///{abs_md_path.replace(chr(92), '/')}[/cyan]\n"
            f"• Interactive HTML Dashboard saved to: [cyan]file:///{abs_html_path.replace(chr(92), '/')}[/cyan]\n\n"
            f"[bold]Baseline vs Multi-Agent summary:[/bold]\n"
            f"- Latency: {baseline_metrics.latency_seconds:.2f}s vs {multi_agent_metrics.latency_seconds:.2f}s\n"
            f"- Cost: ${baseline_metrics.estimated_cost_usd or 0.0:.5f} vs ${multi_agent_metrics.estimated_cost_usd or 0.0:.5f}\n"
            f"- Quality: {baseline_metrics.quality_score or 0.0}/10 vs {multi_agent_metrics.quality_score or 0.0}/10",
            title="Comparison Summary",
        )
    )


if __name__ == "__main__":
    app()
