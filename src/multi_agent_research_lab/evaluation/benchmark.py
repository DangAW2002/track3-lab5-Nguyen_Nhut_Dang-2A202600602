"""Fair benchmark runner for single-agent and multi-agent systems."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.judge import QualityEvaluation, evaluate_quality

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]
Evaluator = Callable[[str, ResearchState], QualityEvaluation]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    evaluator: Evaluator = evaluate_quality,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure execution and apply the same independent evaluator to every run."""
    logger.info("Running benchmark for '%s' with query: '%s'", run_name, query)
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    if state.trace:
        state.trace[-1].setdefault("payload", {})["run_latency_seconds"] = latency

    run_cost_usd = sum(
        float(result.metadata.get("cost_usd", 0.0)) for result in state.agent_results
    )
    total_tokens = sum(
        int(result.metadata.get("input_tokens", 0)) + int(result.metadata.get("output_tokens", 0))
        for result in state.agent_results
    )
    cited_sources = len(set(re.findall(r"\[(\d+)\]", state.final_answer or "")))

    # Evaluation happens after execution timing and is accounted for separately.
    evaluation = evaluator(query, state)
    notes = (
        f"Iterations: {state.iteration}. "
        f"Sources: {len(state.sources)} (cited: {cited_sources}). "
        f"Run tokens: {total_tokens}. "
        f"Internal critic findings: {len(state.errors)}. "
        "Quality uses the same independent judge rubric for every run."
    )
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=run_cost_usd,
        quality_score=evaluation.overall_score,
        quality_dimensions=evaluation.dimensions,
        evaluation_findings=evaluation.findings,
        evaluator_cost_usd=evaluation.evaluator_cost_usd,
        evaluator_tokens=evaluation.evaluator_tokens,
        notes=notes,
    )
    logger.info(
        "Benchmark '%s' finished: latency=%.2fs, run_cost=$%.5f, quality=%.1f/10",
        run_name,
        latency,
        run_cost_usd,
        evaluation.overall_score,
    )
    return state, metrics
