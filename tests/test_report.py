from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.report import render_html_report, render_markdown_report


def test_report_renders_markdown() -> None:
    report = render_markdown_report([BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)])
    assert "Benchmark Report" in report
    assert "baseline" in report


def test_html_report_contains_full_trace_dashboard() -> None:
    baseline = ResearchState(request=ResearchQuery(query="Compare agent workflows"))
    multi = ResearchState(request=ResearchQuery(query="Compare agent workflows"))
    multi.record_route("researcher")
    multi.add_trace_event(
        "researcher",
        {
            "duration_seconds": 1.25,
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.0001,
            "notes": "Evidence collected",
        },
    )
    metrics = [
        BenchmarkMetrics(
            run_name="baseline",
            latency_seconds=1.0,
            quality_score=7.0,
            quality_dimensions={"relevance": 7.0, "groundedness": 6.0},
        ),
        BenchmarkMetrics(
            run_name="multi-agent",
            latency_seconds=2.0,
            quality_score=8.0,
            quality_dimensions={"relevance": 8.0, "groundedness": 8.0},
        ),
    ]

    report = render_html_report(metrics, multi, baseline)

    assert "Trace Explorer" in report
    assert "Evidence collected" in report
    assert "Independent Quality Rubric" in report
    assert "Groundedness" in report
    assert 'class="shell"' in report
    assert "max-width: 1200px" not in report
