"""Benchmark report rendering."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from typing import Any

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


def _safe(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _format_text(value: Any) -> str:
    return _safe(value).replace("\n", "<br>")


def _total_tokens(state: ResearchState) -> int:
    return sum(
        int(result.metadata.get("input_tokens", 0)) + int(result.metadata.get("output_tokens", 0))
        for result in state.agent_results
    )


def _citation_count(state: ResearchState) -> int:
    return len(set(re.findall(r"\[(\d+)\]", state.final_answer or "")))


def _ratio(current: float, baseline: float) -> str:
    return "N/A" if baseline <= 0 else f"{current / baseline:.1f}x"


def _dimension(metric: BenchmarkMetrics, name: str) -> float:
    return float(metric.quality_dimensions.get(name, 0.0))


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render a concise benchmark summary with observed trade-offs."""
    lines = [
        "# Multi-Agent System Benchmark Report",
        "",
        "Measured comparison of the **Single-Agent Baseline** and "
        "the **Multi-Agent Research System**.",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality (0-10) | Run details |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "N/A" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.5f}"
        quality = "N/A" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| **{item.run_name}** | {item.latency_seconds:.2f} | "
            f"{cost} | {quality} | {item.notes} |"
        )
        if item.quality_dimensions:
            lines.append(
                f"| ↳ rubric | relevance {_dimension(item, 'relevance'):.1f} | "
                f"completeness {_dimension(item, 'completeness'):.1f} | "
                f"groundedness {_dimension(item, 'groundedness'):.1f} | "
                f"citation accuracy {_dimension(item, 'citation_accuracy'):.1f} |"
            )
    if len(metrics) >= 2:
        baseline, multi = metrics[0], metrics[1]
        lines.extend(
            [
                "",
                "## Observed Trade-offs",
                "",
                f"- Multi-agent latency: **{_ratio(multi.latency_seconds, baseline.latency_seconds)}** baseline.",
                f"- Multi-agent cost: **{_ratio(multi.estimated_cost_usd or 0, baseline.estimated_cost_usd or 0)}** baseline.",
                f"- Quality delta: **{(multi.quality_score or 0) - (baseline.quality_score or 0):+.1f} points**.",
                "- Both candidates use the same independent evidence-grounded judge. "
                "Internal critic findings are diagnostic and do not directly change quality.",
                "- Open the HTML dashboard for routing, per-agent timing, token usage, sources, outputs, and raw traces.",
            ]
        )
    return "\n".join(lines) + "\n"


def render_html_report(
    metrics: list[BenchmarkMetrics],
    multi_agent_state: ResearchState,
    baseline_state: ResearchState,
) -> str:
    """Generate a self-contained full-screen benchmark and trace dashboard."""
    metric_by_name = {item.run_name.lower(): item for item in metrics}
    baseline_metric = metric_by_name.get("baseline", metrics[0])
    multi_metric = metric_by_name.get("multi-agent", metrics[-1])
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    cards = []
    runs = [
        ("baseline", "Single-Agent Baseline", baseline_metric, baseline_state, "rose"),
        ("multi-agent", "Multi-Agent Workflow", multi_metric, multi_agent_state, "mint"),
    ]
    for slug, title, metric, state, color in runs:
        input_tokens = sum(int(r.metadata.get("input_tokens", 0)) for r in state.agent_results)
        output_tokens = sum(int(r.metadata.get("output_tokens", 0)) for r in state.agent_results)
        input_percent = 100 * input_tokens / max(input_tokens + output_tokens, 1)
        cards.append(
            f"""
            <article class="run-card {color}">
              <header>
                <div><span class="eyebrow">{_safe(slug)}</span><h3>{_safe(title)}</h3></div>
                <div class="quality">{metric.quality_score or 0:.1f}<small>/10</small></div>
              </header>
              <div class="metrics">
                <div><span>Latency</span><strong>{metric.latency_seconds:.2f}s</strong></div>
                <div><span>Estimated cost</span><strong>${metric.estimated_cost_usd or 0:.5f}</strong></div>
                <div><span>Total tokens</span><strong>{_total_tokens(state):,}</strong></div>
                <div><span>LLM calls</span><strong>{len(state.agent_results)}</strong></div>
                <div><span>Sources / cited</span><strong>{len(state.sources)} / {_citation_count(state)}</strong></div>
                <div><span>Internal findings</span><strong>{len(state.errors)}</strong></div>
              </div>
              <div class="token-bar"><span style="width:{input_percent:.1f}%"></span></div>
              <p class="caption">Input {input_tokens:,} · Output {output_tokens:,}</p>
              <p class="caption">Judge: {metric.evaluator_tokens:,} tokens · ${metric.evaluator_cost_usd:.5f} (excluded from run cost)</p>
              <p class="notes">{_safe(metric.notes)}</p>
            </article>
            """
        )

    score_rows = []
    for key, label, weight in [
        ("relevance", "Relevance", "25%"),
        ("completeness", "Completeness", "20%"),
        ("groundedness", "Groundedness", "35%"),
        ("citation_accuracy", "Citation accuracy", "20%"),
    ]:
        baseline_score = _dimension(baseline_metric, key)
        multi_score = _dimension(multi_metric, key)
        winner = (
            "Tie"
            if baseline_score == multi_score
            else ("Multi-agent" if multi_score > baseline_score else "Baseline")
        )
        score_rows.append(
            f"""
            <tr>
              <td><strong>{label}</strong><div class="snippet">Weight {weight}</div></td>
              <td class="mono">{baseline_score:.1f}</td>
              <td class="mono">{multi_score:.1f}</td>
              <td><span class="pill slate">{winner}</span></td>
            </tr>
            """
        )
    finding_cards = "".join(
        f"""
        <article class="panel finding-card">
          <h3>{_safe(metric.run_name)} evaluator findings</h3>
          <ul>{"".join(f"<li>{_safe(finding)}</li>" for finding in metric.evaluation_findings) or "<li>No findings returned.</li>"}</ul>
        </article>
        """
        for metric in (baseline_metric, multi_metric)
    )

    route_nodes = "".join(
        f'<span class="route-node">{idx + 1}. {_safe(route)}</span>'
        for idx, route in enumerate(multi_agent_state.route_history)
    )
    colors = {
        "supervisor": "violet",
        "researcher": "blue",
        "analyst": "amber",
        "writer": "mint",
        "critic": "rose",
    }
    trace_rows = []
    for idx, event in enumerate(multi_agent_state.trace):
        payload = event.get("payload", {})
        name = str(event.get("name", "unknown"))
        duration = float(payload.get("duration_seconds", 0) or 0)
        input_tokens = int(payload.get("input_tokens", 0) or 0)
        output_tokens = int(payload.get("output_tokens", 0) or 0)
        cost = float(payload.get("cost_usd", 0) or 0)
        summary = (
            payload.get("reason")
            or payload.get("feedback")
            or payload.get("notes")
            or payload.get("analysis")
            or payload.get("answer")
            or payload.get("output_notes")
            or payload.get("decision")
            or "Trace event recorded"
        )
        raw_payload = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        trace_rows.append(
            f"""
            <tr class="trace-main">
              <td class="mono">{idx + 1:02d}</td>
              <td><span class="pill {colors.get(name, "slate")}">{_safe(name)}</span></td>
              <td>{_safe(payload.get("iteration", "—"))}</td>
              <td class="mono">{duration:.3f}s</td>
              <td class="mono">{input_tokens:,} / {output_tokens:,}</td>
              <td class="mono">${cost:.6f}</td>
              <td><span class="status">{_safe(payload.get("status", "recorded"))}</span></td>
              <td class="summary">{_safe(str(summary)[:180])}</td>
              <td><button onclick="toggleTrace('trace-{idx}')">Inspect</button></td>
            </tr>
            <tr id="trace-{idx}" class="trace-detail">
              <td colspan="9">
                <div class="detail-grid">
                  <section><h4>Readable output</h4><pre>{_safe(summary)}</pre></section>
                  <section><h4>Raw payload</h4><pre>{_safe(raw_payload)}</pre></section>
                </div>
              </td>
            </tr>
            """
        )

    source_rows = "".join(
        f"""
        <tr>
          <td>[{idx + 1}]</td>
          <td><strong>{_safe(source.title)}</strong><div class="snippet">{_safe(source.snippet)}</div></td>
          <td>{f'<a href="{_safe(source.url)}" target="_blank" rel="noreferrer">Open source ↗</a>' if source.url else "N/A"}</td>
        </tr>
        """
        for idx, source in enumerate(multi_agent_state.sources)
    )
    raw_trace = json.dumps(multi_agent_state.trace, indent=2, ensure_ascii=False, default=str)
    latency_delta = multi_metric.latency_seconds - baseline_metric.latency_seconds
    cost_delta = (multi_metric.estimated_cost_usd or 0) - (baseline_metric.estimated_cost_usd or 0)
    token_delta = _total_tokens(multi_agent_state) - _total_tokens(baseline_state)
    quality_delta = (multi_metric.quality_score or 0) - (baseline_metric.quality_score or 0)
    baseline_quality_per_1k = (
        (baseline_metric.quality_score or 0) * 1000 / max(_total_tokens(baseline_state), 1)
    )
    multi_quality_per_1k = (
        (multi_metric.quality_score or 0) * 1000 / max(_total_tokens(multi_agent_state), 1)
    )
    baseline_quality_per_minute = (
        (baseline_metric.quality_score or 0) * 60 / max(baseline_metric.latency_seconds, 0.001)
    )
    multi_quality_per_minute = (
        (multi_metric.quality_score or 0) * 60 / max(multi_metric.latency_seconds, 0.001)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Agent Benchmark Dashboard</title>
<style>
:root{{--bg:#f3f6fb;--panel:#fff;--soft:#f8fafc;--ink:#0f172a;--muted:#64748b;--line:#dbe3ee;--rose:#f43f5e;--mint:#10b981;--blue:#2563eb;--shadow:0 8px 28px rgba(15,23,42,.07)}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,Segoe UI,Arial,sans-serif}}
button,input{{font:inherit}} .shell{{width:100%;min-height:100vh;padding:16px clamp(14px,2vw,32px) 40px}}
.topbar{{display:flex;justify-content:space-between;gap:20px;margin-bottom:14px}} h1{{margin:0;font-size:clamp(25px,3vw,39px);letter-spacing:-.04em}} h2{{margin:0;font-size:19px}} h3,h4,p{{margin-top:0}}
.subtitle,.meta,.section-head p,.caption,.notes,.snippet{{color:var(--muted)}} .meta{{text-align:right;font-size:12px}}
.query{{display:flex;gap:12px;align-items:center;padding:12px 16px;background:#0f172a;color:#fff;border-radius:12px;box-shadow:var(--shadow)}} .query span,.eyebrow{{font-size:10px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}} .query span{{color:#94a3b8}}
.nav{{position:sticky;top:0;z-index:5;display:flex;gap:5px;overflow:auto;margin:12px 0 18px;padding:7px;background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:12px;backdrop-filter:blur(12px)}} .nav a{{padding:7px 10px;color:var(--muted);font-weight:700;text-decoration:none;white-space:nowrap;border-radius:8px}} .nav a:hover{{background:#eef2f7;color:var(--ink)}}
.section{{margin-top:20px;scroll-margin-top:70px}} .section-head{{display:flex;justify-content:space-between;align-items:end;gap:12px;margin-bottom:10px}} .section-head p{{margin:2px 0 0}}
.run-grid,.compare-grid,.finding-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} .run-card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}}
.run-card{{padding:18px;border-top:4px solid var(--rose)}} .run-card.mint{{border-top-color:var(--mint)}} .run-card header{{display:flex;justify-content:space-between;gap:16px}} .run-card h3{{margin:2px 0 0}}
.quality{{display:grid;place-items:center;width:58px;height:58px;border:6px solid #e2e8f0;border-top-color:currentColor;border-radius:50%;font-size:17px;font-weight:900}} .quality small{{display:block;margin-top:-9px;font-size:9px;color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin-top:16px}} .metrics div,.delta div{{padding:10px;background:var(--soft);border:1px solid #edf1f6;border-radius:10px;min-width:0}} .metrics span,.delta span{{display:block;color:var(--muted);font-size:9px;font-weight:900;letter-spacing:.05em;text-transform:uppercase}} .metrics strong{{display:block;margin-top:4px;font-size:17px;white-space:nowrap}}
.token-bar{{height:6px;margin-top:14px;overflow:hidden;background:#dbeafe;border-radius:99px}} .token-bar span{{display:block;height:100%;background:var(--blue)}} .caption{{margin:4px 0 0;font-size:11px}} .notes{{margin:9px 0 0;font-size:12px}}
.delta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}} .delta strong{{display:block;font-size:21px}} .delta em{{color:var(--muted);font-size:11px;font-style:normal}}
.route{{display:flex;align-items:center;gap:6px;overflow:auto;padding:14px}} .route-node{{white-space:nowrap;padding:7px 10px;background:#ede9fe;color:#5b21b6;border-radius:8px;font-weight:800;font-size:12px}} .route-node:not(:last-child)::after{{content:"→";margin-left:16px;color:#94a3b8}}
.toolbar{{display:flex;gap:7px;flex-wrap:wrap}} input{{width:230px;padding:8px 10px;border:1px solid var(--line);border-radius:8px}} button{{padding:7px 10px;background:#fff;border:1px solid var(--line);border-radius:8px;cursor:pointer;font-weight:700}} button:hover{{background:#f8fafc;border-color:#94a3b8}}
.table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse}} th{{background:#f8fafc;color:var(--muted);font-size:10px;letter-spacing:.06em;text-align:left;text-transform:uppercase}} th,td{{padding:10px 11px;border-bottom:1px solid var(--line);vertical-align:top}} .mono{{font-family:Consolas,monospace}}
.pill,.status{{display:inline-block;padding:3px 7px;border-radius:99px;font-size:10px;font-weight:900;text-transform:uppercase}} .violet{{background:#ede9fe;color:#6d28d9}} .blue{{background:#dbeafe;color:#1d4ed8}} .amber{{background:#fef3c7;color:#b45309}} .mint{{background:#d1fae5;color:#047857}} .rose{{background:#ffe4e6;color:#be123c}} .slate{{background:#e2e8f0;color:#334155}} .status{{background:#dcfce7;color:#166534}}
.summary{{min-width:260px;max-width:520px;color:#475569}} .trace-detail{{display:none;background:#f8fafc}} .trace-detail.open{{display:table-row}} .detail-grid{{display:grid;grid-template-columns:1.2fr .8fr;gap:12px}}
pre{{max-height:440px;margin:0;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;padding:13px;background:#0f172a;color:#dbeafe;border-radius:9px;font:12px/1.55 Consolas,monospace}}
.output header{{display:flex;justify-content:space-between;padding:12px 14px;border-bottom:1px solid var(--line)}} .doc{{height:min(58vh,680px);overflow:auto;padding:18px;font-size:13px;line-height:1.7}} .snippet{{max-width:900px;margin-top:4px;font-size:12px}} a{{color:#2563eb}} .raw{{padding:14px}} .hidden{{display:none!important}}
.finding-card{{padding:16px}} .finding-card h3{{margin-bottom:8px}} .finding-card ul{{margin:0;padding-left:18px;color:#475569}}
@media(max-width:1100px){{.metrics{{grid-template-columns:repeat(3,1fr)}}}} @media(max-width:760px){{.shell{{padding:10px}}.topbar{{display:block}}.meta{{text-align:left;margin-top:6px}}.run-grid,.compare-grid,.finding-grid,.detail-grid{{grid-template-columns:1fr}}.delta{{grid-template-columns:repeat(2,1fr)}}.metrics{{grid-template-columns:repeat(2,1fr)}}.doc{{height:420px}}.query{{align-items:flex-start;flex-direction:column;gap:2px}}}}
</style>
</head>
<body><main class="shell">
<div class="topbar"><div><h1>Multi-Agent Research Benchmark</h1><p class="subtitle">Performance, routing, token usage, evidence, and complete execution trace.</p></div><div class="meta">Generated {_safe(generated_at)}<br>{len(multi_agent_state.trace)} trace events · {len(multi_agent_state.agent_results)} agent calls</div></div>
<div class="query"><span>Research query</span><strong>{_safe(multi_agent_state.request.query)}</strong></div>
<nav class="nav"><a href="#overview">Overview</a><a href="#quality">Quality rubric</a><a href="#tradeoffs">Trade-offs</a><a href="#route">Route</a><a href="#trace">Trace Explorer</a><a href="#outputs">Outputs</a><a href="#sources">Sources</a><a href="#raw">Raw JSON</a></nav>

<section id="overview" class="section"><div class="section-head"><div><h2>Run Overview</h2><p>Measured results from this benchmark execution.</p></div></div><div class="run-grid">{"".join(cards)}</div></section>
<section id="quality" class="section"><div class="section-head"><div><h2>Independent Quality Rubric</h2><p>The same judge scores both final answers. Internal agent findings are diagnostic only.</p></div></div>
<div class="panel table-wrap"><table><thead><tr><th>Dimension</th><th>Baseline</th><th>Multi-agent</th><th>Higher score</th></tr></thead><tbody>{"".join(score_rows)}</tbody></table></div>
<div class="finding-grid" style="margin-top:12px">{finding_cards}</div></section>
<section id="tradeoffs" class="section"><div class="section-head"><div><h2>Observed Trade-offs</h2><p>Multi-agent minus baseline, with ratios relative to baseline.</p></div></div><div class="delta">
<div><span>Latency delta</span><strong>{latency_delta:+.2f}s</strong><em>{_ratio(multi_metric.latency_seconds, baseline_metric.latency_seconds)} baseline</em></div>
<div><span>Cost delta</span><strong>${cost_delta:+.5f}</strong><em>{_ratio(multi_metric.estimated_cost_usd or 0, baseline_metric.estimated_cost_usd or 0)} baseline</em></div>
<div><span>Token delta</span><strong>{token_delta:+,}</strong><em>{_ratio(float(_total_tokens(multi_agent_state)), float(_total_tokens(baseline_state)))} baseline</em></div>
<div><span>Quality delta</span><strong>{quality_delta:+.1f}</strong><em>independent judge points</em></div>
<div><span>Quality / 1k tokens</span><strong>{multi_quality_per_1k:.2f}</strong><em>baseline {baseline_quality_per_1k:.2f}</em></div>
<div><span>Quality / minute</span><strong>{multi_quality_per_minute:.2f}</strong><em>baseline {baseline_quality_per_minute:.2f}</em></div></div></section>
<section id="route" class="section"><div class="section-head"><div><h2>Workflow Route</h2><p>Supervisor decisions in execution order.</p></div></div><div class="panel route">{route_nodes or '<span class="route-node">No route recorded</span>'}</div></section>

<section id="trace" class="section"><div class="section-head"><div><h2>Trace Explorer</h2><p>One row combines output, timing, tokens, cost, and route metadata.</p></div><div class="toolbar"><input id="filter" type="search" placeholder="Filter trace…" oninput="filterTrace()"><button onclick="setAll(true)">Expand all</button><button onclick="setAll(false)">Collapse all</button></div></div>
<div class="panel table-wrap"><table><thead><tr><th>#</th><th>Agent</th><th>Iteration</th><th>Duration</th><th>Tokens in/out</th><th>Cost</th><th>Status</th><th>Summary</th><th></th></tr></thead><tbody id="traceBody">{"".join(trace_rows)}</tbody></table></div></section>

<section id="outputs" class="section"><div class="section-head"><div><h2>Generated Output Comparison</h2><p>Independent scroll areas optimize screen space.</p></div></div><div class="compare-grid">
<article class="panel output"><header><strong>Single-Agent Baseline</strong><span>{len(baseline_state.final_answer or ""):,} chars</span></header><div class="doc">{_format_text(baseline_state.final_answer or "No final answer generated.")}</div></article>
<article class="panel output"><header><strong>Multi-Agent Guarded Output</strong><span>{len(multi_agent_state.final_answer or ""):,} chars</span></header><div class="doc">{_format_text(multi_agent_state.final_answer or "No final answer generated.")}</div></article></div></section>

<section id="sources" class="section"><div class="section-head"><div><h2>Evidence Sources</h2><p>Documents available to the workflow.</p></div></div><div class="panel table-wrap"><table><thead><tr><th>Ref</th><th>Source and snippet</th><th>Link</th></tr></thead><tbody>{source_rows or '<tr><td colspan="3">No sources recorded.</td></tr>'}</tbody></table></div></section>
<section id="raw" class="section"><div class="section-head"><div><h2>Raw Trace Export</h2><p>Complete machine-readable payload.</p></div><button onclick="copyRaw()">Copy JSON</button></div><div class="panel raw"><pre id="rawTrace">{_safe(raw_trace)}</pre></div></section>
</main>
<script>
function toggleTrace(id){{document.getElementById(id).classList.toggle('open')}}
function setAll(open){{document.querySelectorAll('.trace-detail').forEach(row=>row.classList.toggle('open',open))}}
function filterTrace(){{const q=document.getElementById('filter').value.toLowerCase();document.querySelectorAll('.trace-main').forEach(row=>{{const detail=row.nextElementSibling;const visible=(row.innerText+' '+detail.innerText).toLowerCase().includes(q);row.classList.toggle('hidden',!visible);detail.classList.toggle('hidden',!visible)}})}}
async function copyRaw(){{await navigator.clipboard.writeText(document.getElementById('rawTrace').innerText)}}
</script></body></html>"""
