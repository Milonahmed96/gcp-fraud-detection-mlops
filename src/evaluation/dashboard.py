"""Render the A/B test result as a self-contained HTML dashboard.

No JavaScript, no CDN, no chart library: inline SVG only, so the page renders
from a `file://` URL, inside a locked-down Cloud Run service, and in an email
attachment alike.

Design decisions worth stating, because a dashboard that misleads is worse than
no dashboard:

* **The headline is the verdict, not the winner.** When the bootstrap interval
  straddles zero the page says "No significant difference" in the hero slot. A
  reader who skims must not come away believing a coin flip was a result.
* **Two scales, two charts, never a dual axis.** Ranking metrics (0-1) and cost
  per 1,000 transactions (currency) get separate charts.
* **Every bar is directly labelled and a table view is provided.** The aqua
  series sits below 3:1 contrast on the light surface, so colour alone never
  carries a value (the validator's "relief rule").
* **Diverging colour for the cost interval.** The confidence interval is a
  polarity: blue means the incumbent is cheaper, red means the challenger is.
  A grey zero line marks "no difference".

Regenerate with:  uv run python -m src.evaluation.dashboard
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

from src.evaluation.report import RANKING_METRICS, ABReport, ReportError, VariantResult, load_report

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_OUTPUT = Path("artifacts/dashboard.html")

#: Categorical slots 1 and 2 from the validated reference palette. Assigned by
#: entity (variant), never by rank -- a re-ordered chart must not repaint them.
SERIES_COLOURS = {"xgboost": "var(--series-1)", "lightgbm": "var(--series-2)"}

#: Top-N SHAP features per variant. Direct labels stop working when flooded.
TOP_FEATURES = 8

BAR_THICKNESS = 20  # <= 24px per the mark spec
BAR_RADIUS = 4  # rounded data-end
SURFACE_GAP = 2  # between adjacent bars


def _fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def _money(value: float) -> str:
    return f"£{value:,.2f}"


def _esc(text: str) -> str:
    return html.escape(str(text))


def _colour(variant: str) -> str:
    return SERIES_COLOURS.get(variant, "var(--series-1)")


def _legend(report: ABReport) -> str:
    """A legend is always present for >= 2 series; identity is never colour-alone."""
    items = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{_colour(v.variant)}">'
        f"</span>{_esc(v.label)}</span>"
        for v in report.variants
    )
    return f'<div class="legend">{items}</div>'


def _grouped_metric_chart(report: ABReport) -> str:
    """Grouped bars: one group per ranking metric, one bar per variant.

    All five metrics share the 0-1 scale, which is the only reason they may share
    an axis.
    """
    row_height = 46
    label_width = 92
    plot_width = 420
    value_gap = 8
    height = row_height * len(RANKING_METRICS) + 16
    n = len(report.variants)
    group_height = n * BAR_THICKNESS + (n - 1) * SURFACE_GAP

    rows: list[str] = []
    for i, (key, label) in enumerate(RANKING_METRICS):
        top = i * row_height + 8
        centre = top + group_height / 2

        # Recessive gridlines at 0.25 intervals, behind the marks.
        rows.append(
            f'<text x="{label_width - 10}" y="{centre + 4}" class="axis-label" '
            f'text-anchor="end">{_esc(label)}</text>'
        )
        for j, variant in enumerate(report.variants):
            value = variant.metric(key)
            bar_top = top + j * (BAR_THICKNESS + SURFACE_GAP)
            width = max(value * plot_width, 1.0)
            rows.append(
                f'<rect x="{label_width}" y="{bar_top}" width="{width:.1f}" '
                f'height="{BAR_THICKNESS}" rx="{BAR_RADIUS}" fill="{_colour(variant.variant)}">'
                f"<title>{_esc(variant.label)} — {_esc(label)}: {_fmt(value)}</title></rect>"
                f'<text x="{label_width + width + value_gap:.1f}" y="{bar_top + 14}" '
                f'class="value-label">{_fmt(value)}</text>'
            )

    gridlines = "".join(
        f'<line x1="{label_width + f * plot_width}" y1="0" '
        f'x2="{label_width + f * plot_width}" y2="{height}" class="gridline" />'
        for f in (0.25, 0.5, 0.75, 1.0)
    )

    return f"""
    <figure class="chart">
      <figcaption>
        <h3>Ranking quality</h3>
        <p>Higher is better. All five metrics share the 0–1 scale.</p>
      </figcaption>
      {_legend(report)}
      <svg viewBox="0 0 620 {height}" role="img"
           aria-label="Grouped bar chart comparing ROC-AUC, PR-AUC, F1, precision and recall for both variants">
        {gridlines}
        <line x1="{label_width}" y1="0" x2="{label_width}" y2="{height}" class="baseline" />
        {"".join(rows)}
      </svg>
    </figure>"""


def _cost_chart(report: ABReport) -> str:
    """Cost per 1,000 transactions. Its own chart: a different scale entirely."""
    plot_width = 380
    label_width = 92
    row_height = 44
    height = row_height * len(report.variants) + 12
    worst = max(v.cost_per_1000 for v in report.variants)

    rows: list[str] = []
    for i, variant in enumerate(report.variants):
        top = i * row_height + 8
        width = max(variant.cost_per_1000 / worst * plot_width, 1.0)
        cheapest = variant.variant == report.cheaper().variant
        marker = " ◀ cheaper" if cheapest else ""
        rows.append(
            f'<text x="{label_width - 10}" y="{top + 15}" class="axis-label" '
            f'text-anchor="end">{_esc(variant.label)}</text>'
            f'<rect x="{label_width}" y="{top}" width="{width:.1f}" height="{BAR_THICKNESS}" '
            f'rx="{BAR_RADIUS}" fill="{_colour(variant.variant)}">'
            f"<title>{_esc(variant.label)}: {_money(variant.cost_per_1000)} per 1,000</title>"
            f"</rect>"
            f'<text x="{label_width + width + 8:.1f}" y="{top + 14}" class="value-label">'
            f"{_money(variant.cost_per_1000)}{marker}</text>"
        )

    return f"""
    <figure class="chart">
      <figcaption>
        <h3>Expected business cost</h3>
        <p>Per 1,000 transactions. <strong>Lower is better.</strong> A missed fraud costs its
           transaction amount; a wrongly blocked customer costs a flat fee. These are not
           symmetric, which is why accuracy-flavoured metrics pick the wrong model.</p>
      </figcaption>
      <svg viewBox="0 0 620 {height}" role="img"
           aria-label="Bar chart of expected business cost per 1000 transactions for both variants">
        <line x1="{label_width}" y1="0" x2="{label_width}" y2="{height}" class="baseline" />
        {"".join(rows)}
      </svg>
    </figure>"""


def _interval_chart(report: ABReport) -> str:
    """The bootstrap interval for cost(A) − cost(B), on a diverging scale.

    Blue = the incumbent is cheaper, red = the challenger is, grey zero line =
    no difference. When the interval crosses zero, the reader can see it cross.
    """
    low, high = report.confidence_interval
    point = report.cost_difference_per_1000

    span = max(abs(low), abs(high), abs(point)) * 1.25 or 1.0
    width, height = 560, 92
    mid_x, plot_half = width / 2, width / 2 - 40

    def to_x(value: float) -> float:
        return mid_x + (value / span) * plot_half

    crosses_zero = low <= 0.0 <= high
    bar_colour = (
        "var(--muted-fill)"
        if crosses_zero
        else ("var(--series-1)" if point < 0 else "var(--critical)")
    )

    x_low, x_high, x_point = to_x(low), to_x(high), to_x(point)
    bar_y = 34

    return f"""
    <figure class="chart">
      <figcaption>
        <h3>Cost difference, with 95% bootstrap interval</h3>
        <p>XGBoost minus LightGBM, per 1,000 transactions. Negative means XGBoost costs less.
           Resampled on the same transactions for both variants, so the comparison is paired.</p>
      </figcaption>
      <svg viewBox="0 0 {width} {height}" role="img"
           aria-label="Confidence interval from {_fmt(low, 2)} to {_fmt(high, 2)}, point estimate {_fmt(point, 2)}, {"crossing" if crosses_zero else "not crossing"} zero">
        <line x1="{mid_x}" y1="12" x2="{mid_x}" y2="{height - 24}" class="zero-line" />
        <text x="{mid_x}" y="{height - 8}" class="axis-label" text-anchor="middle">0 — no difference</text>

        <rect x="{min(x_low, x_high):.1f}" y="{bar_y}" width="{abs(x_high - x_low):.1f}"
              height="{BAR_THICKNESS}" rx="{BAR_RADIUS}" fill="{bar_colour}" opacity="0.55">
          <title>95% interval: {_fmt(low, 2)} to {_fmt(high, 2)}</title>
        </rect>
        <circle cx="{x_point:.1f}" cy="{bar_y + BAR_THICKNESS / 2}" r="5.5"
                fill="{bar_colour}" stroke="var(--surface-1)" stroke-width="2">
          <title>Point estimate: {_fmt(point, 2)}</title>
        </circle>

        <text x="{x_low:.1f}" y="{bar_y - 8}" class="value-label" text-anchor="middle">{_fmt(low, 1)}</text>
        <text x="{x_high:.1f}" y="{bar_y - 8}" class="value-label" text-anchor="middle">+{_fmt(high, 1)}</text>
      </svg>
      <p class="note">{"The interval spans zero." if crosses_zero else "The interval excludes zero."}</p>
    </figure>"""


def _confusion_table(report: ABReport) -> str:
    rows = "".join(
        f"<tr><th scope=\"row\"><span class='swatch' style='background:{_colour(v.variant)}'></span>"
        f"{_esc(v.label)}</th>"
        f"<td>{v.true_positives}</td><td>{v.false_negatives}</td><td>{v.false_positives}</td>"
        f"<td>{_fmt(v.threshold, 4)}</td><td>{_money(v.total_cost)}</td></tr>"
        for v in report.variants
    )
    return f"""
    <figure class="chart">
      <figcaption>
        <h3>Decisions on the test split</h3>
        <p>At each variant's own cost-minimising threshold, fitted on validation — never on test.</p>
      </figcaption>
      <table>
        <thead><tr>
          <th scope="col">Variant</th>
          <th scope="col">Fraud caught</th>
          <th scope="col">Fraud missed</th>
          <th scope="col">Customers blocked</th>
          <th scope="col">Threshold</th>
          <th scope="col">Total cost</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </figure>"""


def _importance_chart(variant: VariantResult, scale_max: float) -> str:
    """Horizontal bars, descending. One panel per variant (small multiples).

    `scale_max` is the maximum across *both* panels, not this one. Small
    multiples must share a scale: normalising each panel to its own maximum
    would draw LightGBM's 0.73 as long as XGBoost's 1.43 and invite exactly the
    cross-panel comparison the layout encourages.
    """
    if not variant.importance:
        return (
            f'<figure class="chart"><figcaption><h3>{_esc(variant.label)}</h3></figcaption>'
            '<p class="note">No SHAP importance profile found. Re-run training.</p></figure>'
        )

    ranked = sorted(variant.importance.items(), key=lambda kv: kv[1], reverse=True)[:TOP_FEATURES]
    biggest = scale_max or 1.0
    row_height = 30
    label_width = 190
    plot_width = 240
    height = row_height * len(ranked) + 10

    rows = []
    for i, (feature, value) in enumerate(ranked):
        top = i * row_height + 6
        bar_width = max(value / biggest * plot_width, 1.0)
        rows.append(
            f'<text x="{label_width - 10}" y="{top + 13}" class="axis-label" '
            f'text-anchor="end">{_esc(feature)}</text>'
            f'<rect x="{label_width}" y="{top}" width="{bar_width:.1f}" height="18" '
            f'rx="{BAR_RADIUS}" fill="{_colour(variant.variant)}">'
            f"<title>{_esc(feature)}: mean |SHAP| {_fmt(value)}</title></rect>"
            f'<text x="{label_width + bar_width + 8:.1f}" y="{top + 13}" '
            f'class="value-label">{_fmt(value, 2)}</text>'
        )

    return f"""
    <figure class="chart">
      <figcaption><h3>{_esc(variant.label)}</h3></figcaption>
      <svg viewBox="0 0 500 {height}" role="img"
           aria-label="Top {len(ranked)} features by mean absolute SHAP value for {_esc(variant.label)}">
        <line x1="{label_width}" y1="0" x2="{label_width}" y2="{height}" class="baseline" />
        {"".join(rows)}
      </svg>
    </figure>"""


def _data_table(report: ABReport) -> str:
    """The table view. Required: the aqua series is below 3:1 on the light surface."""
    header = "".join(f"<th scope='col'>{_esc(v.label)}</th>" for v in report.variants)
    rows = "".join(
        f"<tr><th scope='row'>{_esc(label)}</th>"
        + "".join(f"<td>{_fmt(v.metric(key))}</td>" for v in report.variants)
        + "</tr>"
        for key, label in RANKING_METRICS
    )
    cost_row = (
        "<tr><th scope='row'>Cost per 1,000</th>"
        + "".join(f"<td>{_money(v.cost_per_1000)}</td>" for v in report.variants)
        + "</tr>"
    )
    return f"""
    <details class="table-view">
      <summary>Table view — every value, no colour required</summary>
      <table>
        <thead><tr><th scope="col">Metric</th>{header}</tr></thead>
        <tbody>{rows}{cost_row}</tbody>
      </table>
    </details>"""


def _hero(report: ABReport) -> str:
    status = "inconclusive" if not report.significant else "conclusive"
    low, high = report.confidence_interval
    disagreement = ""
    if report.metrics_disagree:
        f1_winner = report.best_on("f1")
        cost_winner = report.cheaper()
        disagreement = (
            f'<p class="callout"><strong>F1 and cost disagree.</strong> '
            f"{_esc(f1_winner.label)} has the better F1 ({_fmt(f1_winner.f1)}) but "
            f"{_esc(cost_winner.label)} is cheaper ({_money(cost_winner.cost_per_1000)} per 1,000). "
            "F1 treats a missed fraud and a blocked customer as equally bad. They are not.</p>"
        )

    return f"""
    <section class="hero">
      <p class="eyebrow">A/B test — {_esc(status)}</p>
      <h2>{_esc(report.verdict)}</h2>
      <p class="hero-figure">{_fmt(report.cost_difference_per_1000, 2)}
        <span class="hero-unit">cost delta per 1,000</span></p>
      <p class="hero-interval">95% interval [{_fmt(low, 2)}, +{_fmt(high, 2)}]</p>
      <p class="hero-detail">{_esc(report.verdict_detail)}</p>
      {disagreement}
    </section>"""


STYLES = """
:root {
  --surface-1: #fcfcfb;
  --page: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --gridline: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --series-2: #1baf7a;
  --critical: #d03b3b;
  --muted-fill: #898781;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --gridline: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
    --series-2: #199e70;
    --critical: #d03b3b;
    --muted-fill: #898781;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 2rem 1rem 4rem;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.5;
}
main { max-width: 900px; margin: 0 auto; }
header h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
header p { color: var(--text-secondary); margin: 0 0 1.5rem; }
.hero {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
}
.eyebrow {
  text-transform: uppercase; letter-spacing: .08em; font-size: .72rem;
  color: var(--muted); margin: 0 0 .5rem;
}
.hero h2 { margin: 0 0 1rem; font-size: 1.5rem; }
.hero-figure { font-size: 2.75rem; font-weight: 600; margin: 0; line-height: 1.1; }
.hero-unit { font-size: .85rem; font-weight: 400; color: var(--text-secondary); display: block; }
.hero-interval { font-variant-numeric: tabular-nums; color: var(--text-secondary); margin: .5rem 0 1rem; }
.hero-detail { color: var(--text-secondary); margin: 0; max-width: 62ch; }
.callout {
  margin-top: 1rem; padding: .75rem 1rem; border-left: 3px solid var(--series-1);
  background: var(--page); color: var(--text-secondary); border-radius: 0 6px 6px 0;
}
.chart {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.25rem; margin: 0 0 1.5rem;
}
figcaption h3 { margin: 0 0 .25rem; font-size: 1.05rem; }
figcaption p { margin: 0 0 1rem; color: var(--text-secondary); font-size: .875rem; max-width: 68ch; }
svg { width: 100%; height: auto; overflow: visible; }
.gridline { stroke: var(--gridline); stroke-width: 1; }
.baseline { stroke: var(--baseline); stroke-width: 1; }
.zero-line { stroke: var(--baseline); stroke-width: 2; stroke-dasharray: 4 3; }
.axis-label { fill: var(--muted); font-size: 12px; font-family: inherit; }
.value-label { fill: var(--text-secondary); font-size: 11px; font-family: inherit; font-variant-numeric: tabular-nums; }
.legend { display: flex; gap: 1.25rem; margin-bottom: .75rem; font-size: .8rem; color: var(--text-secondary); }
.legend-item { display: inline-flex; align-items: center; gap: .4rem; }
.swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
.note { color: var(--muted); font-size: .8rem; margin: .5rem 0 0; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; }
.grid-2 .chart { margin: 0; }
table { border-collapse: collapse; width: 100%; font-size: .875rem; }
th, td { text-align: right; padding: .5rem .6rem; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
th[scope="row"] { text-align: left; font-weight: 500; white-space: nowrap; }
thead th { color: var(--muted); font-weight: 500; font-size: .78rem; text-align: right; }
thead th:first-child { text-align: left; }
th[scope="row"] .swatch { margin-right: .5rem; vertical-align: middle; }
.table-view { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.25rem; }
.table-view summary { cursor: pointer; color: var(--text-secondary); font-size: .875rem; }
.table-view table { margin-top: 1rem; }
footer { margin-top: 2rem; color: var(--muted); font-size: .8rem; }
footer code { background: var(--surface-1); padding: .1rem .35rem; border-radius: 4px; border: 1px solid var(--border); }
.scroll-x { overflow-x: auto; }
"""


def importance_scale_max(report: ABReport) -> float:
    """The largest mean-|SHAP| across every variant, so the panels share a scale."""
    values = [value for v in report.variants for value in v.importance.values()]
    return max(values, default=1.0)


def render_html(report: ABReport) -> str:
    """Render the whole dashboard as one self-contained HTML document."""
    scale_max = importance_scale_max(report)
    importance_panels = "".join(_importance_chart(v, scale_max) for v in report.variants)
    train_rows = report.variants[0].n_train
    test_rows = report.variants[0].n_test

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud Detection — A/B Test Results</title>
<style>{STYLES}</style>
</head>
<body>
<main>
  <header>
    <h1>Fraud detection: XGBoost vs LightGBM</h1>
    <p>Offline evaluation on a held-out temporal split —
       {train_rows:,} training rows, {test_rows:,} test rows.</p>
  </header>

  {_hero(report)}
  {_interval_chart(report)}
  {_cost_chart(report)}
  {_grouped_metric_chart(report)}
  <div class="scroll-x">{_confusion_table(report)}</div>

  <h2>What drives each model</h2>
  <p class="note">Mean absolute SHAP value over the training split. Higher means the feature
     moves the prediction more, in either direction. <strong>Both panels share one scale</strong>,
     so bar lengths are comparable across variants.</p>
  <div class="grid-2">{importance_panels}</div>

  {_data_table(report)}

  <footer>
    <p><strong>These are offline numbers.</strong> The inference service does not yet write to the
       BigQuery <code>prediction_log</code>, so no production traffic is aggregated here and
       serving latency is not shown. See <em>Known limitations</em> in the README.</p>
    <p>Regenerate with <code>uv run python -m src.evaluation.dashboard</code>.</p>
  </footer>
</main>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the A/B test dashboard.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        report = load_report(args.artifacts_dir)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(report))
    print(f"wrote {args.output} — verdict: {report.verdict}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
