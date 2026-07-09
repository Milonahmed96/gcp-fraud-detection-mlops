"""Tests for the A/B dashboard.

The assertions that matter are the ones a reader's trust depends on:

* An inconclusive result must be stated as inconclusive in the headline.
* The two SHAP panels must share one scale, or a side-by-side comparison lies.
* The page must be self-contained -- no CDN, no script, no network at all.
* Every bar carries a direct label and a table view exists, because the aqua
  series sits below 3:1 contrast on the light surface.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from src.evaluation.dashboard import (
    SERIES_COLOURS,
    TOP_FEATURES,
    importance_scale_max,
    main,
    render_html,
)
from src.evaluation.report import RANKING_METRICS, load_report


@pytest.fixture(scope="module")
def report(artifacts_dir):
    return load_report(artifacts_dir)


@pytest.fixture(scope="module")
def html(report) -> str:
    return render_html(report)


class TestSelfContained:
    def test_no_external_resources(self, html):
        """A strict CSP, an air-gapped reviewer, and a file:// URL must all work."""
        for forbidden in ("http://", "https://", "cdn.", "<script"):
            assert forbidden not in html, f"page references {forbidden}"

    def test_no_javascript(self, html):
        assert "<script" not in html and "onclick" not in html

    def test_is_a_complete_document(self, html):
        assert html.startswith("<!doctype html>")
        assert "</html>" in html
        assert '<meta name="viewport"' in html

    def test_styles_are_inlined(self, html):
        assert "<style>" in html and "<link" not in html


class TestHonestHeadline:
    def test_an_inconclusive_result_leads_with_no_significant_difference(self, html, report):
        if not report.significant:
            assert "No significant difference" in html
            assert "coin flip" in html

    def test_the_interval_is_always_shown_next_to_the_point_estimate(self, html, report):
        low, high = report.confidence_interval
        assert f"{low:.2f}" in html and f"{high:.2f}" in html

    def test_a_conclusive_result_names_the_winner(self, report):
        conclusive = render_html(replace(report, significant=True))
        assert "wins on business cost" in conclusive
        assert "No significant difference" not in conclusive

    def test_the_zero_line_is_labelled(self, html):
        assert "no difference" in html

    def test_it_says_whether_the_interval_spans_zero(self, html, report):
        low, high = report.confidence_interval
        expected = "The interval spans zero." if low <= 0 <= high else "The interval excludes zero."
        assert expected in html

    def test_the_f1_versus_cost_disagreement_is_called_out(self, html, report):
        """The whole reason the business cost metric exists."""
        if report.metrics_disagree:
            assert "F1 and cost disagree" in html
            assert "equally bad" in html


class TestSharedScale:
    def test_importance_scale_max_spans_both_variants(self, report):
        every_value = [v for variant in report.variants for v in variant.importance.values()]
        assert importance_scale_max(report) == max(every_value)

    def test_the_panels_share_one_scale(self, html, report):
        """Normalising each panel to its own max would draw LightGBM's 0.73 as
        long as XGBoost's 1.43 and invite a false cross-panel comparison."""
        widths = [float(w) for w in re.findall(r'<rect x="190" y="\d+" width="([\d.]+)"', html)]
        assert widths, "no importance bars found"

        scale_max = importance_scale_max(report)
        biggest_bar = max(widths)
        # The bar for the global maximum fills the plot; every other bar is
        # strictly shorter in proportion to its value.
        assert biggest_bar == pytest.approx(240.0, abs=0.5)

        lightgbm = next(v for v in report.variants if v.variant == "lightgbm")
        top_lgbm = max(lightgbm.importance.values())
        expected = top_lgbm / scale_max * 240.0
        assert any(w == pytest.approx(expected, abs=0.5) for w in widths)

    def test_the_shared_scale_is_stated_in_the_copy(self, html):
        assert "share one scale" in html

    def test_only_the_top_features_are_shown(self, html, report):
        """Direct labels stop working when flooded."""
        panels = html.count("mean |SHAP|")
        assert panels == TOP_FEATURES * len(report.variants)


class TestAccessibility:
    def test_a_legend_is_present_for_two_series(self, html):
        assert 'class="legend"' in html

    def test_every_bar_carries_a_direct_value_label(self, html, report):
        """The relief rule: the aqua series is below 3:1 on the light surface,
        so colour alone never carries a value."""
        for key, _ in RANKING_METRICS:
            for variant in report.variants:
                assert f"{variant.metric(key):.3f}" in html

    def test_a_table_view_exists(self, html):
        assert "<details" in html and "Table view" in html

    def test_every_chart_has_an_aria_label(self, html):
        svgs = html.count("<svg")
        assert html.count('role="img"') == svgs
        assert html.count("aria-label=") >= svgs

    def test_marks_carry_hover_titles(self, html):
        assert "<title>" in html

    def test_dark_mode_is_selected_not_inverted(self, html):
        assert "prefers-color-scheme: dark" in html
        assert "#3987e5" in html  # the dark-mode blue step, not the light one

    def test_colour_is_assigned_by_entity_not_rank(self):
        assert SERIES_COLOURS["xgboost"] != SERIES_COLOURS["lightgbm"]
        assert set(SERIES_COLOURS) == {"xgboost", "lightgbm"}


class TestHonestAboutMissingData:
    def test_the_footer_admits_there_is_no_production_data(self, html):
        assert "does not yet write" in html
        assert "prediction_log" in html

    def test_latency_is_not_fabricated(self, html):
        assert "latency is not shown" in html

    def test_a_missing_importance_profile_degrades_gracefully(self, report):
        stripped = replace(
            report, variants=tuple(replace(v, importance={}) for v in report.variants)
        )
        rendered = render_html(stripped)
        assert "No SHAP importance profile found" in rendered


class TestEscaping:
    def test_variant_names_are_escaped(self, report):
        hostile = replace(
            report,
            variants=(replace(report.variants[0], variant="<script>alert(1)</script>"),),
            winner="<script>alert(1)</script>",
        )
        rendered = render_html(hostile)
        assert "<script>alert(1)</script>" not in rendered
        assert "&lt;script&gt;" in rendered


class TestCLI:
    def test_writes_the_dashboard(self, artifacts_dir, tmp_path, capsys):
        output = tmp_path / "nested" / "dashboard.html"
        assert main(["--artifacts-dir", str(artifacts_dir), "--output", str(output)]) == 0
        assert output.exists()
        assert "<!doctype html>" in output.read_text()
        assert "verdict:" in capsys.readouterr().out

    def test_a_missing_metrics_file_exits_nonzero(self, tmp_path, capsys):
        assert main(["--artifacts-dir", str(tmp_path), "--output", str(tmp_path / "d.html")]) == 1
        assert "error:" in capsys.readouterr().err
