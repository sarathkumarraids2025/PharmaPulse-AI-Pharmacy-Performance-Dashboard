"""Chart smoke tests. Skipped automatically when plotly is not installed.
Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import plotly.graph_objects as go
    import charts as ch
    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    HAS_PLOTLY = False

import analytics as an


@unittest.skipUnless(HAS_PLOTLY, "plotly is not installed")
class ChartSmokeTests(unittest.TestCase):
    def frames(self):
        df, _ = an.prepare_data(pd.read_csv(ROOT / "data" / "customer_churn_cleaned.csv"))
        t = an.default_thresholds(df)
        df["Segment"] = an.assign_segments(df, t)
        subsets = {
            "all": df,
            "empty": df.iloc[0:0],
            "one": df.iloc[:1],
            "single_category": df[df["ContractType"] == "Two Year"],
            "churned": df[df["Churn"] == "Yes"],
            "retained": df[df["Churn"] == "No"],
        }
        return subsets

    def test_every_chart_returns_a_figure(self):
        for name, d in self.frames().items():
            order = an.ticket_order(d)
            summary = an.segment_summary(d)
            figs = [
                ch.churn_donut(d), ch.churn_gauge(d), ch.revenue_by_churn_bar(d),
                ch.histogram(d, "Age", "t"), ch.histogram(d, "TotalCharges", "t", by_churn=False),
                ch.category_counts(d, "ContractType", "t"), ch.category_counts(d, "PaymentMethod", "t", horizontal=True),
                ch.category_counts(d, "TicketGroup", "t", order=order),
                ch.churn_rate_bar(d, "ContractType", "t"), ch.churn_rate_bar(d, "PaymentMethod", "t", horizontal=True),
                ch.churn_rate_bar(d, "AgeGroup", "t", order=an.AGE_LABELS),
                ch.churn_rate_line(d, "TenureGroup", "t", an.TENURE_LABELS),
                ch.churn_heatmap(d, "ContractType", "PaymentMethod", "t"),
                ch.tenure_vs_charges_scatter(d), ch.box_by_category(d, "ContractType", "MonthlyCharges", "t"),
                ch.average_by_churn(d, "TenureMonths", "t"), ch.profile_comparison(d),
                ch.revenue_stacked_by(d, "ContractType", "t"), ch.revenue_at_risk_by(d, "PaymentMethod", "t"),
                ch.average_metric_by(d, "SubscriptionType", "MonthlyCharges", "t"),
                ch.segment_donut(summary), ch.segment_metric_bar(summary, "AvgTenure", "t", "Months"),
            ]
            for fig in figs:
                self.assertIsInstance(fig, go.Figure, name)
                fig.to_dict()  # forces full validation of the figure


if __name__ == "__main__":
    unittest.main()
