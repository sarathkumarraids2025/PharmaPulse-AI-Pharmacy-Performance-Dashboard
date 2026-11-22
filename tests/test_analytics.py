"""Unit tests for the analytics layer. Run: python -m unittest discover -s tests -v

Expected values are verified by hand from data/customer_churn_cleaned.csv.
Only pandas and numpy are needed for this file.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analytics as an  # noqa: E402

CSV = ROOT / "data" / "customer_churn_cleaned.csv"


def load():
    df, report = an.prepare_data(pd.read_csv(CSV))
    t = an.default_thresholds(df)
    df["Segment"] = an.assign_segments(df, t)
    df["RiskFlags"] = an.risk_flags(df, t)
    return df, report, t


class KpiTests(unittest.TestCase):
    def setUp(self):
        self.df, self.report, self.t = load()

    def test_validation_passes_on_real_file(self):
        self.assertTrue(self.report.ok)
        self.assertEqual(self.report.rows_used, 15)
        self.assertEqual(self.report.duplicate_rows, 0)
        self.assertEqual(self.report.duplicate_ids, 0)

    def test_headline_kpis(self):
        k = an.compute_kpis(self.df)
        self.assertEqual(k["total_customers"], 15)
        self.assertEqual(k["churned_customers"], 7)
        self.assertEqual(k["retained_customers"], 8)
        self.assertAlmostEqual(k["churn_rate"], 7 / 15 * 100)
        self.assertAlmostEqual(k["retention_rate"], 8 / 15 * 100)
        self.assertAlmostEqual(k["churn_rate"] + k["retention_rate"], 100)

    def test_revenue_at_risk_is_monthly_charges_of_churned(self):
        k = an.compute_kpis(self.df)
        # five Basic at 49.99 + two Pro at 79.99 churned
        self.assertAlmostEqual(k["revenue_at_risk"], 5 * 49.99 + 2 * 79.99, places=2)
        self.assertAlmostEqual(k["total_monthly_revenue"], self.df["MonthlyCharges"].sum())
        self.assertAlmostEqual(k["retained_monthly_revenue"] + k["revenue_at_risk"], k["total_monthly_revenue"])

    def test_averages(self):
        k = an.compute_kpis(self.df)
        self.assertAlmostEqual(k["avg_tenure"], self.df["TenureMonths"].mean())
        self.assertAlmostEqual(k["avg_support_tickets"], 38 / 15)

    def test_empty_dataset_kpis_are_zero(self):
        k = an.compute_kpis(self.df.iloc[0:0])
        self.assertEqual(k["total_customers"], 0)
        self.assertEqual(k["churn_rate"], 0.0)
        self.assertEqual(k["revenue_at_risk"], 0.0)


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.df, _, self.t = load()

    def test_churn_by_contract(self):
        table = an.churn_by(self.df, "ContractType").set_index("ContractType")
        self.assertEqual(table.loc["Month-to-Month", "Customers"], 7)
        self.assertEqual(table.loc["Month-to-Month", "ChurnRate"], 100.0)
        self.assertEqual(table.loc["Two Year", "ChurnRate"], 0.0)
        self.assertEqual(table["Customers"].sum(), 15)

    def test_tenure_groups_cover_every_customer(self):
        table = an.churn_by(self.df, "TenureGroup", order=an.TENURE_LABELS)
        self.assertEqual(int(table["Customers"].sum()), 15)
        self.assertEqual(table.iloc[0]["TenureGroup"], "0-6 months")

    def test_keep_empty_groups(self):
        table = an.churn_by(self.df[self.df["TenureMonths"] > 30], "TenureGroup", order=an.TENURE_LABELS, keep_empty=True)
        self.assertEqual(len(table), 5)
        self.assertEqual(int(table["Customers"].sum()), 3)

    def test_empty_input_returns_empty_table(self):
        self.assertTrue(an.churn_by(self.df.iloc[0:0], "ContractType").empty)

    def test_heatmap_data_shapes(self):
        rates, counts = an.churn_heatmap_data(self.df, "ContractType", "PaymentMethod")
        self.assertEqual(rates.shape, counts.shape)
        self.assertEqual(int(counts.to_numpy().sum()), 15)


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.df, _, _ = load()

    def test_categorical_filter(self):
        out = an.apply_filters(self.df, {"ContractType": ["Two Year"]})
        self.assertEqual(len(out), 4)
        self.assertTrue((out["ContractType"] == "Two Year").all())

    def test_range_filter_inclusive(self):
        out = an.apply_filters(self.df, ranges={"TenureMonths": (12, 24)})
        self.assertTrue(out["TenureMonths"].between(12, 24).all())

    def test_float_range_filter(self):
        out = an.apply_filters(self.df, ranges={"MonthlyCharges": (79.99, 79.99)})
        self.assertEqual(len(out), 4)

    def test_empty_selection_returns_no_rows_without_crash(self):
        out = an.apply_filters(self.df, {"ContractType": []})
        self.assertEqual(len(out), 0)
        self.assertEqual(an.compute_kpis(out)["total_customers"], 0)
        self.assertEqual(an.generate_insights(out), [])
        self.assertEqual(an.generate_recommendations(out), [])

    def test_none_means_no_filter(self):
        self.assertEqual(len(an.apply_filters(self.df, {"ContractType": None}, {"Age": None})), 15)

    def test_combined_filters(self):
        out = an.apply_filters(self.df, {"Churn": ["Yes"], "PaymentMethod": ["UPI"]})
        self.assertEqual(len(out), 3)


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.df, _, self.t = load()

    def test_every_customer_has_exactly_one_known_segment(self):
        self.assertTrue(self.df["Segment"].isin(an.SEGMENT_ORDER).all())
        self.assertEqual(self.df["Segment"].notna().sum(), 15)

    def test_churned_customers_segment_matches_churn(self):
        self.assertEqual(
            set(self.df.loc[self.df["Segment"] == "Churned Customers", "Churn"]), {"Yes"}
        )
        self.assertEqual(int((self.df["Segment"] == "Churned Customers").sum()), 7)

    def test_summary_counts_add_up(self):
        summary = an.segment_summary(self.df)
        self.assertEqual(int(summary["Customers"].sum()), 15)
        self.assertAlmostEqual(summary["SharePct"].sum(), 100)
        self.assertEqual(len(summary), len(an.SEGMENT_ORDER))

    def test_high_revenue_threshold_is_data_driven(self):
        self.assertAlmostEqual(self.t.high_revenue_charge, self.df["MonthlyCharges"].quantile(0.75))

    def test_summary_of_empty_frame(self):
        summary = an.segment_summary(self.df.iloc[0:0])
        self.assertEqual(int(summary["Customers"].sum()), 0)

    def test_risk_flag_prevalence(self):
        prev = an.flag_prevalence(self.df).set_index("Trait")
        self.assertEqual(prev.loc["Month-to-Month", "ChurnedWith"], 7)
        self.assertEqual(prev.loc["Month-to-Month", "RetainedWith"], 0)


class InsightTests(unittest.TestCase):
    def setUp(self):
        self.df, _, _ = load()

    def test_insights_are_generated_from_data(self):
        text = " ".join(i["headline"] + i["detail"] for i in an.generate_insights(self.df))
        self.assertIn("Month-to-Month", text)
        self.assertIn("Debit Card", text)

    def test_single_category_does_not_crash(self):
        one = an.apply_filters(self.df, {"ContractType": ["Two Year"]})
        insights = an.generate_insights(one)
        self.assertTrue(insights)
        recs = an.generate_recommendations(one)
        self.assertTrue(recs)

    def test_no_churn_selection(self):
        none = an.apply_filters(self.df, {"Churn": ["No"]})
        self.assertTrue(an.generate_insights(none))
        self.assertEqual(an.compute_kpis(none)["revenue_at_risk"], 0.0)

    def test_revenue_notes_mention_lifetime_vs_monthly(self):
        self.assertIn("lifetime", " ".join(an.revenue_notes(self.df)))


class ValidationTests(unittest.TestCase):
    def messy(self):
        return pd.DataFrame(
            {
                "CustomerID": ["A1", "A1", "B2", None, "D4"],
                "Gender": ["F", "F", None, "M", "M"],
                "Age": [30, 30, "abc", 40, -5],
                "TenureMonths": [5, 5, 10, None, 12],
                "SubscriptionType": ["Basic"] * 5,
                "MonthlyCharges": [10.0, 10.0, 20.0, 30.0, 40.0],
                "TotalCharges": [50, 50, 200, 300, 480],
                "ContractType": ["Month-to-Month"] * 5,
                "SupportTickets": [1, 1, 2, 3, 4],
                "PaymentMethod": ["UPI"] * 5,
                "Churn": ["Yes", "Yes", "no", "maybe", "YES"],
            }
        )

    def test_messy_data_is_handled_without_crash(self):
        df, report = an.prepare_data(self.messy())
        self.assertTrue(report.ok)
        self.assertEqual(report.rows_used, 4)  # "maybe" removed
        self.assertEqual(report.duplicate_rows, 1)
        self.assertEqual(report.duplicate_ids, 1)
        self.assertTrue(report.warnings)
        self.assertTrue(df["Age"].isna().any())
        self.assertTrue((df["Age"].dropna() >= 0).all())
        self.assertEqual(set(df["Churn"]), {"Yes", "No"})
        k = an.compute_kpis(df)
        self.assertEqual(k["total_customers"], 4)
        an.generate_insights(df)

    def test_missing_columns_reported(self):
        df, report = an.prepare_data(pd.DataFrame({"CustomerID": ["x"]}))
        self.assertFalse(report.ok)
        self.assertTrue(df.empty)

    def test_empty_frame_reported(self):
        _, report = an.prepare_data(pd.DataFrame())
        self.assertFalse(report.ok)

    def test_all_invalid_churn(self):
        raw = self.messy()
        raw["Churn"] = "?"
        _, report = an.prepare_data(raw)
        self.assertFalse(report.ok)


class FormatTests(unittest.TestCase):
    def test_formatters(self):
        self.assertEqual(an.fmt_money(1234.5), "$1,234.50")
        self.assertEqual(an.fmt_pct(46.6667), "46.7%")
        self.assertEqual(an.fmt_money(float("nan")), "-")


if __name__ == "__main__":
    unittest.main()
