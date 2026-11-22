"""Analytics layer for the Customer Churn KPI Dashboard.

This module only depends on pandas and numpy. It never imports Streamlit or
Plotly, so every calculation can be unit-tested without a UI.

Everything is computed from the uploaded dataset - there are no hardcoded
KPI values, no synthetic customers and no machine-learning predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS: List[str] = [
    "CustomerID",
    "Gender",
    "Age",
    "TenureMonths",
    "SubscriptionType",
    "MonthlyCharges",
    "TotalCharges",
    "ContractType",
    "SupportTickets",
    "PaymentMethod",
    "Churn",
]
NUMERIC_COLUMNS: List[str] = [
    "Age",
    "TenureMonths",
    "MonthlyCharges",
    "TotalCharges",
    "SupportTickets",
]
CATEGORICAL_COLUMNS: List[str] = [
    "Gender",
    "SubscriptionType",
    "ContractType",
    "PaymentMethod",
]

TENURE_BINS = [-np.inf, 6, 12, 24, 36, np.inf]
TENURE_LABELS = ["0-6 months", "7-12 months", "13-24 months", "25-36 months", "37+ months"]
AGE_BINS = [-np.inf, 29, 39, 49, np.inf]
AGE_LABELS = ["Under 30", "30-39", "40-49", "50+"]

SMALL_SAMPLE_LIMIT = 30  # below this many customers, results are only indicative

_YES = {"yes", "y", "true", "1", "churned"}
_NO = {"no", "n", "false", "0", "retained"}


# --------------------------------------------------------------------------- #
# Formatting helpers (shared by the UI and the insight text)
# --------------------------------------------------------------------------- #
def fmt_money(value: Any, decimals: int = 2) -> str:
    """Format a number as currency, e.g. 1234.5 -> '$1,234.50'."""
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"${float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_pct(value: Any, decimals: int = 1) -> str:
    """Format a percentage value that is already on a 0-100 scale."""
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.{decimals}f}%"
    except (TypeError, ValueError):
        return "-"


def fmt_num(value: Any, decimals: int = 1) -> str:
    """Format a plain number with thousands separators."""
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _nz(value: Any) -> float:
    """Return 0.0 for NaN/None so KPI cards never show 'nan'."""
    try:
        return 0.0 if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# Validation and preparation
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    """Outcome of validating the raw CSV."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rows_loaded: int = 0
    rows_used: int = 0
    duplicate_rows: int = 0
    duplicate_ids: int = 0
    missing_total: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _normalise_churn(value: Any) -> Optional[str]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text in _YES:
        return "Yes"
    if text in _NO:
        return "No"
    return None


def prepare_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean the raw dataset.

    Returns the analysis-ready DataFrame (with derived columns) and a
    ValidationReport. The function never raises for data problems; fatal
    issues are reported in ``report.errors`` and an empty frame is returned.
    """
    report = ValidationReport()

    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        report.errors.append("The dataset is empty - no customer records were found.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS), report

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    report.rows_loaded = int(len(df))

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.errors.append(
            "Required column(s) missing from the CSV: " + ", ".join(missing_cols) + "."
        )
        return pd.DataFrame(columns=REQUIRED_COLUMNS), report

    df = df[REQUIRED_COLUMNS].copy()
    report.missing_total = int(df.isna().sum().sum())
    report.duplicate_rows = int(df.duplicated().sum())

    # Identifier
    df["CustomerID"] = df["CustomerID"].astype("object").map(
        lambda v: "Unknown" if pd.isna(v) or str(v).strip() == "" else str(v).strip()
    )
    report.duplicate_ids = int(df["CustomerID"].duplicated().sum())
    if report.duplicate_rows:
        report.warnings.append(
            f"{report.duplicate_rows} fully duplicated record(s) found. They are kept so you can inspect them."
        )
    if report.duplicate_ids:
        report.warnings.append(
            f"{report.duplicate_ids} duplicate CustomerID value(s) found - customer counts may be inflated."
        )

    # Numeric columns
    for col in NUMERIC_COLUMNS:
        before_missing = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid = int(df[col].isna().sum()) - before_missing
        negatives = int((df[col] < 0).sum())
        if negatives:
            df.loc[df[col] < 0, col] = np.nan
        if invalid > 0:
            report.warnings.append(
                f"{invalid} non-numeric value(s) in '{col}' were treated as missing."
            )
        if negatives:
            report.warnings.append(
                f"{negatives} negative value(s) in '{col}' are invalid and were treated as missing."
            )
        remaining = int(df[col].isna().sum())
        if remaining:
            report.warnings.append(
                f"'{col}' has {remaining} missing value(s); those records are excluded from averages of that column."
            )

    # Categorical columns
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("object").map(
            lambda v: "Unknown" if pd.isna(v) or str(v).strip() == "" else str(v).strip()
        )

    # Churn status
    df["Churn"] = df["Churn"].map(_normalise_churn)
    bad_churn = int(df["Churn"].isna().sum())
    if bad_churn:
        report.warnings.append(
            f"{bad_churn} record(s) have an unexpected or missing Churn value (expected Yes/No) and were removed."
        )
        df = df[df["Churn"].notna()].copy()

    df = df.reset_index(drop=True)
    report.rows_used = int(len(df))
    if df.empty:
        report.errors.append("No valid records remain after validation.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS), report

    # Derived columns
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    df["TenureGroup"] = pd.cut(
        df["TenureMonths"], bins=TENURE_BINS, labels=TENURE_LABELS
    ).astype("object")
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS).astype("object")
    df["TenureGroup"] = df["TenureGroup"].fillna("Unknown")
    df["AgeGroup"] = df["AgeGroup"].fillna("Unknown")
    df["TicketGroup"] = df["SupportTickets"].map(
        lambda v: "Unknown" if pd.isna(v) else str(int(round(v)))
    )

    if report.rows_used < SMALL_SAMPLE_LIMIT:
        report.warnings.append(
            f"Small dataset: only {report.rows_used} customers. Findings are illustrative, not statistically reliable."
        )
    return df, report


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def apply_filters(
    df: pd.DataFrame,
    categorical: Optional[Dict[str, Optional[Iterable[Any]]]] = None,
    ranges: Optional[Dict[str, Optional[Tuple[float, float]]]] = None,
) -> pd.DataFrame:
    """Return the rows that satisfy every filter.

    ``categorical`` maps column -> allowed values (``None`` means no filter,
    an empty list means nothing is allowed). ``ranges`` maps column ->
    (low, high), inclusive. Rows with a missing numeric value are kept so
    that incomplete records do not silently disappear.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    mask = pd.Series(True, index=df.index)
    for col, allowed in (categorical or {}).items():
        if allowed is None or col not in df.columns:
            continue
        mask &= df[col].isin(list(allowed))
    for col, bounds in (ranges or {}).items():
        if bounds is None or col not in df.columns:
            continue
        low, high = bounds
        series = df[col]
        mask &= series.between(low - 1e-9, high + 1e-9) | series.isna()
    return df[mask].copy()


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Compute every headline KPI from the (filtered) dataset."""
    total = int(len(df))
    if total == 0:
        return {
            "total_customers": 0,
            "churned_customers": 0,
            "retained_customers": 0,
            "churn_rate": 0.0,
            "retention_rate": 0.0,
            "revenue_at_risk": 0.0,
            "total_monthly_revenue": 0.0,
            "retained_monthly_revenue": 0.0,
            "revenue_at_risk_share": 0.0,
            "avg_tenure": 0.0,
            "avg_monthly_charges": 0.0,
            "avg_total_charges": 0.0,
            "avg_support_tickets": 0.0,
        }
    churned_mask = df["Churn"] == "Yes"
    churned = int(churned_mask.sum())
    retained = total - churned
    total_monthly = _nz(df["MonthlyCharges"].sum())
    at_risk = _nz(df.loc[churned_mask, "MonthlyCharges"].sum())
    return {
        "total_customers": total,
        "churned_customers": churned,
        "retained_customers": retained,
        "churn_rate": _safe_div(churned, total) * 100,
        "retention_rate": _safe_div(retained, total) * 100,
        "revenue_at_risk": at_risk,
        "total_monthly_revenue": total_monthly,
        "retained_monthly_revenue": total_monthly - at_risk,
        "revenue_at_risk_share": _safe_div(at_risk, total_monthly) * 100,
        "avg_tenure": _nz(df["TenureMonths"].mean()),
        "avg_monthly_charges": _nz(df["MonthlyCharges"].mean()),
        "avg_total_charges": _nz(df["TotalCharges"].mean()),
        "avg_support_tickets": _nz(df["SupportTickets"].mean()),
    }


# --------------------------------------------------------------------------- #
# Grouped churn tables
# --------------------------------------------------------------------------- #
GROUP_COLUMNS = [
    "Customers",
    "Churned",
    "Retained",
    "ChurnRate",
    "RetentionRate",
    "MonthlyRevenue",
    "RevenueAtRisk",
    "AvgMonthlyCharges",
    "AvgTenure",
    "AvgTickets",
]


def churn_by(
    df: pd.DataFrame,
    column: str,
    order: Optional[Sequence[str]] = None,
    keep_empty: bool = False,
) -> pd.DataFrame:
    """Churn statistics for each value of ``column``.

    Returns one row per group with customers, churned/retained counts, churn
    and retention rate (0-100), monthly revenue, revenue at risk and averages.
    """
    columns = [column] + GROUP_COLUMNS
    if df is None or df.empty or column not in df.columns:
        base = pd.DataFrame(columns=columns)
        if keep_empty and order:
            base = pd.DataFrame({column: list(order)})
            for c in GROUP_COLUMNS:
                base[c] = 0.0
            base["Customers"] = 0
            base["Churned"] = 0
            base["Retained"] = 0
            return base[columns]
        return base

    work = df.copy()
    work["_rar"] = work["MonthlyCharges"].fillna(0) * work["ChurnFlag"]
    grouped = work.groupby(column, observed=True).agg(
        Customers=("ChurnFlag", "size"),
        Churned=("ChurnFlag", "sum"),
        MonthlyRevenue=("MonthlyCharges", "sum"),
        RevenueAtRisk=("_rar", "sum"),
        AvgMonthlyCharges=("MonthlyCharges", "mean"),
        AvgTenure=("TenureMonths", "mean"),
        AvgTickets=("SupportTickets", "mean"),
    )
    out = grouped.reset_index()
    out[column] = out[column].astype(str)
    out["Churned"] = out["Churned"].astype(int)
    out["Retained"] = out["Customers"] - out["Churned"]
    out["ChurnRate"] = out["Churned"] / out["Customers"] * 100
    out["RetentionRate"] = 100 - out["ChurnRate"]

    if order:
        order = [str(o) for o in order]
        extra = [v for v in out[column] if v not in order]
        full_order = order + extra
        out = out.set_index(column).reindex(full_order).reset_index()
        out = out.rename(columns={"index": column})
        if keep_empty:
            for c in ["Customers", "Churned", "Retained"]:
                out[c] = out[c].fillna(0).astype(int)
            for c in ["MonthlyRevenue", "RevenueAtRisk"]:
                out[c] = out[c].fillna(0.0)
        else:
            out = out[out["Customers"].notna()].copy()
            for c in ["Customers", "Churned", "Retained"]:
                out[c] = out[c].astype(int)
    else:
        out = out.sort_values("Customers", ascending=False)
    return out[columns].reset_index(drop=True)


def churn_heatmap_data(
    df: pd.DataFrame, row: str, col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot tables of churn rate (NaN where no customers) and customer counts."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    counts = df.pivot_table(
        index=row, columns=col, values="ChurnFlag", aggfunc="size", fill_value=0
    )
    rates = df.pivot_table(index=row, columns=col, values="ChurnFlag", aggfunc="mean") * 100
    rates = rates.reindex(index=counts.index, columns=counts.columns)
    return rates, counts


# --------------------------------------------------------------------------- #
# Segmentation (explainable rules - NOT machine learning)
# --------------------------------------------------------------------------- #
SEGMENT_ORDER = [
    "Churned Customers",
    "High Support Customers",
    "New Customers",
    "At-Risk Customers",
    "High Revenue Customers",
    "Loyal Customers",
    "Established Customers",
]

ACTIVE_RISK_SEGMENTS = ["High Support Customers", "New Customers", "At-Risk Customers"]


@dataclass(frozen=True)
class SegmentThresholds:
    """Cut-offs used by the segmentation rules."""

    high_support_tickets: int = 3
    new_tenure_months: int = 6
    loyal_tenure_months: int = 24
    high_revenue_charge: float = float("inf")


def default_thresholds(df: pd.DataFrame) -> SegmentThresholds:
    """Derive the data-driven threshold (75th percentile of monthly charges).

    Compute this on the FULL dataset so segments stay stable when filters change.
    """
    if df is None or df.empty or df["MonthlyCharges"].dropna().empty:
        return SegmentThresholds()
    return SegmentThresholds(
        high_revenue_charge=float(df["MonthlyCharges"].quantile(0.75))
    )


def segment_rules(t: SegmentThresholds) -> Dict[str, str]:
    """Plain-English description of each rule (first matching rule wins)."""
    revenue = "-" if not np.isfinite(t.high_revenue_charge) else f"${t.high_revenue_charge:,.2f}"
    return {
        "Churned Customers": "Churn = Yes.",
        "High Support Customers": f"Active customer with {t.high_support_tickets}+ support tickets.",
        "New Customers": f"Active customer with tenure of {t.new_tenure_months} months or less.",
        "At-Risk Customers": "Active customer on a Month-to-Month contract (no long-term commitment).",
        "High Revenue Customers": f"Active customer paying at least {revenue}/month (top quartile of the full dataset).",
        "Loyal Customers": f"Active customer with {t.loyal_tenure_months}+ months of tenure or a Two Year contract.",
        "Established Customers": "Active customers who match none of the rules above.",
    }


SEGMENT_RECOMMENDATIONS: Dict[str, str] = {
    "Churned Customers": "Run exit surveys and win-back offers; study what these customers have in common.",
    "High Support Customers": "Contact proactively, fix root causes of repeated tickets and assign a success owner.",
    "New Customers": "Strengthen onboarding: welcome journey, first-month check-in and quick-start guidance.",
    "At-Risk Customers": "Offer incentives to move to annual or two-year plans and monitor engagement closely.",
    "High Revenue Customers": "Protect with premium service, account reviews and early renewal conversations.",
    "Loyal Customers": "Reward loyalty, request referrals and reviews, and consider upsell opportunities.",
    "Established Customers": "Maintain regular communication and watch for rising ticket volume or plan downgrades.",
}


def assign_segments(
    df: pd.DataFrame, thresholds: Optional[SegmentThresholds] = None
) -> pd.Series:
    """Assign each customer to exactly one segment (first matching rule wins)."""
    if df is None or df.empty:
        return pd.Series([], dtype="object", index=df.index if df is not None else None)
    t = thresholds or default_thresholds(df)
    churned = df["Churn"] == "Yes"
    conditions = [
        churned,
        df["SupportTickets"] >= t.high_support_tickets,
        df["TenureMonths"] <= t.new_tenure_months,
        df["ContractType"].str.lower() == "month-to-month",
        df["MonthlyCharges"] >= t.high_revenue_charge,
        (df["TenureMonths"] >= t.loyal_tenure_months)
        | (df["ContractType"].str.lower() == "two year"),
    ]
    choices = SEGMENT_ORDER[:6]
    values = np.select(conditions, choices, default=SEGMENT_ORDER[6])
    return pd.Series(values, index=df.index, name="Segment")


def segment_summary(df: pd.DataFrame, segment_col: str = "Segment") -> pd.DataFrame:
    """One row per segment (including empty ones) with metrics and a recommendation."""
    columns = [
        "Segment",
        "Customers",
        "SharePct",
        "AvgTenure",
        "AvgMonthlyCharges",
        "AvgTickets",
        "MonthlyRevenue",
        "Recommendation",
    ]
    rows = []
    total = len(df) if df is not None else 0
    for name in SEGMENT_ORDER:
        part = df[df[segment_col] == name] if total else pd.DataFrame()
        n = len(part)
        rows.append(
            {
                "Segment": name,
                "Customers": n,
                "SharePct": _safe_div(n, total) * 100,
                "AvgTenure": _nz(part["TenureMonths"].mean()) if n else 0.0,
                "AvgMonthlyCharges": _nz(part["MonthlyCharges"].mean()) if n else 0.0,
                "AvgTickets": _nz(part["SupportTickets"].mean()) if n else 0.0,
                "MonthlyRevenue": _nz(part["MonthlyCharges"].sum()) if n else 0.0,
                "Recommendation": SEGMENT_RECOMMENDATIONS[name],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def active_revenue_exposed(df: pd.DataFrame, segment_col: str = "Segment") -> float:
    """Monthly charges of ACTIVE customers sitting in the risk-flagged segments."""
    if df is None or df.empty or segment_col not in df.columns:
        return 0.0
    mask = df[segment_col].isin(ACTIVE_RISK_SEGMENTS)
    return _nz(df.loc[mask, "MonthlyCharges"].sum())


def ticket_order(df: pd.DataFrame) -> List[str]:
    """Support-ticket labels in ascending numeric order."""
    if df is None or df.empty or "TicketGroup" not in df.columns:
        return []
    labels = [v for v in df["TicketGroup"].unique() if v != "Unknown"]
    labels = sorted(labels, key=lambda v: int(v))
    if (df["TicketGroup"] == "Unknown").any():
        labels.append("Unknown")
    return labels


def risk_flags(df: pd.DataFrame, thresholds: Optional[SegmentThresholds] = None) -> pd.Series:
    """Per-customer text listing the churn-risk traits that apply (any churn status)."""
    if df is None or df.empty:
        return pd.Series([], dtype="object", index=df.index if df is not None else None)
    t = thresholds or default_thresholds(df)
    flags = []
    for _, row in df.iterrows():
        found = []
        if row["SupportTickets"] >= t.high_support_tickets:
            found.append("High support")
        if row["TenureMonths"] <= t.new_tenure_months:
            found.append("New")
        if str(row["ContractType"]).lower() == "month-to-month":
            found.append("Month-to-Month")
        flags.append(", ".join(found) if found else "None")
    return pd.Series(flags, index=df.index, name="RiskFlags")


def flag_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """How often each risk trait appears among churned vs retained customers."""
    columns = ["Trait", "ChurnedWith", "ChurnedTotal", "RetainedWith", "RetainedTotal"]
    if df is None or df.empty or "RiskFlags" not in df.columns:
        return pd.DataFrame(columns=columns)
    churned = df[df["Churn"] == "Yes"]
    retained = df[df["Churn"] == "No"]
    rows = []
    for trait in ("High support", "New", "Month-to-Month"):
        rows.append(
            {
                "Trait": trait,
                "ChurnedWith": int(churned["RiskFlags"].str.contains(trait, regex=False).sum()),
                "ChurnedTotal": int(len(churned)),
                "RetainedWith": int(retained["RiskFlags"].str.contains(trait, regex=False).sum()),
                "RetainedTotal": int(len(retained)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def revenue_notes(df: pd.DataFrame) -> List[str]:
    """Plain-language interpretation of the revenue figures."""
    if df is None or df.empty:
        return []
    k = compute_kpis(df)
    notes = [
        f"Monthly recurring charges in this selection total {fmt_money(k['total_monthly_revenue'])}. "
        f"{fmt_money(k['revenue_at_risk'])} ({fmt_pct(k['revenue_at_risk_share'])}) of it is billed to customers marked as churned - "
        "the dashboard calls this Revenue at Risk."
    ]
    contract = churn_by(df, "ContractType")
    if len(contract) > 1:
        top = contract.sort_values("MonthlyRevenue", ascending=False).iloc[0]
        notes.append(
            f"{top['ContractType']} contracts bring in the most monthly charges ({fmt_money(top['MonthlyRevenue'])}, "
            f"{int(top['Customers'])} customers)."
        )
    risk = churn_by(df, "PaymentMethod")
    risk = risk[risk["RevenueAtRisk"] > 0]
    if not risk.empty:
        top = risk.sort_values("RevenueAtRisk", ascending=False).iloc[0]
        notes.append(
            f"Most Revenue at Risk is billed through {top['PaymentMethod']} ({fmt_money(top['RevenueAtRisk'])})."
        )
    valid = df.dropna(subset=["MonthlyCharges", "TenureMonths", "TotalCharges"])
    if not valid.empty:
        expected = valid["MonthlyCharges"] * valid["TenureMonths"]
        match = (valid["TotalCharges"] - expected).abs() <= 0.05
        notes.append(
            f"TotalCharges is lifetime billing, not monthly revenue. For {fmt_pct(match.mean() * 100, 0)} of customers it equals "
            "MonthlyCharges x TenureMonths, so it is analysed separately and never added to monthly figures."
        )
    return notes


# --------------------------------------------------------------------------- #
# Data quality helpers (used by the Data Explorer)
# --------------------------------------------------------------------------- #
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values and datatype for every column."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Column", "Datatype", "Missing", "MissingPct"])
    out = pd.DataFrame(
        {
            "Column": df.columns,
            "Datatype": [str(t) for t in df.dtypes],
            "Missing": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    out["MissingPct"] = out["Missing"] / len(df) * 100
    return out


def duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """All rows that are part of a duplicate group (full-row duplicates)."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df.duplicated(keep=False)]


def numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics of the numeric columns."""
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    if df is None or df.empty or not cols:
        return pd.DataFrame()
    return df[cols].describe().T.reset_index().rename(columns={"index": "Column"})


# --------------------------------------------------------------------------- #
# Insights and recommendations (all text is generated from the data)
# --------------------------------------------------------------------------- #
def _top_groups(table: pd.DataFrame, metric: str = "ChurnRate") -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    best = table[metric].max()
    return table[np.isclose(table[metric], best)]


def _describe_groups(label_col: str, rows: pd.DataFrame) -> str:
    names = " / ".join(rows[label_col].astype(str))
    return names


def _group_detail(rows: pd.DataFrame) -> str:
    parts = [
        f"{r.iloc[0]}: {int(r['Churned'])} of {int(r['Customers'])} churned"
        for _, r in rows.iterrows()
    ]
    return "; ".join(parts)


def _highest_churn_insight(
    df: pd.DataFrame, column: str, title: str, icon: str, noun: str, order=None
) -> Dict[str, str]:
    table = churn_by(df, column, order=order)
    if table.empty:
        return {"icon": icon, "title": title, "headline": "No data", "detail": "No customers in the current selection.", "tone": "info"}
    if len(table) == 1:
        only = table.iloc[0]
        return {
            "icon": icon,
            "title": title,
            "headline": str(only[column]),
            "detail": f"Only one {noun} ({only[column]}) in the current selection, so no comparison is possible. Churn rate there: {fmt_pct(only['ChurnRate'])}.",
            "tone": "info",
        }
    top = _top_groups(table)
    rate = float(top["ChurnRate"].iloc[0])
    if rate == 0:
        return {
            "icon": icon,
            "title": title,
            "headline": "No churn",
            "detail": f"No {noun} group has churned customers in the current selection.",
            "tone": "good",
        }
    tied = len(top) > 1
    label_col = column
    detail = (
        f"{fmt_pct(rate)} churn rate"
        + (" (tie between groups)" if tied else "")
        + f". {_group_detail(top[[label_col, 'Churned', 'Customers']])}. Small groups can swing percentages heavily."
    )
    return {
        "icon": icon,
        "title": title,
        "headline": _describe_groups(label_col, top),
        "detail": detail,
        "tone": "danger",
    }


def generate_insights(df: pd.DataFrame, segment_col: str = "Segment") -> List[Dict[str, str]]:
    """Insight cards calculated from the current selection."""
    if df is None or df.empty:
        return []
    insights: List[Dict[str, str]] = []
    kpis = compute_kpis(df)

    insights.append(
        _highest_churn_insight(df, "ContractType", "Contract type with highest churn", "📄", "contract type")
    )
    insights.append(
        _highest_churn_insight(df, "PaymentMethod", "Payment method with highest churn", "💳", "payment method")
    )
    insights.append(
        _highest_churn_insight(
            df, "TenureGroup", "Tenure group with highest churn", "⏳", "tenure group", order=TENURE_LABELS
        )
    )
    insights.append(
        _highest_churn_insight(df, "AgeGroup", "Age group with highest churn", "🎂", "age group", order=AGE_LABELS)
    )

    churned = df[df["Churn"] == "Yes"]
    retained = df[df["Churn"] == "No"]

    # Tenure of churned customers
    if churned.empty:
        insights.append(
            {"icon": "🕒", "title": "Average tenure of churned customers", "headline": "No churn", "detail": "There are no churned customers in the current selection.", "tone": "good"}
        )
    else:
        detail = f"Churned customers stayed {fmt_num(churned['TenureMonths'].mean())} months on average"
        if not retained.empty:
            detail += f" versus {fmt_num(retained['TenureMonths'].mean())} months for retained customers"
        insights.append(
            {"icon": "🕒", "title": "Average tenure of churned customers", "headline": f"{fmt_num(churned['TenureMonths'].mean())} months", "detail": detail + ".", "tone": "warn"}
        )

    # Support tickets
    if churned.empty:
        insights.append(
            {"icon": "🎧", "title": "Support tickets of churned customers", "headline": "No churn", "detail": "There are no churned customers in the current selection.", "tone": "good"}
        )
    else:
        total_t = int(churned["SupportTickets"].sum())
        detail = f"Churned customers raised {total_t} ticket(s) in total, {fmt_num(churned['SupportTickets'].mean())} per customer"
        if not retained.empty:
            detail += f", compared with {fmt_num(retained['SupportTickets'].mean())} per retained customer"
        insights.append(
            {"icon": "🎧", "title": "Support tickets of churned customers", "headline": f"{total_t} tickets", "detail": detail + ".", "tone": "warn"}
        )

    # Segments
    if segment_col in df.columns:
        seg = segment_summary(df, segment_col)
        seg_active = seg[seg["Segment"] != "Churned Customers"]
        exposed = active_revenue_exposed(df, segment_col)
        if seg_active["MonthlyRevenue"].sum() > 0:
            top_seg = seg_active.sort_values("MonthlyRevenue", ascending=False).iloc[0]
            insights.append(
                {
                    "icon": "🧩",
                    "title": "Highest-revenue active segment",
                    "headline": str(top_seg["Segment"]),
                    "detail": f"{fmt_money(top_seg['MonthlyRevenue'])} monthly charges across {int(top_seg['Customers'])} customer(s).",
                    "tone": "info",
                }
            )
        insights.append(
            {
                "icon": "🚨",
                "title": "Active revenue exposed in risk segments",
                "headline": fmt_money(exposed),
                "detail": "Monthly charges of ACTIVE customers flagged as High Support, New or At-Risk by the rule-based segments. This is exposure, not a forecast.",
                "tone": "warn" if exposed > 0 else "good",
            }
        )

    # Best subscription
    sub = churn_by(df, "SubscriptionType")
    if sub.empty:
        pass
    elif len(sub) == 1:
        insights.append(
            {"icon": "🏆", "title": "Best-performing subscription", "headline": str(sub.iloc[0]["SubscriptionType"]), "detail": "Only one subscription type in the current selection, so no comparison is possible.", "tone": "info"}
        )
    else:
        best = sub.sort_values(["ChurnRate", "MonthlyRevenue"], ascending=[True, False]).iloc[0]
        insights.append(
            {
                "icon": "🏆",
                "title": "Best-performing subscription",
                "headline": str(best["SubscriptionType"]),
                "detail": f"Lowest churn rate ({fmt_pct(best['ChurnRate'])}, {int(best['Churned'])} of {int(best['Customers'])}) with {fmt_money(best['MonthlyRevenue'])} monthly charges (ties are broken by revenue).",
                "tone": "good",
            }
        )

    # Revenue at risk share
    insights.append(
        {
            "icon": "💰",
            "title": "Share of monthly charges tied to churned customers",
            "headline": fmt_pct(kpis["revenue_at_risk_share"]),
            "detail": f"{fmt_money(kpis['revenue_at_risk'])} of {fmt_money(kpis['total_monthly_revenue'])} monthly recurring charges belong to churned customers in this dataset.",
            "tone": "danger" if kpis["revenue_at_risk"] > 0 else "good",
        }
    )
    return insights


def generate_recommendations(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Actionable recommendations derived from the current selection."""
    recs: List[Dict[str, str]] = []
    if df is None or df.empty:
        return recs
    churned = df[df["Churn"] == "Yes"]
    retained = df[df["Churn"] == "No"]

    if churned.empty:
        recs.append(
            {
                "priority": "Low",
                "title": "No churn in this selection",
                "action": "Keep monitoring. Widen the filters to compare against customers who churned.",
                "evidence": "The current selection contains no churned customers.",
            }
        )
    else:
        contract = churn_by(df, "ContractType")
        top_c = _top_groups(contract)
        if len(contract) > 1 and not top_c.empty and top_c["ChurnRate"].iloc[0] > 0:
            names = _describe_groups("ContractType", top_c)
            recs.append(
                {
                    "priority": "High",
                    "title": f"Convert {names} customers to longer commitments",
                    "action": "Offer discounts, bonus features or loyalty credits for moving to annual or multi-year plans.",
                    "evidence": f"Highest churn by contract: {_group_detail(top_c[['ContractType', 'Churned', 'Customers']])}.",
                }
            )
        payment = churn_by(df, "PaymentMethod")
        top_p = _top_groups(payment)
        if len(payment) > 1 and not top_p.empty and top_p["ChurnRate"].iloc[0] > 0:
            low_p = payment.sort_values(["ChurnRate", "Customers"], ascending=[True, False]).iloc[0]
            recs.append(
                {
                    "priority": "Medium",
                    "title": f"Review the {_describe_groups('PaymentMethod', top_p)} payment experience",
                    "action": f"Check for payment friction and encourage auto-pay, for example by promoting {low_p['PaymentMethod']}, which has the lowest churn here.",
                    "evidence": f"Highest churn by payment method: {_group_detail(top_p[['PaymentMethod', 'Churned', 'Customers']])}.",
                }
            )
        if not retained.empty and churned["SupportTickets"].mean() > retained["SupportTickets"].mean():
            recs.append(
                {
                    "priority": "High",
                    "title": "Act early on repeated support tickets",
                    "action": "Trigger proactive outreach when a customer reaches 3 tickets and analyse the root causes of recurring issues.",
                    "evidence": f"Churned customers average {fmt_num(churned['SupportTickets'].mean())} tickets versus {fmt_num(retained['SupportTickets'].mean())} for retained customers.",
                }
            )
        tenure = churn_by(df, "TenureGroup", order=TENURE_LABELS)
        top_t = _top_groups(tenure)
        if len(tenure) > 1 and not top_t.empty and top_t["ChurnRate"].iloc[0] > 0:
            recs.append(
                {
                    "priority": "High",
                    "title": f"Improve onboarding and early engagement ({_describe_groups('TenureGroup', top_t)})",
                    "action": "Add a welcome journey, usage check-ins and success milestones during the first months.",
                    "evidence": f"Highest churn by tenure: {_group_detail(top_t[['TenureGroup', 'Churned', 'Customers']])}.",
                }
            )
        sub = churn_by(df, "SubscriptionType")
        top_s = _top_groups(sub)
        if len(sub) > 1 and not top_s.empty and top_s["ChurnRate"].iloc[0] > 0:
            recs.append(
                {
                    "priority": "Medium",
                    "title": f"Reassess value of the {_describe_groups('SubscriptionType', top_s)} plan",
                    "action": "Review plan features and pricing, and test upgrade paths or bundles for these customers.",
                    "evidence": f"Highest churn by subscription: {_group_detail(top_s[['SubscriptionType', 'Churned', 'Customers']])}.",
                }
            )

    recs.append(
        {
            "priority": "Medium",
            "title": "Validate findings on a larger dataset",
            "action": "Re-run this dashboard on a bigger customer export before taking pricing or policy decisions.",
            "evidence": f"The current selection has only {len(df)} customer(s); small samples give unstable percentages.",
        }
    )
    return recs
