"""Data loading, validation, KPI and insight logic for the Churn KPI Dashboard.

Everything here is plain pandas/numpy (no Streamlit), so it can be tested on its own.
Nothing in this module invents data or silently changes values in the source file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = "customer_churn_cleaned.csv"

REQUIRED_COLUMNS = [
    "CustomerID", "Gender", "Age", "TenureMonths", "SubscriptionType",
    "MonthlyCharges", "TotalCharges", "ContractType", "SupportTickets",
    "PaymentMethod", "Churn",
]
NUMERIC_COLUMNS = ["Age", "TenureMonths", "MonthlyCharges", "TotalCharges", "SupportTickets"]
VALID_CHURN = ["Yes", "No"]
VALID_CONTRACTS = ["Month-to-Month", "One Year", "Two Year"]
KNOWN_PAYMENTS = ["Credit Card", "Bank Transfer", "UPI", "Debit Card"]
AGE_BINS = [-np.inf, 29, 39, 49, np.inf]
AGE_LABELS = ["Under 30", "30-39", "40-49", "50+"]


# --------------------------------------------------------------------------- formatting
def fmt_inr(value: float | None) -> str:
    """Format a number as INR with Indian digit grouping (e.g. 1,23,456.00)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    sign = "-" if value < 0 else ""
    whole, dec = f"{abs(value):.2f}".split(".")
    if len(whole) > 3:
        whole = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", whole[:-3]) + "," + whole[-3:]
    return f"{sign}\u20b9{whole}.{dec}"


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}f}%"


# --------------------------------------------------------------------------- file discovery
def find_csv(base_dir: str | Path) -> Path | None:
    """Return the default CSV if present, otherwise the first CSV found next to the app."""
    base = Path(base_dir)
    default = base / DEFAULT_CSV
    if default.exists():
        return default
    candidates = sorted(base.glob("*.csv"))
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------- validation
@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[tuple[str, str, str]] = field(default_factory=list)  # (check, status, detail)
    modifications: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, check: str, passed: bool, detail: str, error: bool = True) -> None:
        status = "Passed" if passed else ("Failed" if error else "Warning")
        self.checks.append((check, status, detail))
        if not passed:
            (self.errors if error else self.warnings).append(f"{check}: {detail}")


def validate(raw: pd.DataFrame) -> ValidationReport:
    """Validate the raw dataframe. Never alters it."""
    rep = ValidationReport()
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    rep.add("Required columns", not missing_cols,
            f"Missing: {', '.join(missing_cols)}" if missing_cols else f"All {len(REQUIRED_COLUMNS)} present")
    if missing_cols:
        return rep
    rep.add("Not empty", len(raw) > 0, f"{len(raw)} rows")
    if raw.empty:
        return rep

    nulls = raw[REQUIRED_COLUMNS].isna().sum()
    nulls = nulls[nulls > 0]
    rep.add("Missing values", nulls.empty,
            "None found" if nulls.empty else ", ".join(f"{k}: {v}" for k, v in nulls.items()),
            error=False)

    dup = raw["CustomerID"].duplicated(keep=False)
    rep.add("Duplicate CustomerID", not dup.any(),
            "None found" if not dup.any() else f"Duplicated IDs: {', '.join(raw.loc[dup, 'CustomerID'].astype(str).unique())}")

    bad_numeric = []
    for col in NUMERIC_COLUMNS:
        conv = pd.to_numeric(raw[col], errors="coerce")
        if (conv.isna() & raw[col].notna()).any():
            bad_numeric.append(col)
    rep.add("Numeric columns are numeric", not bad_numeric,
            "All numeric" if not bad_numeric else f"Non-numeric text in: {', '.join(bad_numeric)}")
    if bad_numeric:
        return rep
    num = raw[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    neg_charges = ((num["MonthlyCharges"] < 0) | (num["TotalCharges"] < 0)).sum()
    rep.add("Negative charges", neg_charges == 0, "None found" if neg_charges == 0 else f"{neg_charges} rows")
    bad_tenure = (num["TenureMonths"] < 0).sum()
    rep.add("Valid tenure", bad_tenure == 0, "All tenure values >= 0" if bad_tenure == 0 else f"{bad_tenure} negative values")
    bad_age = ((num["Age"] < 0) | (num["Age"] > 120)).sum()
    rep.add("Valid age", bad_age == 0, "All ages between 0 and 120" if bad_age == 0 else f"{bad_age} implausible values")
    bad_tix = (num["SupportTickets"] < 0).sum()
    rep.add("Valid support tickets", bad_tix == 0, "All values >= 0" if bad_tix == 0 else f"{bad_tix} negative values")

    churn_bad = sorted(set(raw["Churn"].dropna().astype(str)) - set(VALID_CHURN))
    rep.add("Churn values", not churn_bad and raw["Churn"].notna().all(),
            "Only Yes / No" if not churn_bad and raw["Churn"].notna().all()
            else f"Unexpected or missing: {', '.join(churn_bad) or 'blank values'}")
    contract_bad = sorted(set(raw["ContractType"].dropna().astype(str)) - set(VALID_CONTRACTS))
    rep.add("ContractType values", not contract_bad and raw["ContractType"].notna().all(),
            "Only Month-to-Month / One Year / Two Year" if not contract_bad and raw["ContractType"].notna().all()
            else f"Unexpected or missing: {', '.join(contract_bad) or 'blank values'}")
    pay_bad = sorted(set(raw["PaymentMethod"].dropna().astype(str)) - set(KNOWN_PAYMENTS))
    rep.add("PaymentMethod values", not pay_bad, "All values recognised" if not pay_bad
            else f"New payment methods (kept as-is): {', '.join(pay_bad)}", error=False)

    expected = num["MonthlyCharges"] * num["TenureMonths"]
    mismatch = ((expected - num["TotalCharges"]).abs() > 0.01 * expected.clip(lower=1)).sum()
    rep.add("TotalCharges = MonthlyCharges x Tenure", mismatch == 0,
            "Consistent for every row" if mismatch == 0 else f"{mismatch} rows differ by more than 1% (kept as-is)",
            error=False)
    return rep


def prepare(raw: pd.DataFrame, rep: ValidationReport) -> pd.DataFrame:
    """Type-cast a *valid* dataframe for analysis and record what was done."""
    df = raw.copy()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rep.modifications.append("Numeric columns were parsed as numbers (no values were changed).")
    rep.modifications.append("Age groups (Under 30, 30-39, 40-49, 50+) were derived in a new column 'AgeGroup'.")
    if df[NUMERIC_COLUMNS].isna().any().any():
        rep.modifications.append("Rows with missing numeric values are kept; averages and sums skip the blanks.")
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    return df


def load_dataset(source) -> tuple[pd.DataFrame | None, ValidationReport]:
    """Read a CSV path or file-like object, validate it and return (dataframe or None, report)."""
    rep = ValidationReport()
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # unreadable file, bad encoding, etc.
        rep.add("CSV readable", False, f"Could not read the file ({exc})")
        return None, rep
    rep = validate(raw)
    if not rep.ok:
        return None, rep
    return prepare(raw, rep), rep


# --------------------------------------------------------------------------- filtering
def apply_filters(df: pd.DataFrame, contract: str, tenure: tuple[float, float], payment: str) -> pd.DataFrame:
    mask = df["TenureMonths"].between(tenure[0], tenure[1])
    if contract != "All":
        mask &= df["ContractType"] == contract
    if payment != "All":
        mask &= df["PaymentMethod"] == payment
    return df[mask]


# --------------------------------------------------------------------------- KPIs
def compute_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    churned = df[df["Churn"] == "Yes"]
    retained = df[df["Churn"] == "No"]
    n_ch, n_re = len(churned), len(retained)
    return {
        "total": total,
        "churned": n_ch,
        "retained": n_re,
        "churn_rate": n_ch / total * 100 if total else np.nan,
        "retention_rate": n_re / total * 100 if total else np.nan,
        "revenue_at_risk": float(churned["MonthlyCharges"].sum()),
        "monthly_revenue": float(df["MonthlyCharges"].sum()),
        "avg_tenure": float(df["TenureMonths"].mean()) if total else np.nan,
        "avg_tenure_churned": float(churned["TenureMonths"].mean()) if n_ch else np.nan,
        "avg_tenure_retained": float(retained["TenureMonths"].mean()) if n_re else np.nan,
        "avg_tickets_churned": float(churned["SupportTickets"].mean()) if n_ch else np.nan,
        "avg_tickets_retained": float(retained["SupportTickets"].mean()) if n_re else np.nan,
    }


def tenure_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Average / min / max tenure for churned vs retained customers."""
    rows = []
    for label, flag in (("Churned", "Yes"), ("Retained", "No")):
        s = df.loc[df["Churn"] == flag, "TenureMonths"]
        rows.append({"Group": label, "Customers": len(s),
                     "Average (months)": s.mean() if len(s) else np.nan,
                     "Minimum": s.min() if len(s) else np.nan,
                     "Maximum": s.max() if len(s) else np.nan})
    return pd.DataFrame(rows)


def segment_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Customers, churned, retained and churn rate per value of `col` (empty segments dropped)."""
    if df.empty:
        return pd.DataFrame(columns=[col, "Customers", "Churned", "Retained", "Churn Rate (%)",
                                     "Monthly Revenue", "Revenue at Risk"])
    g = df.assign(_churn=(df["Churn"] == "Yes").astype(int),
                  _risk=np.where(df["Churn"] == "Yes", df["MonthlyCharges"], 0.0))
    out = g.groupby(col, observed=True).agg(
        Customers=("CustomerID", "count"), Churned=("_churn", "sum"),
        **{"Monthly Revenue": ("MonthlyCharges", "sum"), "Revenue at Risk": ("_risk", "sum")},
    ).reset_index()
    out["Retained"] = out["Customers"] - out["Churned"]
    out["Churn Rate (%)"] = out["Churned"] / out["Customers"] * 100
    out[col] = out[col].astype(str)
    return out[[col, "Customers", "Churned", "Retained", "Churn Rate (%)", "Monthly Revenue", "Revenue at Risk"]]


# --------------------------------------------------------------------------- insights
SEGMENT_COLUMNS = {
    "SubscriptionType": "subscription type", "ContractType": "contract type",
    "PaymentMethod": "payment method", "Gender": "gender", "AgeGroup": "age group",
}


def _top(seg: pd.DataFrame, col: str) -> tuple[list[str], float]:
    top_rate = seg["Churn Rate (%)"].max()
    return seg.loc[seg["Churn Rate (%)"] == top_rate, col].tolist(), float(top_rate)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def generate_insights(df: pd.DataFrame) -> list[dict]:
    """Descriptive insights computed from `df`. Each has category / observed / interpretation."""
    if df.empty:
        return []
    k = compute_kpis(df)
    out: list[dict] = []

    # Contract type
    seg = segment_table(df, "ContractType")
    names, rate = _top(seg, "ContractType")
    if len(seg) == 1:
        obs = f"Only {seg.iloc[0]['ContractType']} customers are in view: {seg.iloc[0]['Churned']} of {seg.iloc[0]['Customers']} churned ({pct(rate)})."
    elif seg["Churn Rate (%)"].nunique() == 1:
        obs = f"All contract types in view have the same churn rate ({pct(rate)})."
    else:
        n_top = int(seg.loc[seg["ContractType"].isin(names), "Customers"].sum())
        obs = f"{_join(names)} has the highest churn rate at {pct(rate)} (based on {n_top} customer{'s' if n_top != 1 else ''})."
    out.append({"category": "Contract type", "observed": obs,
                "interpretation": "Contract length may be linked to how easily customers can leave, but this data alone cannot show why."})

    # Highest-churn segment across all dimensions
    best = None
    for col, label in SEGMENT_COLUMNS.items():
        s = segment_table(df, col)
        if s.empty:
            continue
        for _, r in s.iterrows():
            key = (r["Churn Rate (%)"], r["Customers"])
            if best is None or key > best[0]:
                best = (key, label, r[col], r)
    if best:
        _, label, name, r = best
        out.append({"category": "Customer group with highest churn",
                    "observed": f"Among the segments checked, {label} '{name}' shows the highest churn: {int(r['Churned'])} of {int(r['Customers'])} customers ({pct(r['Churn Rate (%)'])}).",
                    "interpretation": "Several segments overlap (for example, many Basic customers may also be Month-to-Month), so this group may simply reflect another factor."})

    # Tenure
    if k["churned"] and k["retained"]:
        diff = k["avg_tenure_retained"] - k["avg_tenure_churned"]
        direction = "longer" if diff > 0 else "shorter" if diff < 0 else "the same"
        obs = (f"Churned customers average {k['avg_tenure_churned']:.1f} months of tenure versus "
               f"{k['avg_tenure_retained']:.1f} months for retained customers"
               + (f" (retained customers' tenure is {abs(diff):.1f} months {direction})." if diff else " (no difference)."))
        interp = "Churned customers tended to leave early in the relationship; tenure is measured up to the churn date, so it partly reflects when they left." if diff > 0 else "No clear early-exit pattern appears in this selection."
    else:
        obs = "A tenure comparison needs both churned and retained customers in the current selection."
        interp = "Widen the filters to compare both groups."
    out.append({"category": "Tenure", "observed": obs, "interpretation": interp})

    # Support tickets
    if k["churned"] and k["retained"]:
        diff = k["avg_tickets_churned"] - k["avg_tickets_retained"]
        obs = (f"Churned customers raised {k['avg_tickets_churned']:.1f} support tickets on average versus "
               f"{k['avg_tickets_retained']:.1f} for retained customers.")
        interp = ("Higher ticket counts go together with churn here, which may point to unresolved service problems. "
                  "This is an association, not proof that tickets cause churn." if diff > 0
                  else "Churned customers did not raise more tickets than retained ones in this selection.")
    else:
        obs = "A support-ticket comparison needs both churned and retained customers in the current selection."
        interp = "Widen the filters to compare both groups."
    out.append({"category": "Support tickets", "observed": obs, "interpretation": interp})

    # Revenue at risk
    share = k["revenue_at_risk"] / k["monthly_revenue"] * 100 if k["monthly_revenue"] else np.nan
    out.append({"category": "Monthly revenue at risk",
                "observed": f"Churned customers account for {fmt_inr(k['revenue_at_risk'])} of {fmt_inr(k['monthly_revenue'])} monthly charges in view ({pct(share)}).",
                "interpretation": "This is monthly recurring revenue in this small sample only. It is not lifetime value and not a forecast."})

    # Payment method
    ps = segment_table(df, "PaymentMethod")
    names, rate = _top(ps, "PaymentMethod")
    if len(ps) == 1:
        obs = f"Only {ps.iloc[0]['PaymentMethod']} is in view: {pct(rate)} churn."
    elif ps["Churn Rate (%)"].nunique() == 1:
        obs = f"All payment methods in view have the same churn rate ({pct(rate)})."
    else:
        obs = f"{_join(names)} shows the highest churn rate among payment methods ({pct(rate)})."
    out.append({"category": "Payment method", "observed": obs,
                "interpretation": "With only a few customers per payment method, differences can be coincidence."})
    return out


def generate_recommendations(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Rule-based recommendations plus a transparent watchlist of retained customers.

    Watchlist rule: a retained customer is flagged on each of three signals that churned customers show
    in this data (Month-to-Month contract, tickets >= churned average, tenure <= churned average) and is
    listed when two or more flags apply. It is a review list, not a prediction.
    """
    empty_watch = pd.DataFrame(columns=["CustomerID", "ContractType", "TenureMonths", "SupportTickets", "MonthlyCharges", "Flags"])
    if df.empty:
        return [], empty_watch
    k = compute_kpis(df)
    recs: list[dict] = []
    retained = df[df["Churn"] == "No"].copy()

    m2m = df[df["ContractType"] == "Month-to-Month"]
    if len(m2m):
        m2m_ch = int((m2m["Churn"] == "Yes").sum())
        m2m_ret = m2m[m2m["Churn"] == "No"]
        recs.append({"title": "Review Month-to-Month customers",
                     "basis": f"{m2m_ch} of {len(m2m)} Month-to-Month customers churned; {len(m2m_ret)} remain, worth {fmt_inr(m2m_ret['MonthlyCharges'].sum())} per month.",
                     "action": ("Consider offering longer-term plans or check-in calls to the remaining Month-to-Month customers."
                                if len(m2m_ret) else
                                "No Month-to-Month customers remain in this selection; review why this group left (pricing, onboarding, service) before the next cohort.")})

    if k["churned"]:
        t_thr, ten_thr = k["avg_tickets_churned"], k["avg_tenure_churned"]
        hi_t = retained[retained["SupportTickets"] >= t_thr]
        recs.append({"title": "Investigate customers with high support ticket counts",
                     "basis": f"Churned customers averaged {t_thr:.1f} tickets; {len(hi_t)} retained customer(s) have at least that many.",
                     "action": "Review those tickets for unresolved issues and follow up proactively."})
        lo_t = retained[retained["TenureMonths"] <= ten_thr]
        recs.append({"title": "Identify customers with low tenure",
                     "basis": f"Churned customers averaged {ten_thr:.1f} months of tenure; {len(lo_t)} retained customer(s) are at or below that.",
                     "action": "Strengthen onboarding and early-life check-ins for newer customers."})
        retained["_m2m"] = retained["ContractType"] == "Month-to-Month"
        retained["_tix"] = retained["SupportTickets"] >= t_thr
        retained["_ten"] = retained["TenureMonths"] <= ten_thr
        retained["_n"] = retained[["_m2m", "_tix", "_ten"]].sum(axis=1)
        watch = retained[retained["_n"] >= 2].copy()
        watch["Flags"] = watch.apply(lambda r: ", ".join(
            n for n, f in (("Month-to-Month", r["_m2m"]), ("High tickets", r["_tix"]), ("Low tenure", r["_ten"])) if f), axis=1)
        watch = watch[["CustomerID", "ContractType", "TenureMonths", "SupportTickets", "MonthlyCharges", "Flags"]]
        recs.append({"title": "Improve retention outreach",
                     "basis": f"{len(watch)} retained customer(s) show two or more of the signals above.",
                     "action": "Contact the watchlist customers first; treat the list as a starting point for review, not a prediction."})
    else:
        watch = empty_watch
        recs.append({"title": "No churned customers in this selection",
                     "basis": "There are no churned customers under the current filters, so risk signals cannot be compared.",
                     "action": "Widen the filters to see churn-related recommendations."})

    recs.append({"title": "Monitor monthly recurring revenue at risk",
                 "basis": f"{fmt_inr(k['revenue_at_risk'])} per month comes from {k['churned']} churned customer(s) in view.",
                 "action": "Track this value each month as new data arrives; with only a small sample, avoid acting on small shifts."})
    return recs, watch.reset_index(drop=True)
