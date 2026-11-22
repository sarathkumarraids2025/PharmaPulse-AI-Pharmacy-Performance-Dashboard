"""Customer Churn Analytics - Business KPI Dashboard (Streamlit)."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
import utils

BASE_DIR = Path(__file__).parent
LOGO = BASE_DIR / "assets" / "dashboard_logo.png"
PAGES = ["Overview", "Churn Analysis", "Revenue Analysis", "Customer Segmentation", "Customer Data", "About Dataset"]

st.set_page_config(page_title="Customer Churn Analytics", page_icon=str(LOGO) if LOGO.exists() else "📊",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.stApp { background: #F4F6FB; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E2E8F0; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
h1, h2, h3 { color: #0F172A; letter-spacing: -0.01em; }
.hdr { background: linear-gradient(120deg,#2563EB 0%,#7C3AED 100%); color: #fff; border-radius: 18px;
       padding: 22px 28px; box-shadow: 0 6px 18px rgba(37,99,235,.22); margin-bottom: 14px; }
.hdr h1 { color:#fff; margin:0; font-size: 1.9rem; }
.hdr p { margin: 4px 0 12px 0; color: #E0E7FF; font-size: 1rem; }
.chip { display:inline-block; background: rgba(255,255,255,.18); border-radius: 999px; padding: 3px 12px;
        margin: 0 8px 4px 0; font-size: .82rem; }
.kpi { background:#fff; border-radius:16px; padding:18px 20px; border:1px solid #E2E8F0;
       box-shadow: 0 2px 10px rgba(15,23,42,.06); border-top: 5px solid var(--accent); height: 100%; }
.kpi .label { color:#475569; font-size:.88rem; font-weight:600; }
.kpi .value { color:#0F172A; font-size:2rem; font-weight:700; line-height:1.25; margin: 2px 0; }
.kpi .sub { color:#475569; font-size:.85rem; }
.bar { background:#E2E8F0; border-radius:999px; height:8px; margin-top:10px; overflow:hidden; }
.bar > span { display:block; height:100%; background: var(--accent); }
.card { background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:14px 18px; margin-bottom:12px;
        box-shadow: 0 2px 8px rgba(15,23,42,.05); }
.card h4 { margin:0 0 6px 0; color:#0F172A; font-size:1rem; }
.card p { margin: 3px 0; color:#334155; font-size:.92rem; }
.tag { font-weight:700; }
.t-obs { color:#2563EB; } .t-int { color:#7C3AED; } .t-rec { color:#0D9488; }
@media (max-width: 640px) { .kpi .value { font-size:1.55rem; } .hdr h1 { font-size:1.4rem; } .hdr { padding:16px; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ data loading
@st.cache_data(show_spinner=False)
def load_from_path(path: str, mtime: float):
    return utils.load_dataset(path)


def get_data():
    """Return (df, report, dataset_name, updated_datetime) or stop with a clear message."""
    with st.sidebar.expander("Use a different CSV"):
        uploaded = st.file_uploader("Upload a customer churn CSV", type="csv", label_visibility="collapsed")
    if uploaded is not None:
        df, rep = utils.load_dataset(uploaded)
        name, updated = uploaded.name, datetime.now()
    else:
        path = utils.find_csv(BASE_DIR)
        if path is None:
            st.error(f"No CSV file was found. Put `{utils.DEFAULT_CSV}` next to app.py or upload a file in the sidebar.")
            st.stop()
        df, rep = load_from_path(str(path), path.stat().st_mtime)
        name, updated = path.name, datetime.fromtimestamp(path.stat().st_mtime)
    if df is None:
        st.error("The dataset did not pass validation, so no dashboard can be shown.")
        for e in rep.errors:
            st.markdown(f"- {e}")
        st.stop()
    return df, rep, name, updated


# ------------------------------------------------------------------ small UI helpers
def kpi_card(label, value, sub, accent, fill=None):
    bar = f'<div class="bar"><span style="width:{max(0, min(fill, 100)):.0f}%"></span></div>' if fill is not None and pd.notna(fill) else ""
    st.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div class="sub">{sub}</div>{bar}</div>', unsafe_allow_html=True)


def insight_card(title, lines):
    body = "".join(f'<p><span class="tag {cls}">{tag}:</span> {text}</p>' for tag, cls, text in lines)
    st.markdown(f'<div class="card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


def chart(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def sample_notice(n_total):
    st.warning(f"**Small sample notice:** this dashboard is based on only {n_total} customers. "
               "Percentages can swing sharply with one customer, so please do not generalise these results to a larger customer population.")


def kpi_row(k):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Churn Rate", utils.pct(k["churn_rate"]), f'{k["churned"]} of {k["total"]} customers churned', "#EA580C", k["churn_rate"])
    with c2:
        kpi_card("Retention Rate", utils.pct(k["retention_rate"]), f'{k["retained"]} of {k["total"]} customers retained', "#0D9488", k["retention_rate"])
    with c3:
        kpi_card("Monthly Revenue at Risk", utils.fmt_inr(k["revenue_at_risk"]),
                 f'Monthly recurring revenue from {k["churned"]} churned customer(s) in this sample', "#7C3AED")
    with c4:
        ch, re_ = k["avg_tenure_churned"], k["avg_tenure_retained"]
        cmp_ = f'Churned {ch:.1f} vs retained {re_:.1f} months' if pd.notna(ch) and pd.notna(re_) else "Comparison needs both groups"
        kpi_card("Average Tenure", f'{k["avg_tenure"]:.1f} months', cmp_, "#2563EB")


# ------------------------------------------------------------------ main
df_all, report, dataset_name, updated = get_data()
n_all = len(df_all)
t_min, t_max = int(df_all["TenureMonths"].min()), int(df_all["TenureMonths"].max())
contract_opts = ["All"] + [c for c in utils.VALID_CONTRACTS if c in set(df_all["ContractType"])]
payment_opts = ["All"] + sorted(df_all["PaymentMethod"].unique())

if LOGO.exists():
    st.sidebar.image(str(LOGO), width=72)
st.sidebar.markdown("### Churn Analytics")
page = st.sidebar.radio("Navigate", PAGES, key="page")
st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")


def reset_filters():
    st.session_state["f_contract"] = "All"
    st.session_state["f_payment"] = "All"
    st.session_state["f_tenure"] = (t_min, t_max)


st.session_state.setdefault("f_contract", "All")
st.session_state.setdefault("f_payment", "All")
st.session_state.setdefault("f_tenure", (t_min, t_max))
# a new CSV may have a different tenure range
lo, hi = st.session_state["f_tenure"]
if lo < t_min or hi > t_max:
    st.session_state["f_tenure"] = (t_min, t_max)

st.sidebar.selectbox("Contract type", contract_opts, key="f_contract")
if t_min < t_max:
    st.sidebar.slider("Tenure (months)", t_min, t_max, key="f_tenure")
else:
    st.sidebar.caption(f"Tenure is {t_min} months for every customer, so the slider is hidden.")
    st.session_state["f_tenure"] = (t_min, t_max)
st.sidebar.selectbox("Payment method", payment_opts, key="f_payment")
st.sidebar.button("Reset filters", on_click=reset_filters, use_container_width=True)

df = utils.apply_filters(df_all, st.session_state["f_contract"], st.session_state["f_tenure"], st.session_state["f_payment"])
k = utils.compute_kpis(df)
st.sidebar.caption(f"Showing {len(df)} of {n_all} customers")

# header
if page != "About Dataset":
    st.markdown(
        f'<div class="hdr"><h1>Customer Churn Analytics</h1><p>Customer Retention, Revenue Risk &amp; Business Insights</p>'
        f'<span class="chip">Dataset: {html.escape(dataset_name)}</span><span class="chip">Total customers: {n_all}</span>'
        f'<span class="chip">Last updated: {updated:%d %b %Y}</span><span class="chip">Small sample</span></div>',
        unsafe_allow_html=True)
    if df.empty:
        sample_notice(n_all)
        st.info("No customers match the current filters. Widen the tenure range or choose **All**, or press **Reset filters**.")
        st.stop()


# ------------------------------------------------------------------ pages
if page == "Overview":
    sample_notice(n_all)
    kpi_row(k)
    st.write("")
    a, b = st.columns(2)
    with a:
        chart(charts.churn_donut(df), "ov_donut")
    with b:
        chart(charts.segment_bar(df, "ContractType", "Churn by contract type", utils.VALID_CONTRACTS), "ov_contract")
    st.subheader("Business insights")
    st.caption("Calculated from the customers currently in view. These are observed patterns, not proof of cause.")
    for ins in utils.generate_insights(df):
        insight_card(ins["category"], [("Observed", "t-obs", ins["observed"]), ("Possible interpretation", "t-int", ins["interpretation"])])
    st.subheader("Recommendations")
    recs, watch = utils.generate_recommendations(df)
    for r in recs:
        insight_card(r["title"], [("Based on", "t-obs", r["basis"]), ("Recommendation", "t-rec", r["action"])])
    if not watch.empty:
        st.markdown("**Retention watchlist** (retained customers with two or more risk signals; a review list, not a prediction)")
        st.dataframe(watch, use_container_width=True, hide_index=True)

elif page == "Churn Analysis":
    sample_notice(n_all)
    kpi_row(k)
    st.write("")
    a, b = st.columns(2)
    with a:
        chart(charts.churn_donut(df), "ca_donut")
    with b:
        chart(charts.segment_bar(df, "ContractType", "Churn by contract type", utils.VALID_CONTRACTS), "ca_contract")
    st.subheader("Customer tenure analysis")
    a, b = st.columns([3, 2])
    with a:
        chart(charts.tenure_box(df), "ca_tenure")
    with b:
        ts = utils.tenure_stats(df)
        st.dataframe(ts.style.format({"Average (months)": "{:.1f}", "Minimum": "{:.0f}", "Maximum": "{:.0f}"}, na_rep="n/a"),
                     use_container_width=True, hide_index=True)
        st.caption("Tenure is the number of months the customer has been (or was) with the company.")
    st.subheader("Support ticket analysis")
    a, b = st.columns([3, 2])
    with a:
        chart(charts.tickets_chart(df), "ca_tickets")
    with b:
        c1, c2 = st.columns(2)
        c1.metric("Avg tickets, churned", f'{k["avg_tickets_churned"]:.1f}' if pd.notna(k["avg_tickets_churned"]) else "n/a")
        c2.metric("Avg tickets, retained", f'{k["avg_tickets_retained"]:.1f}' if pd.notna(k["avg_tickets_retained"]) else "n/a")
        st.caption("Customers who churned also raised more tickets in this sample. This is an association only; the data cannot show that tickets cause churn.")

elif page == "Revenue Analysis":
    sample_notice(n_all)
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Monthly Revenue at Risk", utils.fmt_inr(k["revenue_at_risk"]), f'From {k["churned"]} churned customer(s)', "#7C3AED")
    with c2:
        kpi_card("Total Monthly Charges", utils.fmt_inr(k["monthly_revenue"]), f'All {k["total"]} customers in view', "#2563EB")
    share = k["revenue_at_risk"] / k["monthly_revenue"] * 100 if k["monthly_revenue"] else float("nan")
    with c3:
        kpi_card("Share of Monthly Revenue at Risk", utils.pct(share), "Churned customers' monthly charges / all monthly charges", "#EA580C", share)
    st.caption("Revenue at risk is monthly recurring revenue (MonthlyCharges of churned customers). It is not TotalCharges or lifetime value.")
    a, b = st.columns(2)
    with a:
        chart(charts.monthly_charges_by_status(df), "rv_monthly")
    with b:
        chart(charts.total_charges_by_status(df), "rv_total")
    a, b = st.columns(2)
    with a:
        chart(charts.revenue_at_risk_by_contract(df), "rv_risk")
    with b:
        chart(charts.revenue_share_by_contract(df), "rv_share")

elif page == "Customer Segmentation":
    sample_notice(n_all)
    st.caption("Each segment shows churned and retained counts. Churn rate is churned customers divided by customers in that segment.")
    segs = [("SubscriptionType", "Subscription type"), ("ContractType", "Contract type"), ("PaymentMethod", "Payment method"),
            ("Gender", "Gender"), ("AgeGroup", "Age group")]
    tabs = st.tabs([s[1] for s in segs])
    for tab, (col, label) in zip(tabs, segs):
        with tab:
            order = utils.VALID_CONTRACTS if col == "ContractType" else (utils.AGE_LABELS if col == "AgeGroup" else None)
            a, b = st.columns([3, 2])
            with a:
                chart(charts.segment_bar(df, col, f"Churn by {label.lower()}", order), f"sg_{col}")
            with b:
                seg = utils.segment_table(df, col)
                st.dataframe(seg[[col, "Customers", "Churned", "Churn Rate (%)"]].style.format({"Churn Rate (%)": "{:.1f}"}),
                             use_container_width=True, hide_index=True)
                st.caption("Segments with no customers under the current filters are not shown.")

elif page == "Customer Data":
    sample_notice(n_all)
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Search", placeholder="Search by customer ID, plan, contract, payment method...")
    status = c2.selectbox("Churn status", ["All", "Yes", "No"])
    view = df
    if status != "All":
        view = view[view["Churn"] == status]
    if q.strip():
        text = view.drop(columns=["AgeGroup"]).astype(str).agg(" ".join, axis=1)
        view = view[text.str.contains(q.strip(), case=False, regex=False)]
    cols = ["CustomerID", "Age", "Gender", "TenureMonths", "SubscriptionType", "MonthlyCharges", "TotalCharges",
            "ContractType", "SupportTickets", "PaymentMethod", "Churn"]
    st.caption(f"{len(view)} customer(s) shown. Click a column header to sort.")
    if view.empty:
        st.info("No customers match the search. Clear the search box or the churn status filter.")
    else:
        st.dataframe(view[cols], use_container_width=True, hide_index=True, column_config={
            "MonthlyCharges": st.column_config.NumberColumn(format="\u20b9%.2f"),
            "TotalCharges": st.column_config.NumberColumn(format="\u20b9%.2f")})
    st.download_button("Download filtered data as CSV", view[cols].to_csv(index=False).encode("utf-8"),
                       file_name="filtered_customers.csv", mime="text/csv", disabled=view.empty)

else:  # About Dataset
    st.markdown("## About the dataset")
    sample_notice(n_all)
    a, b, c = st.columns(3)
    a.metric("Records", n_all)
    b.metric("Columns", len(utils.REQUIRED_COLUMNS))
    c.metric("Source file", dataset_name)
    st.markdown("### Columns")
    st.dataframe(pd.DataFrame({
        "Column": utils.REQUIRED_COLUMNS,
        "Meaning": ["Unique customer ID", "Customer gender", "Age in years", "Months with the company",
                    "Plan (Basic / Pro / Enterprise)", "Monthly fee (shown in INR)", "Total billed to date (INR)",
                    "Contract length", "Support tickets raised", "How the customer pays", "Whether the customer left (Yes/No)"],
        "Type": [str(df_all[c].dtype) for c in utils.REQUIRED_COLUMNS]}), use_container_width=True, hide_index=True)
    st.markdown("### KPI definitions")
    st.markdown("- **Churn rate** = churned customers / total customers x 100\n"
                "- **Retention rate** = retained customers / total customers x 100\n"
                "- **Monthly revenue at risk** = sum of MonthlyCharges for customers with Churn = Yes\n"
                "- **Average tenure** = mean of TenureMonths\n\n"
                "All KPIs are calculated from the customers that pass the current sidebar filters.")
    st.markdown("### Data quality checks")
    st.dataframe(pd.DataFrame(report.checks, columns=["Check", "Result", "Detail"]), use_container_width=True, hide_index=True)
    st.markdown("### Modifications made to the data")
    for m in report.modifications:
        st.markdown(f"- {m}")
    st.caption("The currency is not stored in the CSV; amounts are displayed in INR as specified for this project.")
    st.markdown("### Preview")
    st.dataframe(df_all.drop(columns=["AgeGroup"]).head(), use_container_width=True, hide_index=True)
