"""Plotly figure builders. Every function takes an already-filtered dataframe."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import segment_table

CHURN_COLOR = "#EA580C"     # orange  - churned
RETAINED_COLOR = "#0D9488"  # teal    - retained
BLUE, PURPLE, GREEN = "#2563EB", "#7C3AED", "#16A34A"
PALETTE = [BLUE, PURPLE, RETAINED_COLOR, GREEN, CHURN_COLOR]
STATUS_COLORS = {"Churned": CHURN_COLOR, "Retained": RETAINED_COLOR}
FONT = "Inter, Segoe UI, Roboto, Arial, sans-serif"


def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16, color="#0F172A")),
        font=dict(family=FONT, size=13, color="#334155"),
        paper_bgcolor="white", plot_bgcolor="white", height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    return fig


def _status(df: pd.DataFrame) -> pd.Series:
    return df["Churn"].map({"Yes": "Churned", "No": "Retained"})


def churn_donut(df: pd.DataFrame) -> go.Figure:
    counts = _status(df).value_counts().reindex(["Churned", "Retained"]).fillna(0).astype(int)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.6, sort=False,
        marker=dict(colors=[CHURN_COLOR, RETAINED_COLOR], line=dict(color="white", width=3)),
        textinfo="label+value+percent", textfont=dict(size=13),
        hovertemplate="%{label}<br>%{value} customers (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{int(counts.sum())}</b><br>customers", showarrow=False, font=dict(size=15))
    fig.update_layout(showlegend=False)
    return _style(fig, "Churned vs retained customers")


def segment_bar(df: pd.DataFrame, col: str, title: str, order: list[str] | None = None) -> go.Figure:
    """Grouped bars of churned vs retained per segment, churn rate shown in the axis label."""
    seg = segment_table(df, col)
    if order:
        seg[col] = pd.Categorical(seg[col], categories=[o for o in order if o in set(seg[col])], ordered=True)
        seg = seg.sort_values(col)
        seg[col] = seg[col].astype(str)
    labels = [f"{n}<br>churn {r:.0f}%" for n, r in zip(seg[col], seg["Churn Rate (%)"])]
    fig = go.Figure()
    for name, column in (("Churned", "Churned"), ("Retained", "Retained")):
        fig.add_bar(x=labels, y=seg[column], name=name, marker_color=STATUS_COLORS[name],
                    text=seg[column], textposition="outside",
                    customdata=seg[[col, "Customers", "Churn Rate (%)"]].values,
                    hovertemplate=f"<b>%{{customdata[0]}}</b><br>{name}: %{{y}}<br>"
                                  "Total customers: %{customdata[1]}<br>"
                                  "Churn rate: %{customdata[2]:.1f}%<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Customers", dtick=1 if seg["Customers"].max() <= 10 else None)
    return _style(fig, title)


def tenure_box(df: pd.DataFrame) -> go.Figure:
    d = df.assign(Status=_status(df))
    fig = px.box(d, x="Status", y="TenureMonths", color="Status", points="all",
                 color_discrete_map=STATUS_COLORS, category_orders={"Status": ["Churned", "Retained"]},
                 hover_data={"CustomerID": True, "Status": False})
    fig.update_traces(boxmean=True, jitter=0.35, pointpos=0)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Tenure (months)")
    return _style(fig, "Tenure by churn status (diamond = mean)")


def tickets_chart(df: pd.DataFrame) -> go.Figure:
    d = df.assign(Status=_status(df))
    counts = d.groupby(["SupportTickets", "Status"]).size().reset_index(name="Customers")
    fig = px.bar(counts, x="SupportTickets", y="Customers", color="Status", barmode="group",
                 color_discrete_map=STATUS_COLORS, text="Customers")
    fig.update_traces(textposition="outside",
                      hovertemplate="%{x} tickets<br>%{y} customers<extra>%{fullData.name}</extra>")
    fig.update_xaxes(title="Support tickets raised", dtick=1)
    fig.update_yaxes(title="Customers", dtick=1 if counts["Customers"].max() <= 10 else None)
    return _style(fig, "Support tickets by churn status")


def monthly_charges_by_status(df: pd.DataFrame) -> go.Figure:
    d = df.assign(Status=_status(df)).groupby("Status", as_index=False).agg(
        Average=("MonthlyCharges", "mean"), Customers=("CustomerID", "count"))
    fig = px.bar(d, x="Status", y="Average", color="Status", color_discrete_map=STATUS_COLORS,
                 text=d["Average"].map(lambda v: f"\u20b9{v:,.2f}"), custom_data=["Customers"])
    fig.update_traces(textposition="outside",
                      hovertemplate="%{x}<br>Average monthly charge: \u20b9%{y:,.2f}<br>Customers: %{customdata[0]}<extra></extra>")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Average monthly charge (\u20b9)", tickprefix="\u20b9")
    return _style(fig, "Average monthly charges by churn status")


def total_charges_by_status(df: pd.DataFrame) -> go.Figure:
    d = df.assign(Status=_status(df)).groupby("Status", as_index=False).agg(
        Total=("TotalCharges", "sum"), Average=("TotalCharges", "mean"), Customers=("CustomerID", "count"))
    fig = px.bar(d, x="Status", y="Total", color="Status", color_discrete_map=STATUS_COLORS,
                 text=d["Total"].map(lambda v: f"\u20b9{v:,.0f}"), custom_data=["Average", "Customers"])
    fig.update_traces(textposition="outside",
                      hovertemplate="%{x}<br>Total billed to date: \u20b9%{y:,.2f}<br>"
                                    "Average per customer: \u20b9%{customdata[0]:,.2f}<br>"
                                    "Customers: %{customdata[1]}<extra></extra>")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Total charges billed (\u20b9)", tickprefix="\u20b9")
    return _style(fig, "Total charges billed to date by churn status")


def revenue_at_risk_by_contract(df: pd.DataFrame) -> go.Figure:
    seg = segment_table(df, "ContractType")
    fig = px.bar(seg, x="ContractType", y="Revenue at Risk", text=seg["Revenue at Risk"].map(lambda v: f"\u20b9{v:,.2f}"),
                 custom_data=["Churned", "Customers"])
    fig.update_traces(marker_color=CHURN_COLOR, textposition="outside",
                      hovertemplate="%{x}<br>Monthly revenue at risk: \u20b9%{y:,.2f}<br>"
                                    "Churned customers: %{customdata[0]} of %{customdata[1]}<extra></extra>")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Monthly revenue at risk (\u20b9)", tickprefix="\u20b9")
    return _style(fig, "Monthly revenue at risk by contract type")


def revenue_share_by_contract(df: pd.DataFrame) -> go.Figure:
    seg = segment_table(df, "ContractType")
    fig = go.Figure(go.Pie(
        labels=seg["ContractType"], values=seg["Monthly Revenue"], hole=0.55, sort=False,
        marker=dict(colors=[BLUE, PURPLE, GREEN][: len(seg)], line=dict(color="white", width=3)),
        textinfo="label+percent",
        hovertemplate="%{label}<br>Monthly revenue: \u20b9%{value:,.2f} (%{percent})<extra></extra>"))
    fig.update_layout(showlegend=False)
    return _style(fig, "Monthly revenue contribution by contract type")
