import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input
import dash_bootstrap_components as dbc

# ======================================================================
# Configuration
# ======================================================================
DATA_PATH = os.environ.get(
    "NOSHOW_CSV",
    r"C:\Users\Lenovo\Downloads\archive\KaggleV2-May-2016.csv"
)
THEME = dbc.themes.FLATLY
TOP_N = 50

# ======================================================================
# Load & Preprocess (mirrors original steps)
# ======================================================================
df = pd.read_csv(DATA_PATH)

if "PatientId" in df.columns:
    df["PatientId"] = df["PatientId"].astype("int64")

for col in ["ScheduledDay", "AppointmentDay"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col]).dt.date.astype("datetime64[ns]")

for col in ["PatientId", "AppointmentID"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

rename_pairs = {
    "Scholarship": "Scholarship",
    "Hipertension": "Hypertension",
    "Diabetes": "Diabetes",
    "Alcoholism": "Alcoholism",
    "Handcap": "Handicap",
    "SMS_received": "SMSReceived",
}
for src, dst in rename_pairs.items():
    if src in df.columns:
        df[dst] = df[src].astype("object")

if "Age" in df.columns:
    df = df[df["Age"] >= 0]

if "ScheduledDay" in df.columns:
    df["ScheduledDay_DOW"] = df["ScheduledDay"].dt.day_name()
if "AppointmentDay" in df.columns:
    df["AppointmentDay_DOW"] = df["AppointmentDay"].dt.day_name()

# ======================================================================
# Plotly Theme
# ======================================================================
medical_template = pio.templates["plotly_white"].layout.template
medical_template.layout.paper_bgcolor = "#F8FBFF"
medical_template.layout.plot_bgcolor = "#F8FBFF"
medical_template.layout.font.color   = "#1E2A38"
medical_template.layout.colorway     = ["#2E86AB", "#28B463", "#CA6F1E",
                                        "#AF7AC5", "#F39C12", "#D98880"]
pio.templates["medical_theme"] = medical_template
pio.templates.default = "medical_theme"

# ======================================================================
# KPI / Insights helpers
# ======================================================================
# ---------- Pretty Conclusions (cards, badges) ----------
import dash_bootstrap_components as dbc

def _show_rate(series):
    s = series.dropna()
    return float((s == "No").mean()) if len(s) else float("nan")

def _group_show_rate(col):
    if col not in df.columns:
        return None
    return df.groupby(col)["No-show"].apply(_show_rate).reset_index(name="show_rate")

def _fmt_pct(x): 
    return f"{(100*x):.1f}%" if pd.notna(x) else "—"

def conclusions_layout_cards():
    # --- SMS block ---
    sms_col = "SMS_received" if "SMS_received" in df.columns else ("SMSReceived" if "SMSReceived" in df.columns else None)
    sms_card = html.Div()
    if sms_col:
        g = _group_show_rate(sms_col)
        if g is not None and len(g):
            label_map = {0: "No SMS (0)", 1: "SMS (1)", "0": "No SMS (0)", "1": "SMS (1)"}
            items = [
                dbc.ListGroupItem([
                    html.Span(label_map.get(r[sms_col], str(r[sms_col]))),
                    dbc.Badge(_fmt_pct(r["show_rate"]), color="primary", pill=True, className="ms-2")
                ]) for _, r in g.iterrows()
            ]
            sms_card = dbc.Card([
                dbc.CardHeader("SMS Reminder Comparison"),
                dbc.ListGroup(items, flush=True)
            ])

    # --- Age bands block ---
    age_card = html.Div()
    if "Age" in df.columns:
        bins   = [0,5,12,18,30,45,60,75,120]
        labels = ["0–5","6–12","13–18","19–30","31–45","46–60","61–75","76+"]
        tmp = df.copy()
        tmp["AgeBand"] = pd.cut(tmp["Age"].clip(0,120), bins=bins, labels=labels, right=True)
        ab = tmp.groupby("AgeBand")["No-show"].apply(_show_rate).reset_index(name="show_rate").dropna()
        if len(ab):
            best  = ab.iloc[ab["show_rate"].idxmax()]
            worst = ab.iloc[ab["show_rate"].idxmin()]
            age_card = dbc.Card([
                dbc.CardHeader("Age Bands"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Span("Highest Attendance"),
                        dbc.Badge(f"{best['AgeBand']} • {_fmt_pct(best['show_rate'])}", color="success", pill=True, className="ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Lowest Attendance"),
                        dbc.Badge(f"{worst['AgeBand']} • {_fmt_pct(worst['show_rate'])}", color="danger", pill=True, className="ms-2")
                    ])
                ], flush=True)
            ])

    # --- Clinical / socioeconomic conditions (each in its own card) ---
    cond_cards = []
    for cname in ["Hypertension","Diabetes","Alcoholism","Scholarship","Handicap"]:
        if cname in df.columns:
            g = _group_show_rate(cname)
            if g is not None and len(g):
                items = [
                    dbc.ListGroupItem([
                        html.Span(f"{cname} = {r[cname]}"),
                        dbc.Badge(_fmt_pct(r["show_rate"]), color="secondary", pill=True, className="ms-2")
                    ]) for _, r in g.iterrows()
                ]
                cond_cards.append(
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{cname} vs. Attendance"),
                                      dbc.ListGroup(items, flush=True)]),
                            md=6, className="mb-3")
                )

    return html.Div([
        html.H4("Conclusions & Key Insights", className="mb-3"),
        dbc.Row([
            dbc.Col(sms_card, md=6),
            dbc.Col(age_card, md=6),
        ], className="g-3"),
        html.Br(),
        dbc.Row(cond_cards, className="g-3"),
        html.Div("Note: Show-up rate = proportion of appointments where ‘No-show’ = ‘No’.",
                 className="text-muted mt-2")
    ])


def kpi_data():
    d = {
        "overall_rate": _show_rate(df["No-show"]),
        "rows": len(df),
        "best_day": None,
        "worst_day": None
    }
    if "AppointmentDay_DOW" in df.columns:
        g = _group_show_rate("AppointmentDay_DOW").sort_values("show_rate", ascending=False)
        if len(g):
            d["best_day"]  = (g.iloc[0]["AppointmentDay_DOW"], float(g.iloc[0]["show_rate"]))
            d["worst_day"] = (g.iloc[-1]["AppointmentDay_DOW"], float(g.iloc[-1]["show_rate"]))
    return d

def fmt_pct(x): return f"{(100.0*x):.1f}%"

def kpi_header_row():
    d = kpi_data()
    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("Overall Show-Up Rate", className="text-muted"),
            html.H3(fmt_pct(d["overall_rate"]))
        ])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("Total Appointments", className="text-muted"),
            html.H3(f"{d['rows']:,}")
        ])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("Best Attendance Day", className="text-muted"),
            html.H4(d["best_day"][0] if d["best_day"] else "—"),
            dbc.Badge(fmt_pct(d["best_day"][1]), color="success", pill=True, className="mt-1")
            if d["best_day"] else html.Div()
        ])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("Lowest Attendance Day", className="text-muted"),
            html.H4(d["worst_day"][0] if d["worst_day"] else "—"),
            dbc.Badge(fmt_pct(d["worst_day"][1]), color="danger", pill=True, className="mt-1")
            if d["worst_day"] else html.Div()
        ])), md=3),
    ], className="g-3")

# Narrative conclusions (used only inside Conclusions tab)
def insights_markdown() -> str:
    bullets = []
    overall = _show_rate(df["No-show"])
    bullets.append(f"- **Overall show-up rate:** {overall:.1%} across **{len(df):,}** appointments.")
    if "AppointmentDay_DOW" in df.columns:
        dow = _group_show_rate("AppointmentDay_DOW").sort_values("show_rate", ascending=False)
        if len(dow):
            best, worst = dow.iloc[0], dow.iloc[-1]
            bullets.append(f"- **Best day:** {best['AppointmentDay_DOW']} ({best['show_rate']:.1%}); "
                           f"**Lowest:** {worst['AppointmentDay_DOW']} ({worst['show_rate']:.1%}).")
    sms_col = "SMS_received" if "SMS_received" in df.columns else ("SMSReceived" if "SMSReceived" in df.columns else None)
    if sms_col:
        sms = _group_show_rate(sms_col)
        if sms is not None and len(sms) >= 2:
            label_map = {0: "No SMS (0)", 1: "SMS (1)", "0": "No SMS (0)", "1": "SMS (1)"}
            sms["label"] = sms[sms_col].map(label_map).fillna(sms[sms_col].astype(str))
            bullets.append("- **SMS reminder comparison:** " + " • ".join(
                [f"{r['label']}: {r['show_rate']:.1%}" for _, r in sms.iterrows()]
            ))
    for c in ["Hypertension", "Diabetes", "Alcoholism", "Scholarship", "Handicap"]:
        if c in df.columns:
            g = _group_show_rate(c).sort_values(c)
            line = " • ".join([f"{c}={str(k)}: {v:.1%}" for k, v in zip(g[c], g["show_rate"])])
            bullets.append(f"- **{c} vs. attendance:** {line}")
    if "Age" in df.columns:
        bins = [0,5,12,18,30,45,60,75,120]
        labels = ["0–5","6–12","13–18","19–30","31–45","46–60","61–75","76+"]
        tmp = df.copy()
        tmp["AgeBand"] = pd.cut(tmp["Age"].clip(lower=0, upper=120), bins=bins, labels=labels, right=True)
        ab = tmp.groupby("AgeBand")["No-show"].apply(_show_rate).reset_index(name="show_rate").dropna()
        if len(ab):
            best = ab.iloc[ab["show_rate"].idxmax()]
            worst = ab.iloc[ab["show_rate"].idxmin()]
            bullets.append(f"- **Age bands:** Highest in **{best['AgeBand']}** ({best['show_rate']:.1%}); "
                           f"lowest in **{worst['AgeBand']}** ({worst['show_rate']:.1%}).")
    return "### Conclusions & Key Insights\n\n" + "\n".join(bullets) + \
           "\n\n> *Show-up rate = proportion of appointments where `No-show = 'No'`.*"

# ======================================================================
# Figure builders (lazy)
# ======================================================================
def fig_show_vs_noshow():
    return px.histogram(df, x="No-show", nbins=10, title="Show vs. No-Show (Overall)")

def fig_gender_vs_noshow():
    return px.bar(df, x="No-show", color="Gender", barmode="group",
                  title="Show vs. No-Show by Gender")

def fig_age_box():
    return px.box(df, y="Age", title="Age Distribution (Boxplot)")

def fig_age_dist():
    return px.histogram(df, x="Age", nbins=15, title="Age Distribution (Histogram)")

def fig_age_noshow():
    return px.histogram(df, x="Age", color="No-show", barmode="group",
                        title="Show vs. No-Show by Age")

def figs_neighbourhoods():
    neigh_counts = (
        df.groupby("Neighbourhood").size().reset_index(name="count")
          .sort_values("count", ascending=False).head(TOP_N)
    )
    f1 = px.bar(neigh_counts, x="Neighbourhood", y="count",
                title=f"Appointments by Neighborhood (Top {TOP_N})")
    f1.update_layout(xaxis=dict(categoryorder="total descending"),
                     height=500, transition_duration=0,
                     uniformtext_minsize=8, uniformtext_mode="hide")
    f1.update_xaxes(tickangle=-45)

    neigh_ns = (
        df[df["Neighbourhood"].isin(neigh_counts["Neighbourhood"])]
          .groupby(["Neighbourhood", "No-show"]).size().reset_index(name="count")
    )
    f2 = px.bar(neigh_ns, x="Neighbourhood", y="count", color="No-show", barmode="group",
                title=f"Show vs. No-Show by Neighborhood (Top {TOP_N})")
    f2.update_layout(xaxis=dict(categoryorder="array", categoryarray=neigh_counts["Neighbourhood"]),
                     height=500, transition_duration=0,
                     uniformtext_minsize=8, uniformtext_mode="hide")
    f2.update_xaxes(tickangle=-45)
    return f1, f2

def ratio_pair(category_col: str, count_plot_kind: str = "histogram"):
    ratio = (
        df[df["No-show"] == "No"].groupby([category_col]).size()
        / df.groupby([category_col]).size()
    ).reset_index()
    ratio.columns = [category_col, "Show_Up_Ratio"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Show vs. No-Show by {category_col}",
                        f"Show-Up Ratio by {category_col}"]
    )
    if count_plot_kind == "histogram":
        f1 = px.histogram(df, x=category_col, color="No-show", barmode="group")
    else:
        f1 = px.bar(df, x="No-show", color=category_col, barmode="group")
    for tr in f1.data:
        fig.add_trace(tr, row=1, col=1)

    f2 = px.bar(ratio, x=category_col, y="Show_Up_Ratio")
    fig.add_trace(f2.data[0], row=1, col=2)

    fig.update_layout(height=500, width=1000,
                      title_text=f"Attendance Analysis by {category_col}",
                      showlegend=True)
    return fig

def figs_days():
    f1 = (px.bar(df, x="ScheduledDay_DOW", color="ScheduledDay_DOW",
                 title="Scheduled Appointments by Day of Week")
          if "ScheduledDay_DOW" in df.columns else go.Figure())
    f2 = (px.bar(df, x="AppointmentDay_DOW", color="AppointmentDay_DOW",
                 title="Attended Appointments by Day of Week")
          if "AppointmentDay_DOW" in df.columns else go.Figure())
    return f1, f2

# Candidate columns for interactive Conditions tab
COND_CANDIDATES = [
    c for c in ["Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap",
                "SMS_received", "SMSReceived", "Gender", "AppointmentDay_DOW", "ScheduledDay_DOW"]
    if c in df.columns
]
COUNT_KIND_MAP = {"Scholarship": "bar", "Diabetes": "bar", "Alcoholism": "bar"}

# ======================================================================
# Layout
# ======================================================================
app = Dash(__name__, external_stylesheets=[THEME])

app.layout = dbc.Container([
    html.H2("Medical Appointment Attendance Dashboard", className="mt-2"),
    html.P("Exploratory analysis of show/no-show patterns with interactive views.", className="text-muted"),

    # --- KPI header row (replaces dataset/source cards) ---
    kpi_header_row(),
    html.Hr(),

    dcc.Tabs(
        id="tabs", value="overview",
        children=[
            dcc.Tab(label="Overview", value="overview"),
            dcc.Tab(label="Age", value="age"),
            dcc.Tab(label="Neighborhoods", value="neigh"),
            dcc.Tab(label="Conditions", value="cond"),
            dcc.Tab(label="Days", value="days"),
            dcc.Tab(label="Conclusions", value="conclusion"),
        ]
    ),
    html.Div(id="tab-content", className="mt-3"),
], fluid=True)

# ======================================================================
# Tab renderer (lazy)
# ======================================================================
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "overview":
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_show_vs_noshow()), md=6),
            dbc.Col(dcc.Graph(figure=fig_gender_vs_noshow()), md=6),
            dbc.Col(dcc.Graph(figure=fig_age_box()), md=6),
            dbc.Col(dcc.Graph(figure=fig_age_dist()), md=6),
        ], className="g-3")

    if tab == "age":
        return dbc.Row([dbc.Col(dcc.Graph(figure=fig_age_noshow()), md=12)], className="g-3")

    if tab == "neigh":
        f1, f2 = figs_neighbourhoods()
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=f1), md=12),
            dbc.Col(dcc.Graph(figure=f2), md=12),
        ], className="g-3")

    if tab == "cond":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Compare No-show against:", className="fw-bold"),
                    dcc.Dropdown(
                        id="cond-col",
                        options=[{"label": c, "value": c} for c in COND_CANDIDATES],
                        value=COND_CANDIDATES[0] if COND_CANDIDATES else None,
                        clearable=False
                    )
                ], md=4),
            ], className="g-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="cond-fig"), md=12)], className="g-3")
        ])

    if tab == "days":
        f1, f2 = figs_days()
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=f1), md=6),
            dbc.Col(dcc.Graph(figure=f2), md=6),
        ], className="g-3")

    if tab == "conclusion":
        return conclusions_layout_cards()

    return html.Div("Select a tab.")

# ======================================================================
# Callback: interactive Conditions chart
# ======================================================================
@app.callback(
    Output("cond-fig", "figure"),
    Input("cond-col", "value"),
    prevent_initial_call=False
)
def update_condition_figure(col):
    if not col or col not in df.columns:
        return go.Figure()
    kind = COUNT_KIND_MAP.get(col, "histogram")
    return ratio_pair(col, count_plot_kind=kind)

# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
