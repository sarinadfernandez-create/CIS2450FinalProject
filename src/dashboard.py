#!/usr/bin/env python3
"""
Tech Trend Intelligence Dashboard
===================================
Run from the project root:
    python src/dashboard.py

Install extras (on top of requirements.txt):
    pip install dash dash-bootstrap-components plotly
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output

warnings.filterwarnings("ignore")

# ─── PATHS ─────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_SEARCH = [
    os.path.join(PROJECT_ROOT, "data", "processed"),
    os.path.join(SCRIPT_DIR,   "data", "processed"),
    os.path.join(PROJECT_ROOT, "src", "data", "processed"),
]


def find_file(name):
    for d in PROCESSED_SEARCH:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


# ─── DESIGN TOKENS ─────────────────────────────────────────────────────────

P = {
    "bg":       "#080d1a",
    "surface":  "#0f1729",
    "card":     "#111827",
    "border":   "#1c2d4e",
    "primary":  "#00e5c3",
    "violet":   "#7c6af7",
    "amber":    "#ffb547",
    "rose":     "#ff5272",
    "text":     "#d6e8ff",
    "muted":    "#4a6d9c",
    "grid":     "#192038",
}

SRC_COLORS = {
    "arxiv":            "#7c6af7",
    "semantic_scholar": "#a78bfa",
    "reddit":           "#ff5272",
    "hackernews":       "#ffb547",
    "github":           "#00e5c3",
    "newsletter":       "#38bdf8",
}

ALL_TOPICS = [
    "large language model",
    "autonomous AI agent framework",
    "AI coding assistant tool",
    "retrieval augmented generation",
    "open source model weights release",
    "CRISPR gene editing therapy",
    "NVIDIA GPU datacenter",
    "GLP-1 obesity weight loss drug",
    "quantum computing qubit",
    "diffusion model image generation",
    "cybersecurity ransomware attack",
    "protein structure AlphaFold prediction",
    "algorithmic trading strategy",
    "robotics humanoid robot",
    "semiconductor chip manufacturing",
    "reinforcement learning from human feedback",
    "drug discovery clinical trial",
    "AI safety alignment research",
    "decentralized finance DeFi protocol",
    "AI regulation government policy",
    "fine tuning language model",
    "vector database embeddings",
    "AI energy consumption datacenter",
    "synthetic data generation training",
    "multimodal AI system",
]

CHART_COLORS = [
    P["primary"], P["violet"], P["amber"], P["rose"],
    "#38bdf8", "#a78bfa", "#34d399", "#fb923c", "#f472b6", "#818cf8",
]

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color=P["text"], size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hoverlabel=dict(bgcolor=P["surface"], bordercolor=P["border"], font_size=12),
    margin=dict(l=0, r=0, t=15, b=0),
)

AXIS_X = dict(gridcolor=P["grid"], linecolor=P["border"], tickcolor=P["muted"])
AXIS_Y = dict(gridcolor=P["grid"], linecolor=P["border"], tickcolor=P["muted"])


# ─── DATA LAYER ─────────────────────────────────────────────────────────────

def is_lfs_pointer(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return "git-lfs" in f.readline()
    except Exception:
        return True


def safe_load(path):
    if not path or not os.path.exists(path):
        return None
    if is_lfs_pointer(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def generate_synthetic(n_weeks=80):
    """Realistic synthetic weekly signal data."""
    rng = np.random.default_rng(42)
    end = datetime.now()
    weeks = [end - timedelta(weeks=i) for i in range(n_weeks - 1, -1, -1)]
    sources = ["arxiv", "semantic_scholar", "reddit", "hackernews", "github"]

    base  = {t: rng.uniform(0.1, 0.75)  for t in ALL_TOPICS}
    slope = {t: rng.uniform(-0.004, 0.01) for t in ALL_TOPICS}

    HOT = {
        "large language model":           (0.94, 0.018),
        "autonomous AI agent framework":  (0.90, 0.016),
        "AI coding assistant tool":       (0.85, 0.014),
        "retrieval augmented generation": (0.82, 0.013),
        "open source model weights release": (0.79, 0.012),
        "NVIDIA GPU datacenter":          (0.77, 0.011),
        "GLP-1 obesity weight loss drug": (0.73, 0.010),
    }
    for t, (b, s) in HOT.items():
        base[t], slope[t] = b, s

    SRC_WEIGHTS = {
        "arxiv":            lambda t: 0.85 if any(k in t for k in ["model","diffusion","learning","protein","drug","CRISPR","quantum"]) else 0.30,
        "semantic_scholar": lambda t: 0.88 if any(k in t for k in ["protein","drug","CRISPR","quantum","genomics"]) else 0.28,
        "reddit":           lambda t: 0.88 if any(k in t for k in ["AI","agent","coding","trading","crypto","LLM"]) else 0.45,
        "hackernews":       lambda t: 0.80,
        "github":           lambda t: 0.85 if any(k in t for k in ["model","framework","tool","code","open"]) else 0.30,
    }

    rows = []
    for wi, week in enumerate(weeks):
        for topic in ALL_TOPICS:
            pop = float(np.clip(base[topic] + slope[topic] * wi + rng.normal(0, 0.025), 0.02, 1.0))
            for src in sources:
                w = SRC_WEIGHTS[src](topic)
                n = max(0, int(pop * w * rng.lognormal(4.6, 0.55)))
                if n:
                    rows.append({
                        "week": week, "canonical_topic": topic,
                        "source": src, "mentions": n,
                        "avg_sim": float(rng.uniform(0.62, 0.95)),
                    })
    return pd.DataFrame(rows)


def build_from_signal_map(df):
    df = df.copy()
    df["week"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("W").dt.start_time
    df = df.dropna(subset=["week"])
    return (
        df.groupby(["week","canonical_topic","source"])
        .agg(mentions=("phrase","count"), avg_sim=("similarity","mean"))
        .reset_index()
    )


def compute_scores(weekly):
    ACAD = {"arxiv","semantic_scholar"}
    SOC  = {"reddit","hackernews","github","newsletter"}

    acad  = weekly[weekly["source"].isin(ACAD)].groupby(["week","canonical_topic"])["mentions"].sum().rename("academic")
    soc   = weekly[weekly["source"].isin(SOC)].groupby(["week","canonical_topic"])["mentions"].sum().rename("social")
    total = weekly.groupby(["week","canonical_topic"])["mentions"].sum().rename("total")

    s = pd.concat([acad, soc, total], axis=1).fillna(0).reset_index()

    for col in ["academic","social","total"]:
        mx = s[col].max()
        s[f"{col}_n"] = (s[col] / mx) if mx > 0 else 0.0

    s["trend_score"] = 0.38 * s["academic_n"] + 0.62 * s["social_n"]
    s = s.sort_values(["canonical_topic","week"])
    s["velocity"] = (
        s.groupby("canonical_topic")["trend_score"]
         .pct_change().fillna(0).clip(-2, 5)
    )
    s["score"] = (0.68 * s["trend_score"] + 0.32 * s["velocity"].clip(0)).clip(0, 1)
    return s


# ─── LOAD DATA ──────────────────────────────────────────────────────────────

raw = safe_load(find_file("signal_topic_map.csv"))
if raw is not None and len(raw) > 500:
    print(f"✓ Real data loaded ({len(raw):,} rows)")
    WEEKLY = build_from_signal_map(raw)
    DEMO   = False
else:
    print("ℹ  Demo mode — synthetic data")
    WEEKLY = generate_synthetic()
    DEMO   = True

SCORES = compute_scores(WEEKLY)

LATEST = SCORES["week"].max()
PREV   = LATEST - timedelta(weeks=1)

THIS_WEEK = (
    SCORES[SCORES["week"] == LATEST]
    .sort_values("score", ascending=False).reset_index(drop=True)
)
PREV_MAP = (
    SCORES[SCORES["week"] == PREV]
    .set_index("canonical_topic")["score"].to_dict()
)

TOP10 = THIS_WEEK.head(10).copy()
TOP10["prev"]      = TOP10["canonical_topic"].map(PREV_MAP).fillna(0)
TOP10["delta"]     = TOP10["score"] - TOP10["prev"]
TOP10["delta_pct"] = (TOP10["delta"] / TOP10["prev"].clip(0.001) * 100).round(1)


# ─── CHARTS ─────────────────────────────────────────────────────────────────

def chart_top10():
    df = TOP10.copy().reset_index(drop=True)
    n  = len(df)

    # gradient: top topics → cyan, bottom → violet
    def lerp_color(t):
        r = int(0   + 124 * t)
        g = int(229 - 101 * t)
        b = int(195 -  67 * t)
        return f"rgba({r},{g},{b},0.82)"

    colors = [lerp_color(i / (n - 1)) for i in range(n)]

    fig = go.Figure(go.Bar(
        y=df["canonical_topic"].str.title()[::-1],
        x=df["score"][::-1],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        customdata=np.stack([
            df["delta_pct"][::-1],
            df["academic"][::-1].fillna(0),
            df["social"][::-1].fillna(0),
        ], axis=1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Score: %{x:.4f}<br>"
            "Δ vs last week: %{customdata[0]:+.1f}%<br>"
            "Academic: %{customdata[1]:,.0f}  ·  Social: %{customdata[2]:,.0f}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(**BASE_LAYOUT)
    fig.update_layout(
        height=390,
        xaxis=dict(**AXIS_X, title="Composite Trend Score", range=[0, 1.05]),
        yaxis=dict(**AXIS_Y, tickfont=dict(size=10.5)),
        bargap=0.3,
    )
    return fig


def chart_source_donut():
    src_tot = (
        WEEKLY[WEEKLY["week"] == LATEST]
        .groupby("source")["mentions"].sum()
        .reset_index().sort_values("mentions", ascending=False)
    )
    total = int(src_tot["mentions"].sum())
    fig = go.Figure(go.Pie(
        labels=src_tot["source"].str.replace("_"," ").str.title(),
        values=src_tot["mentions"],
        hole=0.62,
        marker=dict(
            colors=[SRC_COLORS.get(s, "#888") for s in src_tot["source"]],
            line=dict(color=P["bg"], width=3),
        ),
        textfont=dict(size=10.5),
        hovertemplate="%{label}<br>%{value:,} mentions · %{percent}<extra></extra>",
    ))
    fig.update_layout(**BASE_LAYOUT)
    fig.update_layout(
        height=310,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=10)),
        annotations=[dict(
            text=f"<b>{total:,}</b><br><span style='font-size:10px'>mentions</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=P["text"], size=13),
        )],
    )
    return fig


def chart_trends(topics=None):
    if not topics:
        topics = TOP10["canonical_topic"].tolist()[:6]
    df = SCORES[SCORES["canonical_topic"].isin(topics)].copy()

    fig = go.Figure()
    for i, topic in enumerate(topics):
        t = df[df["canonical_topic"] == topic].sort_values("week")
        if t.empty:
            continue
        c = CHART_COLORS[i % len(CHART_COLORS)]
        # Hex → rgba for fill
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        fill_c = f"rgba({r},{g},{b},0.07)"
        fig.add_trace(go.Scatter(
            x=t["week"], y=t["score"],
            mode="lines", name=topic.title(),
            line=dict(color=c, width=2.2, shape="spline"),
            fill="tozeroy", fillcolor=fill_c,
            hovertemplate="%{y:.4f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(**BASE_LAYOUT)
    fig.update_layout(
        height=330,
        hovermode="x unified",
        xaxis=dict(**AXIS_X, title=""),
        yaxis=dict(**AXIS_Y, title="Trend Score", range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.14, font=dict(size=9.5)),
    )
    return fig


def chart_scatter():
    df = THIS_WEEK.head(20).copy()
    df["academic_n"] = df["academic_n"].fillna(0)
    df["social_n"]   = df["social_n"].fillna(0)
    short = df["canonical_topic"].apply(lambda x: " ".join(x.split()[-2:]).title())

    fig = go.Figure(go.Scatter(
        x=df["academic_n"], y=df["social_n"],
        mode="markers+text",
        text=short,
        textposition="top center",
        textfont=dict(size=8.5, color=P["muted"]),
        marker=dict(
            size=df["score"] * 38 + 7,
            color=df["score"],
            colorscale=[[0, P["violet"]], [0.5, P["primary"]], [1, "#ffffff"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Score", font=dict(size=10)),
                thickness=10, len=0.6,
                tickfont=dict(size=9),
            ),
            line=dict(width=1, color=P["border"]),
            opacity=0.85,
        ),
        customdata=df[["canonical_topic","score"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Academic: %{x:.3f}  ·  Social: %{y:.3f}<br>"
            "Combined: %{customdata[1]:.4f}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0.5, line=dict(dash="dot", color=P["border"], width=1))
    fig.add_vline(x=0.5, line=dict(dash="dot", color=P["border"], width=1))
    for (ax, ay, txt) in [
        (0.77, 0.87, "📡 Everywhere"),
        (0.08, 0.87, "🌐 Social Hype"),
        (0.77, 0.08, "🔬 Academic"),
        (0.08, 0.08, "🌑 Emerging"),
    ]:
        fig.add_annotation(x=ax, y=ay, text=txt, showarrow=False,
                           font=dict(size=8.5, color=P["muted"]))
    fig.update_layout(**BASE_LAYOUT)
    fig.update_layout(
        height=370,
        xaxis=dict(**AXIS_X, title="Academic Signal", range=[-0.04, 1.1]),
        yaxis=dict(**AXIS_Y, title="Social Signal",   range=[-0.04, 1.1]),
    )
    return fig


def chart_momentum():
    df = TOP10.copy().sort_values("delta")
    colors = [P["primary"] if v >= 0 else P["rose"] for v in df["delta"]]
    fig = go.Figure(go.Bar(
        y=df["canonical_topic"].str.title(),
        x=df["delta"],
        orientation="h",
        marker=dict(color=colors, opacity=0.82, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Δ score: %{x:+.5f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=P["border"], width=1.5))
    fig.update_layout(**BASE_LAYOUT)
    fig.update_layout(
        height=340,
        xaxis=dict(**AXIS_X, title="Score Δ vs Previous Week"),
        yaxis=dict(**AXIS_Y, tickfont=dict(size=10)),
        margin=dict(l=0, r=15, t=15, b=0),
    )
    return fig


def hex_to_rgba(h, a=0.7):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def chart_source_timeline():
    src_weekly = (
        WEEKLY.groupby(["week","source"])["mentions"].sum().reset_index()
    )
    fig = go.Figure()
    for src in ["arxiv","semantic_scholar","reddit","hackernews","github"]:
        d = src_weekly[src_weekly["source"] == src].sort_values("week")
        if d.empty:
            continue
        c = SRC_COLORS.get(src, "#888888")
        fig.add_trace(go.Scatter(
            x=d["week"], y=d["mentions"],
            mode="lines", name=src.replace("_"," ").title(),
            stackgroup="one",
            line=dict(width=0),
            fillcolor=hex_to_rgba(c, 0.72),
            hovertemplate="%{y:,.0f} mentions<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(**BASE_LAYOUT)
    fig.update_layout(
        height=290,
        hovermode="x unified",
        xaxis=dict(**AXIS_X, title=""),
        yaxis=dict(**AXIS_Y, title="Mentions"),
        legend=dict(orientation="h", y=-0.16, font=dict(size=9.5)),
    )
    return fig


# ─── CSS ────────────────────────────────────────────────────────────────────

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html {{ scroll-behavior: smooth; }}

body, #_dash-app-content {{
    background: {P['bg']} !important;
    color: {P['text']};
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    line-height: 1.6;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {P['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {P['border']}; border-radius: 3px; }}

/* ── Navbar ── */
#navbar {{
    background: linear-gradient(135deg, {P['bg']} 0%, {P['surface']} 100%);
    border-bottom: 1px solid {P['border']};
    padding: 1rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
}}

.nav-logo {{
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 1.35rem; letter-spacing: -0.5px;
    background: linear-gradient(100deg, {P['primary']}, {P['violet']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}

.nav-meta {{
    font-size: 0.72rem; color: {P['muted']};
    font-family: 'IBM Plex Mono', monospace; margin-top: 3px;
}}

.badge-demo {{
    background: rgba(255,180,70,0.12); color: {P['amber']};
    border: 1px solid rgba(255,180,70,0.3); border-radius: 20px;
    padding: 3px 12px; font-size: 0.70rem;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.4px;
}}
.badge-live {{
    background: rgba(0,229,195,0.10); color: {P['primary']};
    border: 1px solid rgba(0,229,195,0.28); border-radius: 20px;
    padding: 3px 12px; font-size: 0.70rem;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.4px;
}}

/* ── Content ── */
#content {{
    max-width: 1440px; margin: 0 auto; padding: 2rem 2.5rem 4rem;
}}

/* ── KPI ── */
#kpi-row {{
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 1rem; margin-bottom: 1.5rem;
}}

.kpi {{
    background: {P['card']}; border: 1px solid {P['border']};
    border-radius: 12px; padding: 1.1rem 1.4rem;
    display: flex; align-items: center; gap: 0.9rem;
    transition: border-color .2s, box-shadow .2s;
    cursor: default;
}}
.kpi:hover {{
    border-color: rgba(0,229,195,0.3);
    box-shadow: 0 4px 24px rgba(0,229,195,0.06);
}}

.kpi-icon {{ font-size: 1.6rem; flex-shrink: 0; }}

.kpi-label {{
    font-size: 0.68rem; font-weight: 500; color: {P['muted']};
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 3px;
}}
.kpi-val {{
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem; font-weight: 700; line-height: 1; margin-bottom: 3px;
}}
.kpi-sub {{ font-size: 0.68rem; color: {P['muted']}; }}

/* ── Grid ── */
.row-2l {{
    display: grid; grid-template-columns: 1.75fr 1fr;
    gap: 1.2rem; margin-bottom: 1.2rem;
}}
.row-2 {{
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 1.2rem; margin-bottom: 1.2rem;
}}
.row-full {{ margin-bottom: 1.2rem; }}

/* ── Section card ── */
.card {{
    background: {P['card']}; border: 1px solid {P['border']};
    border-radius: 12px; padding: 1.2rem 1.4rem; height: 100%;
}}

.card-title {{
    font-family: 'Syne', sans-serif; font-size: 0.78rem; font-weight: 600;
    color: {P['muted']}; text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 0.9rem; padding-bottom: 0.6rem;
    border-bottom: 1px solid {P['border']};
    display: flex; align-items: center; gap: 0.5rem;
}}

.card-title::before {{
    content: ''; display: block; width: 3px; height: 14px;
    background: linear-gradient({P['primary']}, {P['violet']});
    border-radius: 2px;
}}

/* ── Dropdown ── */
.Select-control {{
    background: {P['surface']} !important;
    border-color: {P['border']} !important;
}}
.Select-menu-outer {{ background: {P['surface']} !important; }}
.Select-value-label {{ color: {P['text']} !important; }}
.VirtualizedSelectFocusedOption {{ background: {P['border']} !important; }}

/* ── Findings pills ── */
#pills-grid {{
    display: grid; grid-template-columns: repeat(2,1fr); gap: 0.55rem;
}}

.pill {{
    background: {P['surface']}; border: 1px solid {P['border']};
    border-radius: 9px; padding: 0.6rem 0.9rem;
    display: flex; align-items: center; gap: 0.7rem;
    transition: border-color .15s, background .15s;
    cursor: default;
}}
.pill:hover {{
    border-color: rgba(0,229,195,0.25);
    background: rgba(0,229,195,0.03);
}}

.pill-rank {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    color: {P['muted']}; min-width: 20px; flex-shrink: 0;
}}
.pill-name {{ font-size: 0.78rem; flex: 1; }}
.pill-right {{ display: flex; flex-direction: column; align-items: flex-end; gap: 1px; }}
.pill-score {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; font-weight: 500; color: {P['text']};
}}
.pill-delta {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.63rem; }}

/* ── Responsive ── */
@media (max-width: 1024px) {{
    #kpi-row {{ grid-template-columns: repeat(2,1fr); }}
    .row-2l, .row-2 {{ grid-template-columns: 1fr; }}
    #pills-grid {{ grid-template-columns: 1fr; }}
    #content {{ padding: 1.5rem 1.25rem 3rem; }}
}}
@media (max-width: 600px) {{
    #kpi-row {{ grid-template-columns: 1fr; }}
    .nav-logo {{ font-size: 1.1rem; }}
}}
"""


# ─── LAYOUT HELPERS ─────────────────────────────────────────────────────────

def card(title, *children, icon=""):
    return html.Div(className="card", children=[
        html.Div([icon + "  " + title if icon else title], className="card-title"),
        *children,
    ])


def kpi(icon, label, value, sub, color):
    return html.Div(className="kpi", children=[
        html.Div(icon, className="kpi-icon", style={"color": color}),
        html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-val", style={"color": color}),
            html.Div(sub,   className="kpi-sub"),
        ]),
    ])


def pill(rank, topic, score, delta):
    arrow = "↑" if delta >= 0 else "↓"
    col   = P["primary"] if delta >= 0 else P["rose"]
    return html.Div(className="pill", children=[
        html.Span(f"#{rank}", className="pill-rank"),
        html.Span(topic.title(), className="pill-name"),
        html.Div(className="pill-right", children=[
            html.Span(f"{score:.4f}", className="pill-score"),
            html.Span(f"{arrow} {abs(delta)*100:.1f}%", className="pill-delta",
                      style={"color": col}),
        ]),
    ])


# ─── APP ────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="Tech Trend Intelligence",
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
        "https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap",
    ],
)

# Write CSS to assets folder (Dash auto-loads from assets/)
_assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(_assets_dir, exist_ok=True)
with open(os.path.join(_assets_dir, "dashboard.css"), "w") as _f:
    _f.write(CSS)


def serve_layout():
    n_topics  = SCORES["canonical_topic"].nunique()
    top_name  = TOP10.iloc[0]["canonical_topic"].title()
    top_score = TOP10.iloc[0]["score"]
    week_tot  = int(WEEKLY[WEEKLY["week"] == LATEST]["mentions"].sum())
    n_rising  = int((TOP10["delta"] > 0).sum())

    badge = html.Span("⚙  DEMO MODE", className="badge-demo") if DEMO \
            else html.Span("● LIVE DATA",  className="badge-live")

    dropdown = dcc.Dropdown(
        id="topic-selector",
        options=[{"label": t.title(), "value": t} for t in ALL_TOPICS],
        value=TOP10["canonical_topic"].tolist()[:6],
        multi=True, clearable=False,
        style={
            "background": P["surface"], "borderColor": P["border"],
            "borderRadius": "8px", "fontSize": "0.8rem",
            "color": P["text"], "marginBottom": "0.75rem",
        },
    )

    pills = [
        pill(i + 1, row["canonical_topic"], row["score"], row["delta"])
        for i, row in TOP10.iterrows()
    ]

    short_name = (top_name[:24] + "…") if len(top_name) > 26 else top_name

    return html.Div([

        # ── Navbar
        html.Div(id="navbar", children=[
            html.Div([
                html.Div("⚡ Tech Trend Intelligence", className="nav-logo"),
                html.Div(
                    f"Week of {LATEST.strftime('%b %d, %Y')}  ·  {n_topics} canonical topics",
                    className="nav-meta"
                ),
            ]),
            badge,
        ]),

        # ── Main
        html.Div(id="content", children=[

            # KPIs
            html.Div(id="kpi-row", children=[
                kpi("🏷", "Topics Tracked",   str(n_topics),          "canonical + discovered", P["primary"]),
                kpi("🔥", "Top Topic",         short_name,             f"score {top_score:.4f}", P["violet"]),
                kpi("📡", "This-Week Signals", f"{week_tot:,}",        "across all sources",     P["amber"]),
                kpi("📈", "Rising Topics",     f"{n_rising} / {len(TOP10)}", "vs last week",     P["rose"]),
            ]),

            # Row 1: Top-10 bar + source donut
            html.Div(className="row-2l", children=[
                card("Top 10 Topics This Week",
                    dcc.Graph(id="g-top10",  figure=chart_top10(),       config={"displayModeBar": False}),
                ),
                card("Signal Sources (This Week)",
                    dcc.Graph(id="g-donut",  figure=chart_source_donut(), config={"displayModeBar": False}),
                ),
            ]),

            # Row 2: Trend lines (full width)
            html.Div(className="row-full", children=[
                card("Trend Trajectories Over Time",
                    dropdown,
                    dcc.Graph(id="g-trends", figure=chart_trends(), config={"displayModeBar": False}),
                ),
            ]),

            # Row 3: Source timeline (full width)
            html.Div(className="row-full", children=[
                card("Source Activity Over Time",
                    dcc.Graph(id="g-src-timeline", figure=chart_source_timeline(), config={"displayModeBar": False}),
                ),
            ]),

            # Row 4: Scatter + Momentum
            html.Div(className="row-2", children=[
                card("Academic vs Social Signal Map",
                    dcc.Graph(id="g-scatter",  figure=chart_scatter(),  config={"displayModeBar": False}),
                ),
                card("Week-over-Week Momentum",
                    dcc.Graph(id="g-momentum", figure=chart_momentum(), config={"displayModeBar": False}),
                ),
            ]),

            # Row 5: Findings pills
            html.Div(className="row-full", children=[
                card("Findings: Top 10 Topics This Week",
                    html.Div(id="pills-grid", children=pills),
                ),
            ]),

        ]),
    ])


app.layout = serve_layout


# ─── CALLBACKS ──────────────────────────────────────────────────────────────

@app.callback(
    Output("g-trends", "figure"),
    Input("topic-selector", "value"),
)
def update_trends(sel):
    return chart_trends(sel or TOP10["canonical_topic"].tolist()[:6])


# ─── ENTRY POINT ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 54)
    print("  🚀  Tech Trend Intelligence Dashboard")
    print("=" * 54)
    print(f"  Week  : {LATEST.strftime('%Y-%m-%d')}")
    print(f"  Topics: {SCORES['canonical_topic'].nunique()}")
    print(f"  Mode  : {'DEMO (synthetic)' if DEMO else 'LIVE DATA'}")
    print(f"  URL   : http://127.0.0.1:8050")
    print("=" * 54)
    print()
    app.run(debug=True, host="0.0.0.0", port=8050)
