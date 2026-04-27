#!/usr/bin/env python3
"""
Tech Trend Intelligence Dashboard
==================================
Run from the project root:  python src/dashboard.py
Install extras:             pip install dash plotly
"""

import os, re, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from functools import lru_cache

import dash
from dash import dcc, html, Input, Output

warnings.filterwarnings("ignore")

# ─── PATHS ───────────────────────────────────────────────────────────────────

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

# ─── DESIGN TOKENS ───────────────────────────────────────────────────────────

P = {
    "bg":      "#080d1a",  "surface": "#0f1729",  "card": "#111827",
    "border":  "#1c2d4e",  "primary": "#00e5c3",  "violet": "#7c6af7",
    "amber":   "#ffb547",  "rose":    "#ff5272",  "text":   "#d6e8ff",
    "muted":   "#4a6d9c",  "grid":    "#192038",  "green":  "#22c55e",
}

SRC_COLORS = {
    "arxiv": "#7c6af7", "semantic_scholar": "#a78bfa",
    "reddit": "#ff5272", "hackernews": "#ffb547", "github": "#00e5c3",
}

CHART_COLORS = [P["primary"], P["violet"], P["amber"], P["rose"],
                "#38bdf8", "#a78bfa", "#34d399", "#fb923c", "#f472b6", "#818cf8"]

BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color=P["text"], size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hoverlabel=dict(bgcolor=P["surface"], bordercolor=P["border"], font_size=12),
    margin=dict(l=0, r=0, t=20, b=0),
)
AX = dict(gridcolor=P["grid"], linecolor=P["border"], tickcolor=P["muted"])

WEEK_RE = re.compile(r'^\d{4}-W\d{2}$')

# ─── WEEK HELPERS ────────────────────────────────────────────────────────────

@lru_cache(maxsize=2048)
def parse_week(w):
    try:
        return datetime.strptime(w + '-1', '%G-W%V-%u')
    except Exception:
        return None

def week_offset(w, n):
    d = parse_week(w)
    if not d:
        return ""
    d2 = d + timedelta(weeks=n)
    iso = d2.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"

def date_to_week(s):
    try:
        d = datetime.strptime(str(s)[:10], "%Y-%m-%d")
        iso = d.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except Exception:
        return ""

def hex_rgba(h, a=0.7):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

# ─── DATA I/O ────────────────────────────────────────────────────────────────

def is_lfs(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return "git-lfs" in f.readline()
    except Exception:
        return True

def safe_load(path, **kw):
    if not path or not os.path.exists(path):
        return None
    if is_lfs(path):
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None

# ─── ANALYTICS ───────────────────────────────────────────────────────────────

def compute_burst(sig_weekly):
    """Causal 8-week rolling z-score — matches EDA methodology exactly."""
    rows = []
    for topic, grp in sig_weekly.groupby("canonical_topic"):
        g = grp.sort_values("week").reset_index(drop=True)
        if len(g) < 4:
            continue
        weeks    = g["week"].tolist()
        mentions = g["mentions"].to_numpy(dtype=float)
        for i in range(len(weeks)):
            past     = mentions[max(0, i - 8):i]
            mean_v   = past.mean() if len(past) >= 2 else mentions[i]
            std_v    = past.std()  if len(past) >= 2 else 1.0
            burst    = (mentions[i] - mean_v) / (std_v + 1)
            rows.append({"canonical_topic": topic, "week": weeks[i],
                         "mentions": int(mentions[i]),
                         "rolling_mean": round(mean_v, 2),
                         "burst_score":  round(burst, 3)})
    return pd.DataFrame(rows)


def compute_acad_social(sig_src):
    """Compute normalised academic and social scores per topic per week."""
    ACAD = {"arxiv", "semantic_scholar"}
    SOC  = {"reddit", "hackernews", "github"}
    if "source" not in sig_src.columns:
        return pd.DataFrame()
    grp = sig_src.groupby(["canonical_topic", "week", "source"])
    if "mentions" in sig_src.columns:
        cnt = grp["mentions"].sum().reset_index()
    else:
        cnt = grp.size().reset_index(name="mentions")
    a = cnt[cnt["source"].isin(ACAD)].groupby(["canonical_topic","week"])["mentions"].sum().rename("academic")
    s = cnt[cnt["source"].isin(SOC)].groupby(["canonical_topic","week"])["mentions"].sum().rename("social")
    df = pd.concat([a, s], axis=1).fillna(0).reset_index()
    mx_a = df["academic"].max() or 1
    mx_s = df["social"].max()   or 1
    df["academic_n"] = df["academic"] / mx_a
    df["social_n"]   = df["social"]   / mx_s
    df["combined"]   = 0.4 * df["academic_n"] + 0.6 * df["social_n"]
    return df


def compute_hist_features(all_topics, nl_weekly, sig_weekly):
    """Causal historical NL appearance features (EDA 2)."""
    nl_map     = {}
    active_map = {}
    for _, r in nl_weekly.iterrows():
        nl_map.setdefault(r["canonical_topic"], [])
        nl_map[r["canonical_topic"]].append(r["week"])
    for _, r in sig_weekly.iterrows():
        active_map.setdefault(r["canonical_topic"], [])
        active_map[r["canonical_topic"]].append(r["week"])
    for m in (nl_map, active_map):
        for t in m:
            m[t] = sorted(m[t])
    rows = []
    for topic in all_topics:
        aw = active_map.get(topic, [])
        nw = nl_map.get(topic, [])
        for i, week in enumerate(aw):
            pa    = aw[:i]
            pn    = [w for w in nw if w < week]
            ra    = len(pn) / len(pa) if pa else 0.0
            pa8   = pa[-8:]
            pn8   = [w for w in pn if w in set(pa8)]
            r8    = len(pn8) / len(pa8) if pa8 else 0.0
            if pn:
                ld = parse_week(pn[-1]); cd = parse_week(week)
                wsnl = (cd - ld).days // 7 if (ld and cd) else 999
            else:
                wsnl = 999
            rows.append({"canonical_topic": topic, "week": week,
                         "topic_nl_rate_alltime": round(ra, 4),
                         "topic_nl_rate_8wk":     round(r8, 4),
                         "weeks_since_last_nl":   min(wsnl, 200),
                         "is_novel_topic":        int(not pn),
                         "past_nl_count":         len(pn)})
    return pd.DataFrame(rows)


def assign_labels(burst_df, nl_set, window=2):
    labels = [int(any((r["canonical_topic"], week_offset(r["week"], off)) in nl_set
                      for off in range(1, window + 1)))
              for _, r in burst_df.iterrows()]
    return burst_df.assign(label_next2wk=labels)

# ─── SYNTHETIC FALLBACK ──────────────────────────────────────────────────────

SEED_TOPICS = [
    "large language model", "autonomous AI agent framework", "AI coding assistant tool",
    "retrieval augmented generation", "open source model weights release",
    "CRISPR gene editing therapy", "NVIDIA GPU datacenter", "GLP-1 obesity weight loss drug",
    "quantum computing qubit", "diffusion model image generation",
    "cybersecurity ransomware attack", "protein structure AlphaFold prediction",
    "algorithmic trading strategy", "robotics humanoid robot",
    "semiconductor chip manufacturing", "reinforcement learning from human feedback",
    "drug discovery clinical trial", "AI safety alignment research",
    "decentralized finance DeFi protocol", "AI regulation government policy",
    "fine tuning language model", "vector database embeddings",
    "AI energy consumption datacenter", "vision language model", "ai advancements openai",
]

def generate_synthetic():
    rng  = np.random.default_rng(42)
    end  = datetime.now()
    weeks = [f"{(end-timedelta(weeks=i)).isocalendar()[0]}-W{(end-timedelta(weeks=i)).isocalendar()[1]:02d}"
             for i in range(79, -1, -1)]
    srcs = ["arxiv","semantic_scholar","reddit","hackernews","github"]
    base  = {t: rng.uniform(0.1, 0.75)    for t in SEED_TOPICS}
    slope = {t: rng.uniform(-0.004, 0.011) for t in SEED_TOPICS}
    for t, (b, s) in {
        "large language model":           (0.94, 0.018),
        "autonomous AI agent framework":  (0.90, 0.016),
        "AI coding assistant tool":       (0.85, 0.014),
        "retrieval augmented generation": (0.82, 0.013),
        "open source model weights release":(0.79,0.012),
        "NVIDIA GPU datacenter":          (0.77, 0.011),
        "GLP-1 obesity weight loss drug": (0.73, 0.010),
    }.items():
        base[t], slope[t] = b, s

    SRC_W = {
        "arxiv":            lambda t: 0.85 if any(k in t for k in ["model","CRISPR","drug","protein","quantum"]) else 0.30,
        "semantic_scholar": lambda t: 0.88 if any(k in t for k in ["protein","drug","CRISPR","genomics","quantum"]) else 0.28,
        "reddit":           lambda t: 0.88 if any(k in t for k in ["AI","agent","coding","trading","LLM"]) else 0.45,
        "hackernews":       lambda t: 0.80,
        "github":           lambda t: 0.85 if any(k in t for k in ["model","framework","tool","code","open"]) else 0.30,
    }
    sig_rows, nl_rows = [], []
    for wi, week in enumerate(weeks):
        for topic in SEED_TOPICS:
            pop = float(np.clip(base[topic] + slope[topic]*wi + rng.normal(0,0.025), 0.02, 1.0))
            for src in srcs:
                n = max(0, int(pop * SRC_W[src](topic) * rng.lognormal(4.6, 0.55)))
                if n:
                    sig_rows.append({"canonical_topic": topic, "week": week, "source": src, "mentions": n})
            if pop > 0.75 and rng.random() < 0.06:
                nl_rows.append({"canonical_topic": topic, "week": week,
                                "phrase": topic, "score": 0.8, "source": "tldr_ai", "similarity": 0.85})
    sig_src = pd.DataFrame(sig_rows)
    sig_w   = sig_src.groupby(["canonical_topic","week"])["mentions"].sum().reset_index()
    nl_df   = pd.DataFrame(nl_rows) if nl_rows else pd.DataFrame(
        columns=["canonical_topic","week","phrase","score","source","similarity"])
    return sig_w, sig_src, nl_df, True

# ─── PIPELINE ────────────────────────────────────────────────────────────────

def run():
    sig_raw = safe_load(find_file("signal_topic_map.csv"), low_memory=False)
    nl_raw  = safe_load(find_file("newsletter_topic_map.csv"))
    demo    = False

    if sig_raw is not None and len(sig_raw) > 500:
        print(f"✓ Signal data  ({len(sig_raw):,} rows)")
        if "week" not in sig_raw.columns and "date" in sig_raw.columns:
            sig_raw["week"] = sig_raw["date"].apply(date_to_week)
        sig_raw  = sig_raw[sig_raw["week"].str.match(WEEK_RE, na=False)]
        sig_src  = sig_raw.copy()
        grp_cols = ["canonical_topic","week"]
        sig_w    = (sig_raw.groupby(grp_cols)["mentions"].sum().reset_index()
                    if "mentions" in sig_raw.columns else
                    sig_raw.groupby(grp_cols).size().reset_index(name="mentions"))
    else:
        print("ℹ  Demo mode — synthetic data")
        sig_w, sig_src, nl_raw, demo = generate_synthetic()

    if nl_raw is not None and len(nl_raw) > 10:
        print(f"✓ Newsletter data  ({len(nl_raw):,} rows)")
        if "week" not in nl_raw.columns and "date" in nl_raw.columns:
            nl_raw["week"] = nl_raw["date"].apply(date_to_week)
        nl_raw  = nl_raw[nl_raw["week"].str.match(WEEK_RE, na=False)]
        nl_w    = nl_raw.groupby(["canonical_topic","week"]).size().reset_index(name="nl_mentions")
        nl_w["in_newsletter"] = 1
    else:
        nl_w = pd.DataFrame(columns=["canonical_topic","week","nl_mentions","in_newsletter"])

    all_topics = sig_w["canonical_topic"].unique().tolist()
    burst_df   = compute_burst(sig_w)
    nl_set     = set(zip(nl_w["canonical_topic"], nl_w["week"]))
    burst_lab  = assign_labels(burst_df, nl_set)
    hist_df    = compute_hist_features(all_topics, nl_w, sig_w)
    as_df      = compute_acad_social(sig_src)

    M = burst_lab.merge(hist_df, on=["canonical_topic","week"], how="left").fillna(0)
    if not as_df.empty:
        M = M.merge(as_df[["canonical_topic","week","academic_n","social_n","combined"]],
                    on=["canonical_topic","week"], how="left").fillna(0)
    else:
        M["academic_n"] = 0.0; M["social_n"] = 0.0; M["combined"] = 0.0

    if "source" in sig_src.columns:
        src_cnt = sig_src.groupby(["canonical_topic","week"])["source"].nunique().reset_index(name="n_sources")
        M = M.merge(src_cnt, on=["canonical_topic","week"], how="left").fillna(0)
    else:
        M["n_sources"] = 1

    total = len(M); pos = int(M["label_next2wk"].sum()); neg = total - pos
    ratio = neg // max(pos, 1)
    print(f"  Modeling table: {total:,} rows | Pos: {pos} ({pos/max(total,1)*100:.1f}%) | 1:{ratio}")

    return {"sig_w": sig_w, "sig_src": sig_src, "nl_w": nl_w,
            "burst": burst_lab, "M": M, "all_topics": all_topics,
            "demo": demo,
            "stats": {"total": total, "pos": pos, "neg": neg,
                      "ratio": ratio, "n_topics": len(all_topics)}}

DATA = run()
M    = DATA["M"]
B    = DATA["burst"]

LATEST  = B["week"].max() if len(B) else ""
ALL_WKS = sorted(w for w in B["week"].unique() if WEEK_RE.match(w))
PREV    = ALL_WKS[-2] if len(ALL_WKS) >= 2 else ""

THIS_WK = (B[B["week"] == LATEST].sort_values("burst_score", ascending=False)
           .reset_index(drop=True) if LATEST else pd.DataFrame())
TOP10   = THIS_WK.head(10).copy()
PREV_B  = (B[B["week"] == PREV].set_index("canonical_topic")["burst_score"].to_dict()
           if PREV else {})
if len(TOP10):
    TOP10["prev_burst"] = TOP10["canonical_topic"].map(PREV_B).fillna(0)
    TOP10["delta"]      = TOP10["burst_score"] - TOP10["prev_burst"]

WK_DISP = parse_week(LATEST).strftime("%b %d, %Y") if parse_week(LATEST) else LATEST

FEAT_COLS = ["burst_score","mentions","topic_nl_rate_alltime","topic_nl_rate_8wk",
             "weeks_since_last_nl","is_novel_topic","n_sources","past_nl_count","label_next2wk"]
_fc = M[[c for c in FEAT_COLS if c in M.columns]].corr()
FEAT_CORR = _fc["label_next2wk"].drop("label_next2wk", errors="ignore").sort_values()

# ─── CHARTS ──────────────────────────────────────────────────────────────────

# ── 1. Top-10 burst bar (with ★ newsletter predictions) ──────────────────────
def chart_top_burst():
    df   = TOP10.copy().reset_index(drop=True)
    n    = len(df)
    if n == 0:
        return go.Figure()
    nl_s = set(zip(DATA["nl_w"]["canonical_topic"], DATA["nl_w"]["week"]))
    pred = [any((r["canonical_topic"], week_offset(LATEST, off)) in nl_s
                for off in range(1, 3)) for _, r in df.iterrows()]
    labels = [("★ " if pred[i] else "") + df.iloc[i]["canonical_topic"].title() for i in range(n)]
    colors = [P["green"] if pred[i] else
              f"rgba({int(108*i/max(n-1,1))},{229-int(80*i/max(n-1,1))},{195-int(60*i/max(n-1,1))},0.82)"
              for i in range(n)]
    fig = go.Figure(go.Bar(
        y=labels[::-1], x=df["burst_score"][::-1], orientation="h",
        marker=dict(color=colors[::-1], line=dict(width=0)),
        customdata=df["mentions"][::-1].values.reshape(-1, 1),
        hovertemplate="<b>%{y}</b><br>Burst: %{x:.2f}σ  ·  Mentions: %{customdata[0]:,.0f}<extra></extra>",
    ))
    fig.update_layout(**BASE)
    fig.update_layout(height=380, bargap=0.28,
        xaxis=dict(**AX, title="Burst score (σ above 8-wk baseline)"),
        yaxis=dict(**AX, tickfont=dict(size=10.5)),
        annotations=[dict(x=0.99, y=0.01, xref="paper", yref="paper", showarrow=False,
            text="★ = predicted in newsletter next 2 weeks",
            font=dict(size=9, color=P["green"]), xanchor="right")])
    return fig


# ── 2. Source donut ───────────────────────────────────────────────────────────
def chart_source_donut():
    src = DATA["sig_src"]
    if "source" not in src.columns:
        return go.Figure()
    wk_src = src[src["week"] == LATEST] if "week" in src.columns else src
    if "mentions" in wk_src.columns:
        totals = wk_src.groupby("source")["mentions"].sum().reset_index()
    else:
        totals = wk_src.groupby("source").size().reset_index(name="mentions")
    totals = totals.sort_values("mentions", ascending=False)
    total  = int(totals["mentions"].sum())
    fig = go.Figure(go.Pie(
        labels=totals["source"].str.replace("_"," ").str.title(),
        values=totals["mentions"], hole=0.62,
        marker=dict(colors=[SRC_COLORS.get(s,"#888") for s in totals["source"]],
                    line=dict(color=P["bg"], width=3)),
        textfont=dict(size=10.5),
        hovertemplate="%{label}<br>%{value:,} mentions · %{percent}<extra></extra>",
    ))
    fig.update_layout(**BASE)
    fig.update_layout(height=300, showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=10)),
        annotations=[dict(text=f"<b>{total:,}</b><br>mentions",
            x=0.5, y=0.5, showarrow=False, font=dict(color=P["text"], size=13))])
    return fig


# ── 3. Week-over-week momentum ────────────────────────────────────────────────
def chart_momentum():
    df = TOP10.copy()
    if "delta" not in df.columns or len(df) == 0:
        return go.Figure()
    df = df.sort_values("delta")
    colors = [P["green"] if v >= 0 else P["rose"] for v in df["delta"]]
    fig = go.Figure(go.Bar(
        y=df["canonical_topic"].str.title(),
        x=df["delta"], orientation="h",
        marker=dict(color=colors, opacity=0.82, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Burst Δ: %{x:+.3f}σ<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=P["border"], width=1.5))
    fig.update_layout(**BASE)
    fig.update_layout(height=300, margin=dict(l=0, r=15, t=20, b=0),
        xaxis=dict(**AX, title="Burst score change vs. previous week"),
        yaxis=dict(**AX, tickfont=dict(size=10)))
    return fig


# ── 4. Academic vs Social scatter ─────────────────────────────────────────────
def chart_scatter():
    latest_M = M[M["week"] == LATEST].copy() if LATEST else pd.DataFrame()
    if len(latest_M) == 0 or "academic_n" not in latest_M.columns:
        return go.Figure()
    df = latest_M.nlargest(22, "burst_score").reset_index(drop=True)

    # Scale marker size by burst score — floor at 12 so tiny bursts are still visible
    sizes  = df["burst_score"].clip(0) * 22 + 12
    # Color by combined score; use the full 0-1 range explicitly
    c_vals = df["combined"].clip(0, 1)

    fig = go.Figure()

    # Main scatter — markers only, full name on hover
    fig.add_trace(go.Scatter(
        x=df["academic_n"], y=df["social_n"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=c_vals,
            colorscale=[[0, P["violet"]], [0.45, P["primary"]], [1, "#ffffff"]],
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(
                title=dict(text="Combined score", font=dict(size=10)),
                thickness=12, len=0.55, tickfont=dict(size=9),
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ),
            line=dict(width=1.5, color=P["border"]),
            opacity=0.9,
        ),
        customdata=np.stack([
            df["canonical_topic"],
            df["burst_score"].round(2),
            df["academic_n"].round(3),
            df["social_n"].round(3),
        ], axis=1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Academic: %{customdata[2]}  ·  Social: %{customdata[3]}<br>"
            "Burst: %{customdata[1]}σ"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # Number labels centred on each bubble — far more readable than overlapping names
    fig.add_trace(go.Scatter(
        x=df["academic_n"], y=df["social_n"],
        mode="text",
        text=[str(i + 1) for i in range(len(df))],
        textfont=dict(size=8, color=P["bg"], family="IBM Plex Mono"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Quadrant dividers
    fig.add_hline(y=0.5, line=dict(dash="dot", color=P["border"], width=1))
    fig.add_vline(x=0.5, line=dict(dash="dot", color=P["border"], width=1))

    # Quadrant labels — plain text, no emojis
    for px, py, label in [
        (0.76, 0.93, "Trending Everywhere"),
        (0.04, 0.93, "Social Hype"),
        (0.76, 0.04, "Academic Focus"),
        (0.04, 0.04, "Emerging"),
    ]:
        fig.add_annotation(x=px, y=py, text=label, showarrow=False,
                           font=dict(size=9, color=P["muted"], family="IBM Plex Sans"),
                           xref="paper", yref="paper", xanchor="left" if px < 0.5 else "right")

    # Numbered legend below the chart
    legend_lines = []
    for i, row in df.iterrows():
        legend_lines.append(f"{i+1}. {row['canonical_topic'].title()}")

    fig.update_layout(**BASE)
    fig.update_layout(
        height=400,
        xaxis=dict(**AX, title="Academic signal (normalised)", range=[-0.04, 1.08]),
        yaxis=dict(**AX, title="Social signal (normalised)",   range=[-0.04, 1.08]),
    )
    return fig, legend_lines


# ── 5. Burst heatmap (EDA 3) ──────────────────────────────────────────────────
def chart_heatmap():
    top_topics = (M.sort_values("burst_score", ascending=False)
                  .drop_duplicates("canonical_topic").head(20)["canonical_topic"].tolist())
    sampled = ALL_WKS[::2]
    if len(sampled) > 60:
        sampled = sampled[-60:]
    b_lk = {(r["canonical_topic"],r["week"]): r["burst_score"] for _,r in B.iterrows()}
    l_lk = {(r["canonical_topic"],r["week"]): r.get("label_next2wk",0) for _,r in M.iterrows()}
    Z  = np.clip(np.array([[b_lk.get((t,w),0) for w in sampled] for t in top_topics]), -1, 8)
    sx, sy = [], []
    for ti,t in enumerate(top_topics):
        for wi,w in enumerate(sampled):
            if l_lk.get((t,w),0) == 1:
                sx.append(wi); sy.append(ti)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=Z, x=sampled, y=[t[:42] for t in top_topics],
        colorscale=[[0,"#7f0000"],[0.35,"#CC3300"],[0.5,"#FF8C00"],
                    [0.65,"#FFFFCC"],[0.82,"#90EE90"],[1,"#005700"]],
        zmin=-1, zmax=6,
        colorbar=dict(title=dict(text="Burst (σ)", font=dict(size=10)),
                      thickness=12, len=0.6, tickfont=dict(size=9)),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}σ<extra></extra>",
    ))
    if sx:
        fig.add_trace(go.Scatter(
            x=[sampled[i] for i in sx], y=[top_topics[i][:42] for i in sy],
            mode="text", text=["★"]*len(sx),
            textfont=dict(size=8, color="black"),
            hoverinfo="skip", showlegend=False))
    fig.update_layout(**BASE)
    fig.update_layout(height=480,
        xaxis=dict(**AX, tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(**AX, tickfont=dict(size=9.5), autorange="reversed"))
    return fig


# ── 6. Editorial habit distributions (EDA 2) ─────────────────────────────────
def chart_editorial_habit():
    pos  = M[M["label_next2wk"] == 1]
    neg  = M[M["label_next2wk"] == 0]
    FEATS = [("topic_nl_rate_alltime","All-time NL rate",0,1.5),
             ("topic_nl_rate_8wk",    "Recent 8-week NL rate",0,0.6),
             ("weeks_since_last_nl",  "Weeks since last NL",0,100)]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f[1] for f in FEATS],
                        horizontal_spacing=0.09)
    for ci,(feat,_,lo,hi) in enumerate(FEATS,1):
        if feat not in M.columns:
            continue
        pv = pos[feat].clip(lo,hi); nv = neg[feat].clip(lo,hi)
        for vals, name, col, show in [(nv,"Not in NL",P["muted"],ci==1),(pv,"In NL (next 2wk)",P["green"],ci==1)]:
            fig.add_trace(go.Histogram(x=vals, nbinsx=30, name=name,
                marker_color=col, opacity=0.6 if "Not" in name else 0.75,
                histnorm="probability density", showlegend=show), row=1, col=ci)
        for vals, col in [(pv,P["green"]),(nv,P["muted"])]:
            if len(vals):
                fig.add_vline(x=float(vals.median()), line_color=col,
                              line_dash="dash", line_width=1.5, row=1, col=ci)
    fig.update_layout(**BASE)
    fig.update_layout(height=280, barmode="overlay",
        legend=dict(orientation="h", y=-0.22, font=dict(size=10)))
    for ax in ["xaxis","xaxis2","xaxis3","yaxis","yaxis2","yaxis3"]:
        fig.update_layout(**{ax: dict(**AX)})
    return fig


# ── 7. Feature correlations (EDA 6) ──────────────────────────────────────────
def chart_feat_corr():
    df = FEAT_CORR.reset_index()
    df.columns = ["feature","corr"]
    LABELS = {"burst_score":"Burst score","mentions":"Raw mentions",
              "topic_nl_rate_alltime":"NL rate all-time ★★",
              "topic_nl_rate_8wk":"NL rate 8-week ★★",
              "weeks_since_last_nl":"Weeks since last NL ★★",
              "is_novel_topic":"Novel topic flag","n_sources":"# sources active",
              "past_nl_count":"Past NL appearances"}
    df["label"]  = df["feature"].map(LABELS).fillna(df["feature"])
    df["color"]  = [P["green"] if v>0 else P["rose"] for v in df["corr"]]
    fig = go.Figure(go.Bar(
        y=df["label"], x=df["corr"], orientation="h",
        marker=dict(color=df["color"], opacity=0.85, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>r = %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=P["border"], width=1.5))
    fig.update_layout(**BASE)
    fig.update_layout(height=310, margin=dict(l=0,r=15,t=20,b=0),
        xaxis=dict(**AX, title="Pearson r with label_next2wk"),
        yaxis=dict(**AX, tickfont=dict(size=10.5)),
        annotations=[dict(x=0.99,y=0.01,xref="paper",yref="paper",showarrow=False,
            text="★★ p<0.0001 (EDA 2)", font=dict(size=9,color=P["green"]),xanchor="right")])
    return fig


# ── 8. Class imbalance donut ──────────────────────────────────────────────────
def chart_imbalance():
    s = DATA["stats"]
    fig = go.Figure(go.Pie(
        labels=["Negative","Positive (NL next 2wk)"],
        values=[s["neg"], s["pos"]], hole=0.62,
        marker=dict(colors=[P["muted"],P["green"]], line=dict(color=P["bg"],width=3)),
        textfont=dict(size=10.5),
        hovertemplate="%{label}<br>%{value:,} · %{percent}<extra></extra>",
    ))
    fig.update_layout(**BASE)
    fig.update_layout(height=280, showlegend=True,
        legend=dict(orientation="h", y=-0.08, font=dict(size=10)),
        annotations=[dict(text=f"<b>1:{s['ratio']}</b><br><span style='font-size:10px'>class ratio</span>",
            x=0.5,y=0.5,showarrow=False,font=dict(color=P["amber"],size=14))])
    return fig


# ── 9. Source timeline ────────────────────────────────────────────────────────
def chart_src_timeline():
    src = DATA["sig_src"]
    if "source" not in src.columns:
        return go.Figure()
    if "mentions" in src.columns:
        src_w = src.groupby(["week","source"])["mentions"].sum().reset_index()
    else:
        src_w = src.groupby(["week","source"]).size().reset_index(name="mentions")
    fig = go.Figure()
    for s_name in ["arxiv","semantic_scholar","reddit","hackernews","github"]:
        d = src_w[src_w["source"]==s_name].sort_values("week")
        if d.empty: continue
        c = SRC_COLORS.get(s_name,"#888888")
        fig.add_trace(go.Scatter(
            x=d["week"], y=d["mentions"], mode="lines",
            name=s_name.replace("_"," ").title(),
            stackgroup="one", line=dict(width=0), fillcolor=hex_rgba(c,0.72),
            hovertemplate="%{y:,.0f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(**BASE)
    fig.update_layout(height=260, hovermode="x unified",
        xaxis=dict(**AX), yaxis=dict(**AX, title="Mentions"),
        legend=dict(orientation="h", y=-0.22, font=dict(size=9.5)))
    return fig


# ── 10. Trend lines (interactive) ─────────────────────────────────────────────
def chart_trends(topics=None):
    if not topics:
        topics = TOP10["canonical_topic"].tolist()[:5] if len(TOP10) else []
    df = M[M["canonical_topic"].isin(topics)]
    fig = go.Figure()
    for i, topic in enumerate(topics):
        t = df[df["canonical_topic"]==topic].sort_values("week")
        if t.empty: continue
        c = CHART_COLORS[i % len(CHART_COLORS)]
        r2,g2,b2 = int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)
        fig.add_trace(go.Scatter(
            x=t["week"], y=t["burst_score"], mode="lines",
            name=topic.title(),
            line=dict(color=c, width=2.2, shape="spline"),
            fill="tozeroy", fillcolor=f"rgba({r2},{g2},{b2},0.07)",
            hovertemplate="%{y:.2f}σ<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(**BASE)
    fig.update_layout(height=290, hovermode="x unified",
        xaxis=dict(**AX), yaxis=dict(**AX, title="Burst score (σ)"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=9.5)))
    return fig


# ── 11. Topic pills ────────────────────────────────────────────────────────────
def make_pills():
    nl_set = set(zip(DATA["nl_w"]["canonical_topic"], DATA["nl_w"]["week"]))
    pills  = []
    for i, row in TOP10.iterrows():
        topic = row["canonical_topic"]
        pred  = any((topic, week_offset(LATEST, off)) in nl_set for off in range(1,3))
        pills.append(html.Div(className="pill", children=[
            html.Span(f"#{i+1}", className="pill-rank"),
            html.Span(("★ " if pred else "") + topic.title(), className="pill-name",
                      style={"color": P["green"] if pred else P["text"]}),
        ]))
    return pills

# ─── CSS ─────────────────────────────────────────────────────────────────────

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600&family=IBM+Plex+Mono:wght@400;500&display=swap');
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
html{{scroll-behavior:smooth;}}
body,#_dash-app-content{{background:{P['bg']} !important;color:{P['text']};font-family:'IBM Plex Sans',sans-serif;font-size:14px;line-height:1.6;}}
::-webkit-scrollbar{{width:6px;}} ::-webkit-scrollbar-track{{background:{P['bg']};}} ::-webkit-scrollbar-thumb{{background:{P['border']};border-radius:3px;}}

/* ── Navbar ── */
#navbar{{background:linear-gradient(135deg,{P['bg']} 0%,{P['surface']} 100%);border-bottom:1px solid {P['border']};padding:1rem 2.5rem;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;}}
.nav-logo{{font-family:'Syne',sans-serif;font-weight:800;font-size:1.35rem;letter-spacing:-0.5px;background:linear-gradient(100deg,{P['primary']},{P['violet']});-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.nav-meta{{font-size:0.70rem;color:{P['muted']};font-family:'IBM Plex Mono',monospace;margin-top:3px;}}
.badge-demo{{background:rgba(255,180,70,0.12);color:{P['amber']};border:1px solid rgba(255,180,70,0.3);border-radius:20px;padding:3px 12px;font-size:0.70rem;font-family:'IBM Plex Mono',monospace;}}
.badge-live{{background:rgba(0,229,195,0.10);color:{P['primary']};border:1px solid rgba(0,229,195,0.28);border-radius:20px;padding:3px 12px;font-size:0.70rem;font-family:'IBM Plex Mono',monospace;}}

/* ── Layout ── */
#content{{max-width:1440px;margin:0 auto;padding:2rem 2.5rem 4rem;}}
#kpi-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-bottom:1.5rem;}}
.row-2l{{display:grid;grid-template-columns:1.7fr 1fr;gap:1.2rem;margin-bottom:1.2rem;}}
.row-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.2rem;margin-bottom:1.2rem;}}
.row-2{{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin-bottom:1.2rem;}}
.row-full{{margin-bottom:1.2rem;}}

/* ── KPI cards ── */
.kpi{{background:{P['card']};border:1px solid {P['border']};border-radius:12px;padding:1rem 1.2rem;display:flex;align-items:center;gap:0.8rem;transition:border-color .2s,box-shadow .2s;}}
.kpi:hover{{border-color:rgba(0,229,195,0.3);box-shadow:0 4px 24px rgba(0,229,195,0.06);}}
.kpi-icon{{font-size:1.5rem;flex-shrink:0;}}
.kpi-label{{font-size:0.64rem;font-weight:500;color:{P['muted']};text-transform:uppercase;letter-spacing:0.8px;margin-bottom:3px;}}
.kpi-val{{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;line-height:1;margin-bottom:3px;}}
.kpi-sub{{font-size:0.63rem;color:{P['muted']};}}

/* ── Section cards ── */
.card{{background:{P['card']};border:1px solid {P['border']};border-radius:12px;padding:1.2rem 1.4rem;height:100%;}}
.card-title{{font-family:'Syne',sans-serif;font-size:0.76rem;font-weight:600;color:{P['muted']};text-transform:uppercase;letter-spacing:0.9px;margin-bottom:0.9rem;padding-bottom:0.6rem;border-bottom:1px solid {P['border']};display:flex;align-items:center;gap:0.5rem;}}
.card-title::before{{content:'';display:block;width:3px;height:13px;background:linear-gradient({P['primary']},{P['violet']});border-radius:2px;flex-shrink:0;}}
.section-label{{font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;color:{P['violet']};text-transform:uppercase;letter-spacing:1.5px;margin:1.5rem 0 0.7rem;padding-left:2px;}}

/* ── EDA finding boxes ── */
.finding{{background:{P['surface']};border:1px solid {P['border']};border-radius:8px;padding:0.7rem 0.9rem;margin-bottom:0.55rem;font-size:0.80rem;line-height:1.55;}}
.finding-null{{border-left:3px solid {P['rose']};}}
.finding-pos{{border-left:3px solid {P['green']};}}
.finding-neutral{{border-left:3px solid {P['amber']};}}
.finding b{{color:{P['primary']};}}

/* ── Topic pills (from original) ── */
#pills-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:0.55rem;}}
.pill{{background:{P['surface']};border:1px solid {P['border']};border-radius:9px;padding:0.6rem 0.9rem;display:flex;align-items:center;gap:0.7rem;transition:border-color .15s,background .15s;}}
.pill:hover{{border-color:rgba(0,229,195,0.25);background:rgba(0,229,195,0.03);}}
.pill-rank{{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:{P['muted']};min-width:22px;flex-shrink:0;}}
.pill-name{{font-size:0.79rem;flex:1;}}
.pill-right{{display:flex;flex-direction:column;align-items:flex-end;gap:1px;}}
.pill-score{{font-family:'IBM Plex Mono',monospace;font-size:0.79rem;font-weight:500;}}
.pill-delta{{font-family:'IBM Plex Mono',monospace;font-size:0.65rem;}}

/* ── Dropdown ── */
.Select-control,.Select-menu-outer{{background:{P['surface']} !important;border-color:{P['border']} !important;}}
.Select-value-label{{color:{P['text']} !important;}}

/* ── Responsive ── */
@media(max-width:1200px){{#kpi-row{{grid-template-columns:repeat(3,1fr);}} .row-2l,.row-3,.row-2{{grid-template-columns:1fr;}} #content{{padding:1.5rem 1.25rem 3rem;}}}}
@media(max-width:600px){{#kpi-row{{grid-template-columns:1fr 1fr;}} #pills-grid{{grid-template-columns:1fr;}}}}
"""

# ─── APP ─────────────────────────────────────────────────────────────────────

_assets = os.path.join(SCRIPT_DIR, "assets")
os.makedirs(_assets, exist_ok=True)
with open(os.path.join(_assets, "dashboard.css"), "w") as _f:
    _f.write(CSS)

app = dash.Dash(__name__, title="Tech Trend Intelligence",
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
        "https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600&family=IBM+Plex+Mono:wght@400;500&display=swap",
    ])

def card(title, *children):
    return html.Div(className="card", children=[
        html.Div(title, className="card-title"),
        *children,
    ])

def kpi(icon, label, value, sub, color):
    return html.Div(className="kpi", children=[
        html.Div(icon, className="kpi-icon", style={"color": color}),
        html.Div([html.Div(label, className="kpi-label"),
                  html.Div(value, className="kpi-val", style={"color": color}),
                  html.Div(sub,   className="kpi-sub")]),
    ])

def finding(children, kind="neutral"):
    return html.Div(children, className=f"finding finding-{kind}")

def _scatter_children():
    """Render scatter chart + numbered key below it."""
    fig, legend_lines = chart_scatter()
    # Build a two-column key from the numbered legend
    mid   = (len(legend_lines) + 1) // 2
    left  = legend_lines[:mid]
    right = legend_lines[mid:]
    key_cols = html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr",
                                "gap":"2px 1.5rem","marginTop":"0.65rem"}, children=[
        html.Span(line, style={"fontSize":"0.72rem","color":P["muted"],
                               "fontFamily":"IBM Plex Sans, sans-serif",
                               "whiteSpace":"nowrap","overflow":"hidden",
                               "textOverflow":"ellipsis"})
        for line in left + right
    ])
    return [dcc.Graph(figure=fig, config={"displayModeBar": False}), key_cols]


def serve_layout():
    s     = DATA["stats"]
    demo  = DATA["demo"]
    badge = (html.Span("⚙  DEMO MODE", className="badge-demo") if demo
             else html.Span("● LIVE DATA", className="badge-live"))

    top_feat = FEAT_CORR.abs().idxmax() if len(FEAT_CORR) else "N/A"
    top_feat_label = {"topic_nl_rate_8wk":"NL rate 8wk",
                      "topic_nl_rate_alltime":"NL rate all-time",
                      "weeks_since_last_nl":"Weeks since NL",
                      "burst_score":"Burst score"}.get(top_feat, str(top_feat))
    top_feat_r = abs(float(FEAT_CORR[top_feat])) if top_feat in FEAT_CORR.index else 0

    top_name = TOP10.iloc[0]["canonical_topic"].title()[:22] if len(TOP10) else "—"
    top_val  = f"{TOP10.iloc[0]['burst_score']:.2f}σ" if len(TOP10) else ""

    trend_drop = dcc.Dropdown(
        id="trend-sel",
        options=[{"label": t.title(), "value": t} for t in sorted(DATA["all_topics"])],
        value=TOP10["canonical_topic"].tolist()[:5] if len(TOP10) else [],
        multi=True, clearable=False,
        style={"background": P["surface"], "borderColor": P["border"],
               "borderRadius": "8px", "fontSize": "0.8rem",
               "color": P["text"], "marginBottom": "0.7rem"},
    )

    return html.Div([
        # ── Navbar
        html.Nav(id="navbar", children=[
            html.Div([
                html.Div("⚡ Tech Trend Intelligence", className="nav-logo"),
                html.Div(
                    f"Week of {WK_DISP}  ·  {s['n_topics']} topics  ·  "
                    f"{s['total']:,} active topic-weeks  ·  {s['pos']} newsletter labels",
                    className="nav-meta"),
            ]),
            badge,
        ]),

        html.Main(id="content", children=[

            # ── KPIs (5 cards)
            html.Div(id="kpi-row", children=[
                kpi("🏷", "Topics Tracked",     str(s["n_topics"]),         "canonical + discovered", P["primary"]),
                kpi("📰", "Newsletter Labels",   str(s["pos"]),              f"of {s['total']:,} active rows", P["green"]),
                kpi("⚠️", "Class Ratio",          f"1 : {s['ratio']}",       "use F1 / AUPRC",         P["amber"]),
                kpi("🏆", "Top Predictor",        top_feat_label,            f"r = {top_feat_r:.3f}",  P["violet"]),
                kpi("🔥", "Top Burst This Week",  top_name,                  top_val + " above baseline", P["rose"]),
            ]),

            # ── SECTION 1: Signal Intelligence
            html.P("SIGNAL INTELLIGENCE — THIS WEEK", className="section-label"),
            html.Div(className="row-2l", children=[
                card("Top 10 Burst Topics  ·  ★ = predicted newsletter in next 2 weeks",
                    dcc.Graph(figure=chart_top_burst(), config={"displayModeBar": False})),
                html.Div([
                    card("Source Mix This Week",
                        dcc.Graph(figure=chart_source_donut(), config={"displayModeBar": False})),
                ]),
            ]),

          

            # ── SECTION 2: EDA 3 — Burst Heatmap
            html.P("EDA 3 — BURST HEATMAP: TOP 20 TOPICS OVER TIME", className="section-label"),
            card("Burst score relative to each topic's own 8-week baseline  ·  ★ = newsletter hit within next 2 weeks",
                dcc.Graph(figure=chart_heatmap(), config={"displayModeBar": False})),

            # ── SECTION 3: EDA 2 — Editorial Habit
            html.P("EDA 2 — EDITORIAL HABIT: STRONGEST PREDICTORS (ALL p < 0.0001)", className="section-label"),
            html.Div(className="row-2", children=[
                card("Historical NL appearance features vs. forward coverage",
                    dcc.Graph(figure=chart_editorial_habit(), config={"displayModeBar": False})),
                card("EDA Key Findings", *[
                    finding(["EDA 1 — ", html.B("Burst score alone p=0.54"),
                              " (forward labels). Signal is contemporaneous, not leading. "
                              "Was p=0.026 with same-week labels — leakage, not predictiveness."], "null"),
                    finding(["EDA 2 — ", html.B("Editorial habit is the dominant signal"),
                              ". NL rate 8wk r=+0.22, recency r=−0.18. All three historical "
                              "features p<0.0001."], "pos"),
                    finding(["EDA 4 — Multi-source co-occurrence: monotonic pattern ",
                              html.B("disappears with forward labels"),
                              ". Contemporaneous, not leading."], "null"),
                    finding(["EDA 5 — Trajectory shape (spike vs. sustained) ",
                              html.B("not significant"),
                              " with forward labels (all p>0.34). Keep for interaction effects only."], "null"),
                    finding(["EDA 6 — 1:", html.B(f"{s['ratio']} imbalance"),
                              ". Use SMOTE on training only. "
                              "Predicting all-zero gives 96% accuracy — useless."], "neutral"),
                ]),
            ]),

            # ── SECTION 4: Modeling
            html.P("EDA 6 — FEATURE CORRELATIONS & CLASS BALANCE", className="section-label"),
            html.Div(className="row-2", children=[
                card("Feature correlations with label_next2wk",
                    dcc.Graph(figure=chart_feat_corr(), config={"displayModeBar": False})),
                card("Class balance — active topic-weeks only",
                    dcc.Graph(figure=chart_imbalance(), config={"displayModeBar": False})),
            ]),

            # ── SECTION 5: Source Activity
            html.P("SOURCE ACTIVITY OVER TIME", className="section-label"),
            card("Stacked signal volume by source",
                dcc.Graph(figure=chart_src_timeline(), config={"displayModeBar": False})),

            # ── SECTION 6: Trend drill-down
            html.P("BURST TRAJECTORY — TOPIC DRILL-DOWN", className="section-label"),
            card("Compare burst score trajectories over time",
                trend_drop,
                dcc.Graph(id="trend-chart", figure=chart_trends(),
                          config={"displayModeBar": False})),

            # ── SECTION 7: Findings pills (from original)
            html.P("FINDINGS — TOP 10 THIS WEEK", className="section-label"),
            card("Ranked by burst score  ·  ★ = newsletter-predicted  ·  arrow = vs. last week",
                html.Div(id="pills-grid", children=make_pills())),

        ]),
    ])

app.layout = serve_layout

@app.callback(Output("trend-chart","figure"), Input("trend-sel", "value"))
def cb_trends(sel):
    return chart_trends(sel or TOP10["canonical_topic"].tolist()[:5])

# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    s = DATA["stats"]
    print()
    print("=" * 60)
    print("  🚀  Tech Trend Intelligence Dashboard")
    print("=" * 60)
    print(f"  Week     : {WK_DISP}")
    print(f"  Topics   : {s['n_topics']}")
    print(f"  Labels   : {s['pos']} positive / {s['total']:,} active rows")
    print(f"  Ratio    : 1:{s['ratio']}")
    print(f"  Mode     : {'DEMO (synthetic)' if DATA['demo'] else 'LIVE DATA'}")
    print(f"  URL      : http://127.0.0.1:8050")
    print("=" * 60)
    print()
    app.run(debug=True, host="0.0.0.0", port=8050)
