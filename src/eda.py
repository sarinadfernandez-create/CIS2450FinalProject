import os
import warnings
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

ROOT      = os.path.dirname(os.path.abspath(__file__))
PROC_DIR  = os.path.join(ROOT, "data", "processed")
EXT_DIR   = os.path.join(ROOT, "data", "external")
OUT_DIR   = os.path.join(ROOT, "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE   = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2"]
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "sans-serif",
})

#  Constants 
# Sources treated as "signal" 
SIGNAL_SOURCES = ["arxiv", "semantic_scholar", "hackernews", "reddit", "github"]
# Sources treated as ground truth labels
NEWSLETTER_SOURCES = [
    "tldr_ai", "tldr_tech", "tldr_fintech",
    "tldr_founders", "import_ai", "bits_in_bio", "the_batch",
]
# Minimum weekly appearances for a topic to be considered "active"
MIN_TOPIC_APPEARANCES = 3
# Top N topics to show in frequency chart
TOP_N_FREQ = 20
# Topics to highlight in lead-lag chart
TOP_N_LEADLAG = 6


# DATA LOADING

def load_topics() -> pl.DataFrame:
    """
    Load topics.csv 

    Expected columns (from entity-linking pipeline):
        phrase, canonical_topic, score, source, date, week
    """
    path = os.path.join(PROC_DIR, "topics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"topics.csv not found at {path}\n"
            "Run entity-linking pipeline first."
        )
    df = pl.read_csv(path, infer_schema_length=5000)
    _require_cols(df, ["canonical_topic", "source", "date", "week"], "topics.csv")

    # Normalise source label: newsletter editions → "newsletter"
    df = df.with_columns(
        pl.when(pl.col("source").is_in(NEWSLETTER_SOURCES))
          .then(pl.lit("newsletter"))
          .otherwise(pl.col("source"))
          .alias("source_type")
    )
    return df


def load_newsletter_topics() -> pl.DataFrame:
    path = os.path.join(PROC_DIR, "newsletter_topics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"newsletter_topics.csv not found at {path}\n"
        )
    df = pl.read_csv(path, infer_schema_length=5000)
    _require_cols(df, ["week"], "newsletter_topics.csv")
    # canonical_topic may or may not exist depending on how far entity-linking ran
    if "canonical_topic" not in df.columns:
        if "phrase" in df.columns:
            df = df.rename({"phrase": "canonical_topic"})
        else:
            raise ValueError("newsletter_topics.csv needs 'canonical_topic' or 'phrase' column.")
    return df


def load_raw_for_sentiment() -> pl.DataFrame:
    """
    Load Reddit and HackerNews raw text for VADER sentiment scoring.
    Falls back to an empty DataFrame if files are missing.
    """
    frames = []
    for fname, text_col, src in [
        ("reddit.csv",      "text",  "reddit"),
        ("hackernews.csv",  "title", "hackernews"),
    ]:
        path = os.path.join(EXT_DIR, fname)
        if not os.path.exists(path):
            print(f"  [sentiment] {fname} not found — skipping")
            continue
        df = pl.read_csv(path, infer_schema_length=1000)
        if text_col not in df.columns:
            continue
        df = df.select([
            pl.col(text_col).alias("text"),
            pl.col("published").alias("date") if "published" in df.columns
              else pl.lit("").alias("date"),
        ]).filter(pl.col("text").str.len_chars() > 10)
        df = df.with_columns(pl.lit(src).alias("source"))
        frames.append(df)

    if not frames:
        return pl.DataFrame(schema={"text": pl.Utf8, "date": pl.Utf8, "source": pl.Utf8})
    return pl.concat(frames)


def _require_cols(df: pl.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}\nFound: {df.columns}")


# HELPERS

def build_weekly_counts(df: pl.DataFrame, sources: list[str] | None = None) -> pl.DataFrame:
    """
    Return a (canonical_topic, week, source → count) pivot-friendly DataFrame.
    """
    q = df
    if sources:
        q = df.filter(pl.col("source").is_in(sources))
    return (
        q.group_by(["canonical_topic", "week", "source"])
         .agg(pl.len().alias("count"))
         .sort(["canonical_topic", "week"])
    )


def get_newsletter_winners(nl_df: pl.DataFrame, min_appearances: int = 2) -> set[str]:
    """
    Topics that appear in newsletters at least `min_appearances` times.
    These are the positive labels.
    """
    counts = (
        nl_df.group_by("canonical_topic")
             .agg(pl.len().alias("n"))
             .filter(pl.col("n") >= min_appearances)
    )
    return set(counts["canonical_topic"].to_list())


def topic_first_seen(df: pl.DataFrame, source_type: str) -> pl.DataFrame:
    """Return (canonical_topic, first_week) per source type."""
    return (
        df.filter(pl.col("source_type") == source_type)
          .group_by("canonical_topic")
          .agg(pl.col("week").min().alias("first_week"))
    )


def weeks_between(w1: str, w2: str) -> int | None:
    """Return signed integer week difference between two ISO week strings (e.g. '2024-W05')."""
    try:
        def to_date(w):
            yr, wk = w.split("-W")
            return datetime.strptime(f"{yr}-W{int(wk):02d}-1", "%G-W%V-%u")
        d1, d2 = to_date(w1), to_date(w2)
        return round((d2 - d1).days / 7)
    except Exception:
        return None


# EDA 1 — TOPIC FREQUENCY DISTRIBUTION

def eda_topic_frequency(topics: pl.DataFrame, nl_df: pl.DataFrame):
    """
    Long-tail bar chart of topic frequency + per-source breakdown heatmap.
    """
    print("\n[EDA 1] Topic frequency distribution")

    winners = get_newsletter_winners(nl_df)

    # Total mentions per topic across signal sources
    signal_df = topics.filter(pl.col("source").is_in(SIGNAL_SOURCES))
    freq = (
        signal_df.group_by("canonical_topic")
                 .agg(pl.len().alias("total_mentions"))
                 .sort("total_mentions", descending=True)
    )

    top = freq.head(TOP_N_FREQ)
    topic_names = top["canonical_topic"].to_list()
    mention_counts = top["total_mentions"].to_list()
    is_winner = [t in winners for t in topic_names]

    # Per-source breakdown for the top topics
    src_breakdown = (
        signal_df.filter(pl.col("canonical_topic").is_in(topic_names))
                 .group_by(["canonical_topic", "source"])
                 .agg(pl.len().alias("count"))
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 11),
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("EDA 1 — Topic Frequency Distribution", fontsize=14, fontweight="bold", y=0.98)

    #  Bar chart 
    ax = axes[0]
    colours = [PALETTE[0] if w else PALETTE[5] for w in is_winner]
    bars = ax.barh(range(len(topic_names)), mention_counts, color=colours, height=0.7)
    ax.set_yticks(range(len(topic_names)))
    ax.set_yticklabels(topic_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Total mentions (all signal sources, 2023–present)")
    ax.set_title(f"Top {TOP_N_FREQ} topics by mention volume", fontsize=11)

    # Annotate counts
    for i, v in enumerate(mention_counts):
        ax.text(v + max(mention_counts) * 0.005, i, f"{v:,}", va="center", fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE[0], label="Newsletter winner"),
        Patch(facecolor=PALETTE[5], label="Not in newsletters"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    #  Per-source stacked breakdown for top 20 
    ax2 = axes[1]
    top20 = topic_names[:20]
    src_names = SIGNAL_SOURCES
    data_matrix = []
    for src in src_names:
        row = []
        for t in top20:
            sub = src_breakdown.filter(
                (pl.col("canonical_topic") == t) & (pl.col("source") == src)
            )
            row.append(sub["count"][0] if len(sub) > 0 else 0)
        data_matrix.append(row)

    x = np.arange(len(top20))
    bar_w = 0.15
    for i, (src, row) in enumerate(zip(src_names, data_matrix)):
        offset = (i - len(src_names) / 2) * bar_w
        ax2.bar(x + offset, row, bar_w, label=src, color=PALETTE[i % len(PALETTE)])

    ax2.set_xticks(x)
    ax2.set_xticklabels(top20, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Mentions")
    ax2.set_title("Source breakdown — top 20 topics", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right", ncol=len(src_names))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "01_topic_frequency.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")

    # Quick stats for summary
    total_topics = freq.height
    pct_covered = len(winners) / max(total_topics, 1) * 100
    print(f"  Total unique topics: {total_topics}")
    print(f"  Newsletter winners:  {len(winners)} ({pct_covered:.1f}%)")
    top5 = freq.head(5)["canonical_topic"].to_list()
    print(f"  Top 5 topics: {top5}")

    return {
        "total_topics": total_topics,
        "newsletter_winners": len(winners),
        "pct_winners": round(pct_covered, 1),
        "top5_topics": top5,
    }



# EDA 2 — LEAD-LAG RELATIONSHIPS


def eda_lead_lag(topics: pl.DataFrame, nl_df: pl.DataFrame):
    """
    For the most frequently discussed topics, plot normalised weekly timeseries
    per source and compute cross-correlation to measure lead times.
    """
    print("\n[EDA 2] Lead-lag relationships")

    winners = get_newsletter_winners(nl_df)

    # Focus on winner topics with enough data
    signal_df = topics.filter(pl.col("source").is_in(SIGNAL_SOURCES + ["newsletter"]))
    topic_counts = (
        signal_df.group_by("canonical_topic")
                 .agg(pl.len().alias("n"))
                 .filter(pl.col("n") >= MIN_TOPIC_APPEARANCES)
                 .filter(pl.col("canonical_topic").is_in(winners))
                 .sort("n", descending=True)
    )

    if topic_counts.is_empty():
        print("  No winner topics with sufficient data — skipping lead-lag plot.")
        _save_placeholder("02_lead_lag.png", "Lead-lag: insufficient data")
        return {"avg_lead_weeks": None, "note": "insufficient data"}

    focus_topics = topic_counts.head(TOP_N_LEADLAG)["canonical_topic"].to_list()

    # Build weekly pivot: week × source for each focus topic
    SOURCE_ORDER = ["arxiv", "semantic_scholar", "github", "hackernews", "reddit", "newsletter"]
    SOURCE_LABELS = {
        "arxiv": "Research (arXiv)",
        "semantic_scholar": "Research (SS)",
        "github": "GitHub activity",
        "hackernews": "HN discussion",
        "reddit": "Reddit discussion",
        "newsletter": "Newsletter pickup",
    }
    SOURCE_COLORS = {k: PALETTE[i] for i, k in enumerate(SOURCE_ORDER)}

    # All weeks in dataset
    all_weeks = sorted(signal_df["week"].drop_nulls().unique().to_list())

    fig, axes = plt.subplots(
        nrows=(len(focus_topics) + 1) // 2, ncols=2,
        figsize=(16, 4 * ((len(focus_topics) + 1) // 2)),
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle("EDA 2 — Lead-Lag Relationships Across Sources", fontsize=14,
                 fontweight="bold", y=1.01)

    lead_times = []

    for idx, topic in enumerate(focus_topics):
        ax = axes[idx]
        topic_df = signal_df.filter(pl.col("canonical_topic") == topic)

        series_dict = {}
        for src in SOURCE_ORDER:
            src_df = topic_df.filter(
                (pl.col("source") == src) |
                ((src == "newsletter") & pl.col("source").is_in(NEWSLETTER_SOURCES))
            )
            wkly = (
                src_df.group_by("week")
                      .agg(pl.len().alias("count"))
                      .sort("week")
            )
            # Align to all_weeks
            counts = []
            for w in all_weeks:
                row = wkly.filter(pl.col("week") == w)
                counts.append(row["count"][0] if len(row) > 0 else 0)
            series_dict[src] = np.array(counts, dtype=float)

        # Normalise each series to [0,1] range
        plotted = []
        for src in SOURCE_ORDER:
            s = series_dict[src]
            mx = s.max()
            if mx > 0:
                s_norm = s / mx
                plotted.append((src, s_norm))

        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(topic, fontsize=9)
            continue

        x = np.arange(len(all_weeks))
        for src, s_norm in plotted:
            ax.plot(x, s_norm, label=SOURCE_LABELS.get(src, src),
                    color=SOURCE_COLORS[src],
                    linewidth=1.8 if src == "newsletter" else 1.2,
                    linestyle="--" if src == "newsletter" else "-",
                    alpha=0.9)

        # Compute cross-correlation: research vs newsletter to estimate lead time
        res = series_dict.get("arxiv", np.zeros(len(all_weeks)))
        nl  = series_dict.get("newsletter", np.zeros(len(all_weeks)))
        if res.sum() > 0 and nl.sum() > 0:
            corr = np.correlate(nl - nl.mean(), res - res.mean(), mode="full")
            lags = np.arange(-(len(all_weeks) - 1), len(all_weeks))
            best_lag = lags[np.argmax(corr)]
            lead_times.append(best_lag)
            ax.set_title(f"{topic}  (research leads by ~{abs(best_lag)}w)", fontsize=8.5)
        else:
            ax.set_title(topic, fontsize=8.5)

        # X-axis: show every 4th week label
        tick_pos  = x[::4]
        tick_labs = [all_weeks[i] if i < len(all_weeks) else "" for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labs, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Relative activity (norm.)", fontsize=8)
        ax.set_ylim(-0.05, 1.15)

        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=2)

    # Hide any unused subplot panels
    for j in range(len(focus_topics), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "02_lead_lag.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")

    avg_lead = float(np.mean(lead_times)) if lead_times else None
    print(f"  Avg estimated research-to-newsletter lead time: "
          f"{avg_lead:.1f} weeks" if avg_lead else "  Could not compute lead time.")
    return {"avg_lead_weeks": round(avg_lead, 1) if avg_lead else None}


# EDA 3 — WINNING VS. LOSING TRAJECTORIES

def eda_winner_loser_trajectories(topics: pl.DataFrame, nl_df: pl.DataFrame):
    """
    Compare average normalised growth trajectories for topics that eventually
    appear in newsletters (winners) vs. those that don't (losers).
    Also shows individual trajectory examples.
    """
    print("\n[EDA 3] Winning vs. losing trajectories")

    winners = get_newsletter_winners(nl_df)
    signal_df = topics.filter(pl.col("source").is_in(SIGNAL_SOURCES))

    # Get all active topics
    active_topics = (
        signal_df.group_by("canonical_topic")
                 .agg(pl.len().alias("n"))
                 .filter(pl.col("n") >= MIN_TOPIC_APPEARANCES)
                 ["canonical_topic"].to_list()
    )

    all_weeks = sorted(signal_df["week"].drop_nulls().unique().to_list())
    W = len(all_weeks)
    week_idx = {w: i for i, w in enumerate(all_weeks)}

    # Bucket topics into winner / loser, build aligned time series
    winner_series, loser_series = [], []
    winner_examples, loser_examples = [], []

    for topic in active_topics:
        t_df = (
            signal_df.filter(pl.col("canonical_topic") == topic)
                     .group_by("week")
                     .agg(pl.len().alias("count"))
        )
        series = np.zeros(W)
        for row in t_df.iter_rows(named=True):
            if row["week"] in week_idx:
                series[week_idx[row["week"]]] = row["count"]

        mx = series.max()
        if mx == 0:
            continue
        norm = series / mx

        if topic in winners:
            winner_series.append(norm)
            if len(winner_examples) < 4:
                winner_examples.append((topic, norm))
        else:
            loser_series.append(norm)
            if len(loser_examples) < 4:
                loser_examples.append((topic, norm))

    if not winner_series or not loser_series:
        print("  Insufficient data for winner/loser comparison.")
        _save_placeholder("03_winner_loser_trajectories.png",
                          "Winner/loser trajectories: insufficient data")
        return {}

    avg_winner = np.mean(winner_series, axis=0)
    avg_loser  = np.mean(loser_series,  axis=0)
    std_winner = np.std(winner_series,  axis=0)
    std_loser  = np.std(loser_series,   axis=0)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    x = np.arange(W)
    tick_pos  = x[::4]
    tick_labs = [all_weeks[i] if i < W else "" for i in tick_pos]

    #  Top-left: Average trajectories 
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, avg_winner, color=PALETTE[1], linewidth=2.2, label=f"Winners (n={len(winner_series)})")
    ax1.fill_between(x,
                     np.clip(avg_winner - std_winner, 0, None),
                     avg_winner + std_winner,
                     color=PALETTE[1], alpha=0.15)
    ax1.plot(x, avg_loser, color=PALETTE[2], linewidth=2.2,
             linestyle="--", label=f"Non-winners (n={len(loser_series)})")
    ax1.fill_between(x,
                     np.clip(avg_loser - std_loser, 0, None),
                     avg_loser + std_loser,
                     color=PALETTE[2], alpha=0.15)
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_labs, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Relative mention volume (normalised)")
    ax1.set_title("Average trajectory: newsletter winners vs. non-winners  (±1 std dev)",
                  fontsize=11)
    ax1.legend(fontsize=9)

    #  Bottom-left: Individual winner examples 
    ax2 = fig.add_subplot(gs[1, 0])
    for topic, s in winner_examples:
        ax2.plot(x, s, linewidth=1.4, alpha=0.85,
                 label=topic[:30] + ("…" if len(topic) > 30 else ""))
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labs, rotation=30, ha="right", fontsize=7)
    ax2.set_title("Winner examples (individual)", fontsize=10)
    ax2.set_ylabel("Normalised activity")
    ax2.legend(fontsize=7, loc="upper left")

    #  Bottom-right: Individual loser examples 
    ax3 = fig.add_subplot(gs[1, 1])
    for topic, s in loser_examples:
        ax3.plot(x, s, linewidth=1.4, alpha=0.85, linestyle="--",
                 label=topic[:30] + ("…" if len(topic) > 30 else ""))
    ax3.set_xticks(tick_pos)
    ax3.set_xticklabels(tick_labs, rotation=30, ha="right", fontsize=7)
    ax3.set_title("Non-winner examples (individual)", fontsize=10)
    ax3.set_ylabel("Normalised activity")
    ax3.legend(fontsize=7, loc="upper left")

    fig.suptitle("EDA 3 — Growth Trajectory: Signal vs. Noise", fontsize=14,
                 fontweight="bold")

    out = os.path.join(OUT_DIR, "03_winner_loser_trajectories.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")

    # T-test on mean activity at peak
    peak_w = np.mean([s.max() for s in winner_series])
    peak_l = np.mean([s.max() for s in loser_series])
    tstat, pval = stats.ttest_ind(
        [s.max() for s in winner_series],
        [s.max() for s in loser_series],
    )
    print(f"  Mean peak activity — winners: {peak_w:.3f}, losers: {peak_l:.3f}")
    print(f"  T-test (peak): t={tstat:.2f}, p={pval:.4f}")
    return {
        "n_winners": len(winner_series),
        "n_losers":  len(loser_series),
        "mean_peak_winner": round(peak_w, 4),
        "mean_peak_loser":  round(peak_l, 4),
        "ttest_p": round(pval, 4),
    }


# EDA 4 — SENTIMENT DISTRIBUTION

def eda_sentiment_distribution(topics: pl.DataFrame, nl_df: pl.DataFrame):
    """
    Uses VADER to score Reddit posts and HN titles, then compares sentiment
    distributions for winner vs. non-winner topics.
    """
    print("\n[EDA 4] Sentiment distribution")

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
    except ImportError:
        print("  vaderSentiment not installed. Run: pip install vaderSentiment")
        _save_placeholder("04_sentiment_distribution.png",
                          "Sentiment: vaderSentiment not installed")
        return {"note": "vaderSentiment not installed"}

    winners = get_newsletter_winners(nl_df)
    raw_df  = load_raw_for_sentiment()

    if raw_df.is_empty():
        print("  No raw text files found — skipping sentiment plot.")
        _save_placeholder("04_sentiment_distribution.png",
                          "Sentiment: no raw text found")
        return {"note": "no raw text available"}

    # Score each document
    print(f"  Scoring {len(raw_df)} documents with VADER...")
    texts  = raw_df["text"].to_list()
    dates  = raw_df["date"].to_list()
    srcs   = raw_df["source"].to_list()
    scored = []
    for text, date, src in zip(texts, dates, srcs):
        ss = sia.polarity_scores(str(text))
        scored.append({
            "text":     text,
            "date":     str(date)[:10],
            "source":   src,
            "compound": ss["compound"],
            "pos":      ss["pos"],
            "neg":      ss["neg"],
            "neu":      ss["neu"],
        })
    sent_df = pl.DataFrame(scored)

    # Join sentiment back to canonical topics via the topics table
    # Strategy: for each source post, find the canonical topic with the highest
    # phrase overlap. We approximate by matching source + week.
    # Simpler approach that works without phrase-level join: classify by source
    # and week, then look up which canonical topics were active that week.
    signal_df = topics.filter(pl.col("source").is_in(SIGNAL_SOURCES))
    signal_df = signal_df.with_columns(
        pl.when(pl.col("canonical_topic").is_in(winners))
          .then(pl.lit("Winner"))
          .otherwise(pl.lit("Non-winner"))
          .alias("label")
    )

    # Aggregate: mean sentiment per (source, week, canonical_topic)
    # Then join sent_df's weekly averages onto topic labels
    sent_df_with_week = sent_df.with_columns(
        pl.col("date").map_elements(
            lambda d: _date_to_week(d), return_dtype=pl.Utf8
        ).alias("week")
    )

    src_week_sent = (
        sent_df_with_week.group_by(["source", "week"])
                         .agg(
                             pl.col("compound").mean().alias("mean_compound"),
                             pl.col("pos").mean().alias("mean_pos"),
                             pl.col("neg").mean().alias("mean_neg"),
                         )
    )

    topic_week_labels = (
        signal_df.group_by(["source", "week", "label"])
                 .agg(pl.len().alias("mention_count"))
    )

    joined = topic_week_labels.join(src_week_sent, on=["source", "week"], how="inner")

    if joined.is_empty():
        print("  Could not join sentiment to topic labels — check week formats.")
        _save_placeholder("04_sentiment_distribution.png",
                          "Sentiment: join failed")
        return {"note": "join failed"}

    winner_sent  = joined.filter(pl.col("label") == "Winner")["mean_compound"].to_list()
    loser_sent   = joined.filter(pl.col("label") == "Non-winner")["mean_compound"].to_list()

    #  Plot 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("EDA 4 — Sentiment Distribution: Winners vs. Non-winners",
                 fontsize=14, fontweight="bold")

    # KDE comparison
    ax = axes[0]
    if len(winner_sent) > 2:
        sns.kdeplot(winner_sent, ax=ax, color=PALETTE[1], fill=True, alpha=0.35,
                    label=f"Winners (n={len(winner_sent)})")
    if len(loser_sent) > 2:
        sns.kdeplot(loser_sent,  ax=ax, color=PALETTE[2], fill=True, alpha=0.35,
                    linestyle="--", label=f"Non-winners (n={len(loser_sent)})")
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("VADER compound score  (−1 = negative, +1 = positive)")
    ax.set_title("Compound sentiment distribution")
    ax.legend(fontsize=8)

    # Box-plot
    ax2 = axes[1]
    data_box  = winner_sent + loser_sent
    labels_box = ["Winner"] * len(winner_sent) + ["Non-winner"] * len(loser_sent)
    box_df = pl.DataFrame({"compound": data_box, "label": labels_box}).to_pandas()
    sns.boxplot(data=box_df, x="label", y="compound",
                palette={
                    "Winner":     PALETTE[1],
                    "Non-winner": PALETTE[2],
                },
                width=0.4, ax=ax2)
    ax2.set_xlabel("")
    ax2.set_ylabel("VADER compound score")
    ax2.set_title("Sentiment box-plot")

    # Mean pos/neg stacked bar
    ax3 = axes[2]
    labels_bar = ["Winners", "Non-winners"]
    pos_means  = [
        joined.filter(pl.col("label") == "Winner")["mean_pos"].mean() or 0,
        joined.filter(pl.col("label") == "Non-winner")["mean_pos"].mean() or 0,
    ]
    neg_means  = [
        joined.filter(pl.col("label") == "Winner")["mean_neg"].mean() or 0,
        joined.filter(pl.col("label") == "Non-winner")["mean_neg"].mean() or 0,
    ]
    x_bar = np.arange(len(labels_bar))
    ax3.bar(x_bar, pos_means, color=PALETTE[1], alpha=0.8, label="Positive")
    ax3.bar(x_bar, [-n for n in neg_means], color=PALETTE[2], alpha=0.8, label="Negative")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks(x_bar)
    ax3.set_xticklabels(labels_bar)
    ax3.set_ylabel("Mean VADER score")
    ax3.set_title("Avg pos/neg intensity")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "04_sentiment_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")

    # Stats
    mean_w = float(np.mean(winner_sent)) if winner_sent else None
    mean_l = float(np.mean(loser_sent))  if loser_sent  else None
    if winner_sent and loser_sent and len(winner_sent) > 1 and len(loser_sent) > 1:
        tstat, pval = stats.ttest_ind(winner_sent, loser_sent)
        print(f"  Mean sentiment — winners: {mean_w:.4f}, non-winners: {mean_l:.4f}")
        print(f"  T-test: t={tstat:.2f}, p={pval:.4f}")
        print(f"  Interpretation: sentiment {'IS' if pval < 0.05 else 'is NOT'} "
              f"significantly different between winners and non-winners (p<0.05)")
        return {
            "mean_sentiment_winner":     round(mean_w, 4) if mean_w else None,
            "mean_sentiment_non_winner": round(mean_l, 4) if mean_l else None,
            "ttest_p": round(pval, 4),
            "sentiment_significant":     pval < 0.05,
        }
    return {
        "mean_sentiment_winner":     mean_w,
        "mean_sentiment_non_winner": mean_l,
    }


# UTILITIES

def _date_to_week(date_str: str) -> str:
    try:
        d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        iso = d.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except Exception:
        return ""


def _save_placeholder(filename: str, message: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=13, color="grey")
    ax.axis("off")
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  Placeholder saved -> {out}")


def main():
    topics = load_topics()
    nl_df  = load_newsletter_topics()
    print(f"  topics.csv:            {len(topics):,} rows")
    print(f"  newsletter_topics.csv: {len(nl_df):,} rows")
    print(f"  Columns in topics.csv: {topics.columns}")

    results = {}

    # Run each EDA section
    results["eda1"] = eda_topic_frequency(topics, nl_df)
    results["eda2"] = eda_lead_lag(topics, nl_df)
    results["eda3"] = eda_winner_loser_trajectories(topics, nl_df)
    results["eda4"] = eda_sentiment_distribution(topics, nl_df)



if __name__ == "__main__":
    main()
