import os
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

PROCESSED_DIR = "/Users/sarinafernandez-grinshpun/CIS2450FinalProject/data/processed"

topics_df = pl.read_csv(os.path.join(PROCESSED_DIR, "canonical_topics.csv"))
nl_df = pl.read_csv(os.path.join(PROCESSED_DIR, "newsletter_topic_map.csv"))
sig_df = pl.read_csv(os.path.join(PROCESSED_DIR, "signal_topic_map.csv"),
    infer_schema_length=10000, schema_overrides={"date": pl.Utf8})

print(topics_df.shape)
print(nl_df.shape)
print(sig_df.shape)

print(sig_df.head())
print(sig_df.columns)
print(nl_df.columns)

#ok so we have phrase, score, source, date, week, canonical_topic, similarity

SOURCE_TYPE_MAP = {
    "arxiv": "research",
    "semantic_scholar": "research",
    "github": "developer",
    "reddit": "discussion",
    "hackernews": "discussion",
}

sig_df = sig_df.with_columns(
    pl.col("source").replace_strict(SOURCE_TYPE_MAP, default="other").alias("source_type")
)

#weekly aggregation
sig_weekly = (
    sig_df
    .filter(pl.col("week").is_not_null() & (pl.col("week") != ""))
    .group_by(["canonical_topic", "week"])
    .agg(pl.len().alias("mentions"))
)

print(sig_weekly.shape)
print(sig_weekly.head())

nl_weekly = (
    nl_df
    .filter(pl.col("week").is_not_null() & (pl.col("week") != ""))
    .group_by(["canonical_topic", "week"])
    .agg(pl.len().alias("nl_mentions"))
    .with_columns(pl.lit(1).alias("in_newsletter"))
)

all_topics = sig_weekly["canonical_topic"].unique().to_list()
print(f"num topics: {len(all_topics)}")

#look at what topics we actually have
top = (
    sig_weekly
    .group_by("canonical_topic")
    .agg(pl.col("mentions").sum())
    .sort("mentions", descending=True)
    .head(15)
)
print(top)

#as expected LLM and openai stuff dominates
#raw counts are useless, need to normalize somehow
#idea: z-score against each topic's own rolling baseline
#o a topic that normally gets 2 mentions spiking to 30 is more interesting than LLM going from 500 to 510

def parse_week(w):
    try:
        return datetime.strptime(w + '-1', '%G-W%V-%u')
    except:
        return None

#build burst scores
#was doing .filter() inside the loop before and it was insanely slow
#wwitching to dict

topic_mentions_lookup = defaultdict(list)
for row in sig_weekly.sort("week").iter_rows(named=True):
    topic_mentions_lookup[row["canonical_topic"]].append((row["week"], row["mentions"]))

burst_rows = []
for topic, pairs in topic_mentions_lookup.items():
    if len(pairs) < 4:
        continue
    weeks = [p[0] for p in pairs]
    mentions = np.array([p[1] for p in pairs], dtype=float)
    for i in range(len(weeks)):
        past = mentions[max(0, i-8):i]
        if len(past) >= 2:
            m = past.mean()
            s = past.std()
        else:
            m = mentions[i]
            s = 1.0
        burst = (mentions[i] - m) / (s + 1)
        burst_rows.append({
            "canonical_topic": topic,
            "week": weeks[i],
            "mentions": int(mentions[i]),
            "burst_score": round(burst, 3),
        })

burst_df = pl.DataFrame(burst_rows)
print(burst_df.shape)
print(burst_df.head())

#distribution
plt.hist(burst_df["burst_score"].to_numpy(), bins=50)
plt.title("burst scores")
plt.show()

#there are some insane outliers
print(burst_df["burst_score"].quantile(0.99))
print(burst_df["burst_score"].max())

#cap at 99th percentile
p99 = float(burst_df["burst_score"].quantile(0.99))
m99 = float(burst_df["mentions"].quantile(0.99))
burst_df = burst_df.with_columns([
    pl.col("burst_score").clip(upper_bound=p99),
    pl.col("mentions").clip(upper_bound=int(m99)),
])

#forward labels
#IMPORTANT: label = newsletter in NEXT 2 weeks not same week
#same week was leakage - the signal and newsletter coverage happen simultaneously
#so we cant use that to predict anything

def week_offset(w, offset):
    d = parse_week(w)
    if d is None:
        return ""
    d2 = d + timedelta(weeks=offset)
    iso = d2.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"

nl_week_set = set(zip(
    nl_weekly["canonical_topic"].to_list(),
    nl_weekly["week"].to_list()
))

def forward_label(topic, week, window=2):
    for offset in range(1, window+1):
        if (topic, week_offset(week, offset)) in nl_week_set:
            return 1
    return 0

labels = [forward_label(r["canonical_topic"], r["week"]) for r in burst_df.iter_rows(named=True)]
burst_df = burst_df.with_columns(pl.Series("label", labels))

print(f"positive rate: {burst_df['label'].mean():.4f}")
# ok about 4%

#historical features
#how often has this topic been in newsletter before
#adds this as feature bc editors might have preferences

nl_lookup = {}
for row in nl_weekly.iter_rows(named=True):
    nl_lookup.setdefault(row["canonical_topic"], [])
    nl_lookup[row["canonical_topic"]].append(row["week"])
for t in nl_lookup:
    nl_lookup[t] = sorted(nl_lookup[t])

active_lookup = {}
for row in sig_weekly.iter_rows(named=True):
    active_lookup.setdefault(row["canonical_topic"], [])
    active_lookup[row["canonical_topic"]].append(row["week"])
for t in active_lookup:
    active_lookup[t] = sorted(active_lookup[t])

hist_rows = []
for topic in all_topics:
    active_weeks = active_lookup.get(topic, [])
    nl_weeks = nl_lookup.get(topic, [])
    for i, week in enumerate(active_weeks):
        past_active = active_weeks[:i]
        past_nl = [w for w in nl_weeks if w < week]
        rate_all = len(past_nl) / len(past_active) if past_active else 0.0
        past_8 = past_active[-8:]
        past_nl_8 = [w for w in past_nl if w in set(past_8)]
        rate_8 = len(past_nl_8) / len(past_8) if past_8 else 0.0
        if past_nl:
            last = parse_week(past_nl[-1])
            cur = parse_week(week)
            since = (cur - last).days // 7 if (last and cur) else 999
        else:
            since = 999
        hist_rows.append({
            "canonical_topic": topic,
            "week": week,
            "nl_rate_8wk": round(rate_8, 4),
            "nl_rate_all": round(rate_all, 4),
            "weeks_since_nl": min(since, 200),
            "is_novel": int(len(past_nl) == 0),
            "past_nl_count": len(past_nl),
        })

hist_df = pl.DataFrame(hist_rows)
print(hist_df.head())

modeling_df = burst_df.join(hist_df, on=["canonical_topic", "week"], how="left").fill_null(0)
print(modeling_df.shape)

#check what actually correlates with label before doing any modeling
import scipy.stats as stats

print("\ncorrelations with label:")
for feat in ["burst_score", "mentions", "nl_rate_8wk", "nl_rate_all", "weeks_since_nl"]:
    arr = modeling_df[feat].to_numpy()
    lbl = modeling_df["label"].to_numpy()
    r, p = stats.pointbiserialr(arr, lbl)
    print(f"  {feat}: r={r:.4f} p={p:.4f}")

#ok nl_rate_8wk is WAY stronger than burst_score
#like r=0.22 vs r=0.02 - burst_score is basically nothing
#honestly kind of surprising, thought signal activity would be more predictive
#but makes sense - editors just cover what they always cover
#weeks_since_nl is negative which makes sense too

#temporal split - 70/15/15
#cannot shuffle, that would be cheating
sorted_weeks = sorted(modeling_df["week"].unique().to_list())
n = len(sorted_weeks)
t1 = sorted_weeks[int(n * 0.70)]
t2 = sorted_weeks[int(n * 0.85)]

print(f"\ntrain cutoff: {t1}")
print(f"val cutoff: {t2}")

train_df = modeling_df.filter(pl.col("week") < t1)
val_df = modeling_df.filter((pl.col("week") >= t1) & (pl.col("week") < t2))
test_df = modeling_df.filter(pl.col("week") >= t2)

print(train_df.shape, val_df.shape, test_df.shape)
print(f"train pos: {train_df['label'].sum()} val pos: {val_df['label'].sum()} test pos: {test_df['label'].sum()}")

#save splits so i dont have to recompute every time
splits_dir = os.path.join(os.path.dirname(PROCESSED_DIR), "splits")
os.makedirs(splits_dir, exist_ok=True)
modeling_df.write_csv(os.path.join(PROCESSED_DIR, "topic_week_features.csv"))
train_df.write_csv(os.path.join(splits_dir, "train.csv"))
val_df.write_csv(os.path.join(splits_dir, "val.csv"))
test_df.write_csv(os.path.join(splits_dir, "test.csv"))
print("saved splits")

#quick baseline-just try decision tree to see if anything works at all
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

FEAT_COLS = ["burst_score", "mentions", "nl_rate_8wk", "nl_rate_all",
             "weeks_since_nl", "is_novel", "past_nl_count"]

X_train = train_df.select(FEAT_COLS).to_pandas()
y_train = train_df["label"].to_numpy()
X_val = val_df.select(FEAT_COLS).to_pandas()
y_val = val_df["label"].to_numpy()
X_test = test_df.select(FEAT_COLS).to_pandas()
y_test = test_df["label"].to_numpy()

#first try without class weights
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
print(f"accuracy: {accuracy_score(y_test, dt.predict(X_test)):.4f}")
print(f"f1: {f1_score(y_test, dt.predict(X_test)):.4f}")

#accuracy looks fine but f1 is terrible
#oh right, class imbalance,its just predicting 0 for everything
#need class_weight='balanced'

cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
print(f"class weights: {cw}")

dt2 = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt2.fit(X_train, y_train)
print(f"f1 with weights: {f1_score(y_val, dt2.predict(X_val)):.4f}")
#better

#ok this is working enough to build on
#going to make three separate files so i can work each model out individually
#model1_dt.py, model2_lr.py, model3_rf.py
