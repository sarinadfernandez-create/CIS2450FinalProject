import os
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime, timedelta
import re
from functools import lru_cache
import scipy.stats as stats

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    roc_curve, confusion_matrix, average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight
import sklearn.metrics

PROCESSED_DIR = "/Users/sarinafernandez-grinshpun/CIS2450FinalProject/data/processed"

SOURCE_TYPE_MAP = {
    "arxiv": "research",
    "semantic_scholar": "research",
    "github": "developer",
    "reddit": "discussion",
    "hackernews": "discussion",
}

topics_df = pl.read_csv(os.path.join(PROCESSED_DIR, "canonical_topics.csv"))
nl_df = pl.read_csv(os.path.join(PROCESSED_DIR, "newsletter_topic_map.csv"))
sig_df = pl.read_csv(
    os.path.join(PROCESSED_DIR, "signal_topic_map.csv"),
    schema_overrides={
        "date": pl.Utf8,
        "week": pl.Utf8,
        "phrase": pl.Utf8,
        "source": pl.Utf8,
        "canonical_topic": pl.Utf8,
        "score": pl.Float64,
        "similarity": pl.Float64,
    }
)
sig_df = sig_df.with_columns(
    pl.col("source").replace_strict(SOURCE_TYPE_MAP, default="other").alias("source_type")
)

print(sig_df.shape)
print(nl_df.shape)

WEEK_PATTERN = re.compile(r'^\d{4}-W\d{2}$')

@lru_cache(maxsize=512)
def parse_week(w):
    try:
        return datetime.strptime(w + '-1', '%G-W%V-%u')
    except:
        return None

def week_offset(week_str, offset):
    d = parse_week(week_str)
    if d is None:
        return ""
    d2 = d + timedelta(weeks=offset)
    iso = d2.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"

sig_weekly_total = (
    sig_df
    .filter(pl.col("week").is_not_null() & (pl.col("week") != ""))
    .group_by(["canonical_topic", "week"])
    .agg(pl.len().alias("mentions"))
)

sig_weekly = (
    sig_df
    .filter(pl.col("week").is_not_null() & (pl.col("week") != ""))
    .group_by(["canonical_topic", "week", "source_type"])
    .agg(pl.len().alias("mentions"))
)

nl_weekly = (
    nl_df
    .filter(pl.col("week").is_not_null() & (pl.col("week") != ""))
    .group_by(["canonical_topic", "week"])
    .agg(pl.len().alias("nl_mentions"))
    .with_columns(pl.lit(1).alias("in_newsletter"))
)

all_topics_list = sig_weekly_total["canonical_topic"].unique().to_list()
all_weeks = sorted(set(
    sig_weekly_total["week"].unique().to_list() +
    nl_weekly["week"].unique().to_list()
))
all_weeks = [w for w in all_weeks if WEEK_PATTERN.match(w)]

print(f"topics: {len(all_topics_list)} weeks: {len(all_weeks)}")

#burst scores-dict lookup is way faster than filtering polars each iteration
topic_week_mentions = defaultdict(list)
for row in sig_weekly_total.sort("week").iter_rows(named=True):
    topic_week_mentions[row["canonical_topic"]].append((row["week"], row["mentions"]))

burst_rows = []
for topic, pairs in topic_week_mentions.items():
    if len(pairs) < 4:
        continue
    weeks = [p[0] for p in pairs]
    mentions = np.array([p[1] for p in pairs], dtype=float)
    for i in range(len(weeks)):
        past = mentions[max(0, i-8):i]
        mean_val = past.mean() if len(past) >= 2 else mentions[i]
        std_val = past.std() if len(past) >= 2 else 1.0
        burst = (mentions[i] - mean_val) / (std_val + 1)
        burst_rows.append({
            "canonical_topic": topic,
            "week": weeks[i],
            "mentions": int(mentions[i]),
            "rolling_mean": round(mean_val, 2),
            "burst_score": round(burst, 3),
        })

burst_df = pl.DataFrame(burst_rows)
print(f"burst rows: {burst_df.height}")

#cap outliers at 99th percentile
burst_99 = float(burst_df["burst_score"].quantile(0.99))
mentions_99 = float(burst_df["mentions"].quantile(0.99))
print(f"capping burst at {burst_99:.2f}, mentions at {mentions_99:.0f}")

burst_df = burst_df.with_columns([
    pl.col("burst_score").clip(upper_bound=burst_99),
    pl.col("mentions").clip(upper_bound=int(mentions_99)),
])

#historical features
#BUG FIX: was storing len(past_nl) as the rate instead of dividing
#so rate_alltime was like 4.0, 3.0 instead of 0-1
#fixed below-divide by len(past_active)
nl_topic_week_lookup = {}
for row in nl_weekly.iter_rows(named=True):
    nl_topic_week_lookup.setdefault(row["canonical_topic"], [])
    nl_topic_week_lookup[row["canonical_topic"]].append(row["week"])
for t in nl_topic_week_lookup:
    nl_topic_week_lookup[t] = sorted(nl_topic_week_lookup[t])

active_week_lookup = {}
for row in sig_weekly_total.iter_rows(named=True):
    active_week_lookup.setdefault(row["canonical_topic"], [])
    active_week_lookup[row["canonical_topic"]].append(row["week"])
for t in active_week_lookup:
    active_week_lookup[t] = sorted(active_week_lookup[t])

hist_rows = []
for topic in all_topics_list:
    active_weeks_sorted = active_week_lookup.get(topic, [])
    nl_weeks_sorted = nl_topic_week_lookup.get(topic, [])
    for i, week in enumerate(active_weeks_sorted):
        past_active = active_weeks_sorted[:i]
        past_nl = [w for w in nl_weeks_sorted if w < week]
        #fixed: was len(past_nl) before which is wrong
        rate_alltime = len(past_nl) / len(past_active) if past_active else 0.0
        past_active_8 = past_active[-8:]
        past_nl_8 = [w for w in past_nl if w in set(past_active_8)]
        #fixed same thing here
        rate_8wk = len(past_nl_8) / len(past_active_8) if past_active_8 else 0.0
        if past_nl:
            last_nl = parse_week(past_nl[-1])
            current = parse_week(week)
            weeks_since = (current - last_nl).days // 7 if (last_nl and current) else 999
        else:
            weeks_since = 999
        hist_rows.append({
            "canonical_topic": topic,
            "week": week,
            "topic_nl_rate_8wk": round(rate_8wk, 4),
            "topic_nl_rate_alltime": round(rate_alltime, 4),
            "weeks_since_last_nl": min(weeks_since, 200),
            "is_novel_topic": int(len(past_nl) == 0),
            "past_nl_count": len(past_nl),
        })

hist_df = pl.DataFrame(hist_rows)
print(f"hist rows: {hist_df.height}")
#sanity check
print(f"max alltime rate: {hist_df['topic_nl_rate_alltime'].max():.4f} (must be <= 1 now)")

#forward labels
nl_week_set = set(zip(
    nl_weekly["canonical_topic"].to_list(),
    nl_weekly["week"].to_list(),
))

def forward_label(topic, week, window=2):
    #label = 1 if topic in newsletter next 2 weeks not same week
    for offset in range(1, window + 1):
        if (topic, week_offset(week, offset)) in nl_week_set:
            return 1
    return 0

labels = [forward_label(r["canonical_topic"], r["week"]) for r in burst_df.iter_rows(named=True)]
burst_labeled = burst_df.with_columns(pl.Series("label_next2wk", labels))

modeling_df = (
    burst_labeled
    .join(hist_df, on=["canonical_topic", "week"], how="left")
    .fill_null(0)
)

total = modeling_df.height
positive = modeling_df["label_next2wk"].sum()
print(f"total: {total} positive: {positive} rate: {positive/total:.4f}")
print(f"class ratio 1:{(total-positive)//positive}")

#temporal split 70/15/15 - no shuffling
sorted_weeks = sorted(modeling_df["week"].unique().to_list())
n = len(sorted_weeks)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train_cutoff = sorted_weeks[train_end]
val_cutoff = sorted_weeks[val_end]

print(f"train: up to {train_cutoff}")
print(f"val: {train_cutoff} to {val_cutoff}")
print(f"test: {val_cutoff} onward")

train_df = modeling_df.filter(pl.col("week") < train_cutoff)
val_df = modeling_df.filter((pl.col("week") >= train_cutoff) & (pl.col("week") < val_cutoff))
test_df = modeling_df.filter(pl.col("week") >= val_cutoff)

print(train_df.height, val_df.height, test_df.height)
print(f"pos: {train_df['label_next2wk'].sum()} {val_df['label_next2wk'].sum()} {test_df['label_next2wk'].sum()}")

FEATURE_COLS = [
    "burst_score",
    "mentions",
    "topic_nl_rate_8wk",
    "topic_nl_rate_alltime",
    "weeks_since_last_nl",
    "is_novel_topic",
    "past_nl_count",
]
TARGET_COL = "label_next2wk"

X_train = train_df.select(FEATURE_COLS).to_pandas()
y_train = train_df[TARGET_COL].to_numpy()
X_val = val_df.select(FEATURE_COLS).to_pandas()
y_val = val_df[TARGET_COL].to_numpy()
X_test = test_df.select(FEATURE_COLS).to_pandas()
y_test = test_df[TARGET_COL].to_numpy()

#scale for LR only-trees don't need it
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
print(f"class weights: {class_weights}")

#MODEL 1: decision tree
#tune max_depth on val not test
print("tuning DT...")
best_depth = 2
best_val_f1 = 0.0

for depth in [2, 3, 4, 5, 6, 8, None]:
    dt = DecisionTreeClassifier(
        max_depth=depth,
        class_weight='balanced',
        criterion='gini',
        random_state=42,
    )
    dt.fit(X_train, y_train)
    val_pred = dt.predict(X_val)
    val_f1 = f1_score(y_val, val_pred)
    print(f"depth={str(depth):<5} val_f1={val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_depth = depth

print(f"best depth: {best_depth}")

dt_best = DecisionTreeClassifier(
    max_depth=best_depth,
    class_weight='balanced',
    criterion='gini',
    random_state=42,
)
dt_best.fit(X_train, y_train)
dt_pred = dt_best.predict(X_test)
dt_prob = dt_best.predict_proba(X_test)[:, 1]

print("DT test results:")
print(classification_report(y_test, dt_pred, target_names=["not NL", "in NL"]))
print(f"auc: {roc_auc_score(y_test, dt_prob):.4f}")
print(f"auprc: {average_precision_score(y_test, dt_prob):.4f}")

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt_best, feature_names=FEATURE_COLS, class_names=["no", "yes"],
          filled=True, max_depth=3, ax=ax, fontsize=9)
ax.set_title(f"DT depth={best_depth}")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "dt_tree.png"), dpi=150, bbox_inches='tight')
plt.show()

#MODEL 2: logistic regression
#tune C on val
print("tuning LR...")
best_C = 0.1
best_val_f1_lr = 0.0

for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    val_pred = lr.predict(X_val_sc)
    val_f1 = f1_score(y_val, val_pred)
    print(f"C={C} val_f1={val_f1:.4f}")
    if val_f1 > best_val_f1_lr:
        best_val_f1_lr = val_f1
        best_C = C

print(f"best C: {best_C}")

lr_best = LogisticRegression(C=best_C, class_weight='balanced', max_iter=1000, random_state=42)
lr_best.fit(X_train_sc, y_train)
lr_pred = lr_best.predict(X_test_sc)
lr_prob = lr_best.predict_proba(X_test_sc)[:, 1]

print("LR test results:")
print(classification_report(y_test, lr_pred, target_names=["not NL", "in NL"]))
print(f"auc: {roc_auc_score(y_test, lr_prob):.4f}")
print(f"auprc: {average_precision_score(y_test, lr_prob):.4f}")

coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": lr_best.coef_[0],
}).sort_values("coefficient", ascending=False)
print(coef_df.to_string(index=False))
#after the rate fix is_novel_topic should not be dominating anymore

#MODEL 3: random forest - primary model
print("tuning RF...")
best_rf = None
best_val_f1_rf = 0.0
best_rf_params = {}

for n_est in [100, 200, 300]:
    for depth in [4, 6, 8]:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        val_pred = rf.predict(X_val)
        val_f1 = f1_score(y_val, val_pred)
        print(f"n={n_est} d={depth} val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1_rf:
            best_val_f1_rf = val_f1
            best_rf = rf
            best_rf_params = {"n_estimators": n_est, "max_depth": depth}

print(f"best RF: {best_rf_params}")

rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]

print("RF test results:")
print(classification_report(y_test, rf_pred, target_names=["not NL", "in NL"]))
print(f"auc: {roc_auc_score(y_test, rf_prob):.4f}")
print(f"auprc: {average_precision_score(y_test, rf_prob):.4f}")

rf_importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": best_rf.feature_importances_,
}).sort_values("importance", ascending=False)
print(rf_importance.to_string(index=False))

#comparison
print("\nfinal comparison:")
for name, prob, pred in [("DT", dt_prob, dt_pred), ("LR", lr_prob, lr_pred), ("RF", rf_prob, rf_pred)]:
    auc = roc_auc_score(y_test, prob)
    ap = average_precision_score(y_test, prob)
    f1 = f1_score(y_test, pred)
    print(f"{name}: auc={auc:.4f} auprc={ap:.4f} f1={f1:.4f}")

#precision@k recall@k
def p_at_k(ytrue, yprob, k):
    if len(yprob) < k:
        return None
    idx = np.argsort(yprob)[::-1][:k]
    return ytrue[idx].mean()

def r_at_k(ytrue, yprob, k):
    if ytrue.sum() == 0:
        return None
    idx = np.argsort(yprob)[::-1][:k]
    return ytrue[idx].sum() / ytrue.sum()

test_pd = test_df.to_pandas()
test_weeks = sorted(test_pd["week"].unique().tolist())

for name, prob in [("DT", dt_prob), ("LR", lr_prob), ("RF", rf_prob)]:
    p3, p5, p10, r3, r5, r10 = [], [], [], [], [], []
    for week in test_weeks:
        mask = test_pd["week"].values == week
        wt = y_test[mask]
        wp = prob[mask]
        if wt.sum() == 0:
            continue
        for k, pl_, rl_ in [(3, p3, r3), (5, p5, r5), (10, p10, r10)]:
            pv = p_at_k(wt, wp, k)
            rv = r_at_k(wt, wp, k)
            if pv is not None: pl_.append(pv)
            if rv is not None: rl_.append(rv)
    print(f"{name} P@3={np.mean(p3):.3f} P@5={np.mean(p5):.3f} R@3={np.mean(r3):.3f} R@5={np.mean(r5):.3f}")

#lead time
test_pd["rf_prob"] = rf_prob
THRESHOLD = 0.3
lead_times = []
nl_hits = test_pd[test_pd["label_next2wk"] == 1][["canonical_topic", "week"]].drop_duplicates()

for _, row in nl_hits.iterrows():
    topic = row["canonical_topic"]
    nl_week = row["week"]
    nl_date = parse_week(nl_week)
    if nl_date is None:
        continue
    candidates = test_pd[
        (test_pd["canonical_topic"] == topic) &
        (test_pd["week"] < nl_week) &
        (test_pd["rf_prob"] >= THRESHOLD)
    ].sort_values("week")
    if len(candidates) == 0:
        continue
    first_date = parse_week(candidates.iloc[0]["week"])
    if first_date:
        lead = (nl_date - first_date).days // 7
        if lead > 0:
            lead_times.append(lead)

if lead_times:
    print(f"lead time: mean={np.mean(lead_times):.1f} median={np.median(lead_times):.1f} max={max(lead_times)}")
else:
    print("no lead times found")

#novel vs established
print("\nnnovel vs established:")
for label, mv in [("novel", 1), ("established", 0)]:
    mask = test_pd["is_novel_topic"].values == mv
    ys = y_test[mask]
    ps = rf_prob[mask]
    if ys.sum() == 0:
        print(f"{label}: no positives in test set")
        continue
    auc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float('nan')
    ap = average_precision_score(ys, ps)
    r, p = stats.pointbiserialr(test_pd.loc[mask, "burst_score"].values, ys)
    print(f"{label}: auc={auc:.3f} auprc={ap:.3f} burst_r={r:.3f} p={p:.3f}")

#save splits
splits_dir = os.path.join(os.path.dirname(PROCESSED_DIR), "splits")
os.makedirs(splits_dir, exist_ok=True)
modeling_df.write_csv(os.path.join(PROCESSED_DIR, "topic_week_features.csv"))
train_df.write_csv(os.path.join(splits_dir, "train.csv"))
val_df.write_csv(os.path.join(splits_dir, "val.csv"))
test_df.write_csv(os.path.join(splits_dir, "test.csv"))
print("saved")
