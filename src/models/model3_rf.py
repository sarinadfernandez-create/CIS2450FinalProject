import os
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibrationDisplay
import seaborn as sns
import scipy.stats as stats

PROCESSED_DIR = "/Users/sarinafernandez-grinshpun/CIS2450FinalProject/data/processed"

train_df = pl.read_csv(os.path.join(PROCESSED_DIR, "../splits/train.csv"))
val_df = pl.read_csv(os.path.join(PROCESSED_DIR, "../splits/val.csv"))
test_df = pl.read_csv(os.path.join(PROCESSED_DIR, "../splits/test.csv"))

print(f"train: {train_df.shape} | val: {val_df.shape} | test: {test_df.shape}")
print(f"train positives: {train_df['label'].sum()}")

FEAT_COLS = ["burst_score", "mentions", "nl_rate_8wk", "nl_rate_all",
             "weeks_since_nl", "is_novel", "past_nl_count"]

X_train = train_df.select(FEAT_COLS).to_pandas()
y_train = train_df["label"].to_numpy()
X_val = val_df.select(FEAT_COLS).to_pandas()
y_val = val_df["label"].to_numpy()
X_test = test_df.select(FEAT_COLS).to_pandas()
y_test = test_df["label"].to_numpy()

#RF doesn't need scaling - same as DT, scale invariant
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
print(f"class weights: {cw}")

#tune n_estimators and max_depth on val set
#RF hyperparams from class: n_estimators, max_depth, feature set size
print("tuning RF on val set...")
best_rf = None
best_f1 = 0.0
best_params = {}

for n in [100, 200, 300]:
    for d in [4, 6, 8]:
        rf = RandomForestClassifier(
            n_estimators=n,
            max_depth=d,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,  #parallelize,RF is parallelizable
        )
        rf.fit(X_train, y_train)
        vp = rf.predict(X_val)
        vf1 = f1_score(y_val, vp)
        vauc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
        print(f"  n={n} d={d} | f1={vf1:.4f} auc={vauc:.4f}")
        if vf1 > best_f1:
            best_f1 = vf1
            best_rf = rf
            best_params = {"n_estimators": n, "max_depth": d}

print(f"\nbest: {best_params} (val f1={best_f1:.4f})")

# final eval on test set
rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]

print("\n--- test set ---")
print(classification_report(y_test, rf_pred, target_names=["no", "yes"]))
print(f"auc: {roc_auc_score(y_test, rf_prob):.4f}")
print(f"auprc: {average_precision_score(y_test, rf_prob):.4f}")

cm = confusion_matrix(y_test, rf_pred)
tn, fp, fn, tp = cm.ravel()
print(f"tp={tp} fp={fp} fn={fn} tn={tn}")
#fp = hype that editors didn't pick up
#fn = signals we missed entirely

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=["pred no", "pred yes"],
            yticklabels=["actual no", "actual yes"])
ax.set_title(f"RF confusion (n={best_params['n_estimators']} d={best_params['max_depth']})")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "rf_confusion.png"))
plt.show()

#feature importance, gini importance across all trees
imp = pd.DataFrame({
    "feat": FEAT_COLS,
    "imp": best_rf.feature_importances_
}).sort_values("imp", ascending=False)
print("\nRF feature importance:")
print(imp.to_string(index=False))
#expect nl_rate_8wk and weeks_since_nl to dominate

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp["feat"][::-1], imp["imp"][::-1], color='#1D9E75')
ax.set_xlabel("mean gini decrease (across all trees)")
ax.set_title("RF feature importance")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "rf_importance.png"))
plt.show()

#calibration
fig, ax = plt.subplots(figsize=(6, 5))
CalibrationDisplay.from_predictions(y_test, rf_prob, n_bins=8, ax=ax, name="RF")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_title("RF calibration curve")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "rf_calibration.png"))
plt.show()

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
test_pd["rf_prob"] = rf_prob
test_weeks = sorted(test_pd["week"].unique())

p3, p5, p10, r3, r5, r10 = [], [], [], [], [], []
for week in test_weeks:
    mask = test_pd["week"].values == week
    wt = y_test[mask]
    wp = rf_prob[mask]
    if wt.sum() == 0:
        continue
    for k, pl_, rl_ in [(3, p3, r3), (5, p5, r5), (10, p10, r10)]:
        pv = p_at_k(wt, wp, k)
        rv = r_at_k(wt, wp, k)
        if pv is not None: pl_.append(pv)
        if rv is not None: rl_.append(rv)

print(f"\nRF Precision@K: P@3={np.mean(p3):.3f} P@5={np.mean(p5):.3f} P@10={np.mean(p10):.3f}")
print(f"RF Recall@K:    R@3={np.mean(r3):.3f} R@5={np.mean(r5):.3f} R@10={np.mean(r10):.3f}")

#lead time, how early do we detect topics before editors cover them
def parse_week(w):
    try:
        return datetime.strptime(w + '-1', '%G-W%V-%u')
    except:
        return None

def week_offset(w, offset):
    d = parse_week(w)
    if d is None:
        return ""
    from datetime import timedelta
    d2 = d + timedelta(weeks=offset)
    iso = d2.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"

THRESHOLD = 0.3
lead_times = []
nl_hits = test_pd[test_pd["label"] == 1][["canonical_topic", "week"]].drop_duplicates()

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
    print(f"\nlead time analysis (threshold={THRESHOLD}):")
    print(f"  topics detected early: {len(lead_times)}")
    print(f"  mean lead time: {np.mean(lead_times):.1f} weeks")
    print(f"  median: {np.median(lead_times):.1f} weeks")
    print(f"  max: {max(lead_times)} weeks")
else:
    print("no early detections found - might need to lower threshold")

#novel vs established thesis test
print("\n--- does burst matter more for novel topics? ---")
for label, mv in [("novel (never in NL)", 1), ("established (in NL before)", 0)]:
    mask = test_pd["is_novel"].values == mv
    ys = y_test[mask]
    ps = rf_prob[mask]
    if ys.sum() == 0:
        print(f"{label}: no positives in test set")
        continue
    auc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float('nan')
    ap = average_precision_score(ys, ps)
    r, p = stats.pointbiserialr(test_pd.loc[mask, "burst_score"].values, ys)
    print(f"{label}")
    print(f"  n={mask.sum()} pos={ys.sum()} ({ys.mean()*100:.1f}%)")
    print(f"  auc={auc:.3f} auprc={ap:.3f}")
    print(f"  burst correlation: r={r:.3f} p={p:.3f}")
