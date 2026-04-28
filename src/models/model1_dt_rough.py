import os
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

PROCESSED_DIR = "/Users/sarinafernandez-grinshpun/CIS2450FinalProject/data/processed"

#load splits
train_df = pl.read_csv(os.path.join(PROCESSED_DIR, "../splits/train.csv"))
val_df = pl.read_csv(os.path.join(PROCESSED_DIR, "../splits/val.csv"))
test_df = pl.read_csv(os.path.join(PROCESSED_DIR, "../splits/test.csv"))

print(train_df.shape, val_df.shape, test_df.shape)
print(train_df.columns)

#TODO double check these are the right feature cols
FEAT_COLS = ["burst_score", "mentions", "nl_rate_8wk", "nl_rate_all",
             "weeks_since_nl", "is_novel", "past_nl_count"]

X_train = train_df.select(FEAT_COLS).to_pandas()
y_train = train_df["label"].to_numpy()
X_val = val_df.select(FEAT_COLS).to_pandas()
y_val = val_df["label"].to_numpy()
X_test = test_df.select(FEAT_COLS).to_pandas()
y_test = test_df["label"].to_numpy()

print(f"train pos rate: {y_train.mean():.4f}")
print(f"val pos rate: {y_val.mean():.4f}")

#tried without class weights first, recall was 0.0 lol
#dt = DecisionTreeClassifier(max_depth=4, random_state=42)
#dt.fit(X_train, y_train)
#print(accuracy_score(y_test, dt.predict(X_test)))
#^ accuracy looked great (96%) but it was just predicting all 0s

# need class weights
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
print(f"class weights: {cw}")

#tune max_depth, from the class exercises
#decision trees dont need scaling (scale invariant)
best_depth = 2
best_f1 = 0.0

for depth in [2, 3, 4, 5, 6, 8, None]:
    dt = DecisionTreeClassifier(
        max_depth=depth,
        class_weight='balanced',
        criterion='gini',
        random_state=42,
    )
    dt.fit(X_train, y_train)
    vp = dt.predict(X_val)
    vf1 = f1_score(y_val, vp)
    print(f"depth={str(depth):<5} f1={vf1:.4f}")
    if vf1 > best_f1:
        best_f1 = vf1
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

print(classification_report(y_test, dt_pred))
print(f"auc: {roc_auc_score(y_test, dt_prob):.4f}")

#confusion matrix
cm = confusion_matrix(y_test, dt_pred)
tn, fp, fn, tp = cm.ravel()
print(f"tp={tp} fp={fp} fn={fn} tn={tn}")

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f"dt confusion depth={best_depth}")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "dt_confusion.png"))
plt.show()

#visualize the tree
#want to see what feature it splits on first
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt_best, feature_names=FEAT_COLS, class_names=["no", "yes"],
          filled=True, max_depth=3, ax=ax, fontsize=8)
ax.set_title(f"dt depth={best_depth}")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "dt_tree.png"), dpi=120, bbox_inches='tight')
plt.show()

# feature importance
imp = pd.DataFrame({
    "feat": FEAT_COLS,
    "imp": dt_best.feature_importances_
}).sort_values("imp", ascending=False)
print(imp)

#ok nl_rate_8wk is dominating which makes sense from EDA
#burst_score is basically 0 importance... interesting

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp["feat"][::-1], imp["imp"][::-1])
ax.set_title("dt feature importance")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "dt_importance.png"))
plt.show()

#precision at k
#this is more useful than accuracy for our use case
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
test_weeks = sorted(test_pd["week"].unique())

p3, p5, p10, r3, r5, r10 = [], [], [], [], [], []
for week in test_weeks:
    mask = test_pd["week"].values == week
    wt = y_test[mask]
    wp = dt_prob[mask]
    if wt.sum() == 0:
        continue
    for k, pl_, rl_ in [(3, p3, r3), (5, p5, r5), (10, p10, r10)]:
        pv = p_at_k(wt, wp, k)
        rv = r_at_k(wt, wp, k)
        if pv is not None: pl_.append(pv)
        if rv is not None: rl_.append(rv)

print(f"P@3={np.mean(p3):.3f} P@5={np.mean(p5):.3f} P@10={np.mean(p10):.3f}")
print(f"R@3={np.mean(r3):.3f} R@5={np.mean(r5):.3f} R@10={np.mean(r10):.3f}")
print(f"baseline: {y_test.mean():.3f}")

#not great but better than random at least
#LR next
