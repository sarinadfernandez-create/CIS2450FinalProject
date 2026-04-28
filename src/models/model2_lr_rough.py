import os
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

PROCESSED_DIR = "/Users/sarinafernandez-grinshpun/CIS2450FinalProject/data/processed"

train_df = pl.read_csv(os.path.join(PROCESSED_DIR, "../splits/train.csv"))
val_df = pl.read_csv(os.path.join(PROCESSED_DIR, "../splits/val.csv"))
test_df = pl.read_csv(os.path.join(PROCESSED_DIR, "../splits/test.csv"))

FEAT_COLS = ["burst_score", "mentions", "nl_rate_8wk", "nl_rate_all",
             "weeks_since_nl", "is_novel", "past_nl_count"]

X_train = train_df.select(FEAT_COLS).to_pandas()
y_train = train_df["label"].to_numpy()
X_val = val_df.select(FEAT_COLS).to_pandas()
y_val = val_df["label"].to_numpy()
X_test = test_df.select(FEAT_COLS).to_pandas()
y_test = test_df["label"].to_numpy()

#LR needs scaling unlike DT
#fit on train only, dont touch val or test with fit
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

#forgot to do this the first time and got a convergence warning
#also the coefficients were all over the place without scaling

cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
print(f"weights: {cw}")

#tune C on val
#C = inverse of regularization, smaller = stronger L2
#sing L2 because nl_rate_8wk and nl_rate_all are correlated (saw in EDA heatmap)
#ridge handles that

best_C = 0.1
best_f1 = 0.0

for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    vp = lr.predict(X_val_sc)
    vf1 = f1_score(y_val, vp)
    print(f"C={C} f1={vf1:.4f}")
    if vf1 > best_f1:
        best_f1 = vf1
        best_C = C

print(f"best C: {best_C}")

#originally tried L1 (lasso) to see if it zeroed out burst_score
#it did, burst_score coef went to exactly 0.0 which is interesting
#but F1 was worse so sticking with L2
#lr_l1 = LogisticRegression(C=best_C, penalty='l1', solver='liblinear',class_weight='balanced', max_iter=1000, random_state=42)
#lr_l1.fit(X_train_sc, y_train)
#print(lr_l1.coef_)

lr_best = LogisticRegression(C=best_C, class_weight='balanced', max_iter=1000, random_state=42)
lr_best.fit(X_train_sc, y_train)

lr_pred = lr_best.predict(X_test_sc)
lr_prob = lr_best.predict_proba(X_test_sc)[:, 1]

print(classification_report(y_test, lr_pred))
print(f"auc: {roc_auc_score(y_test, lr_prob):.4f}")
print(f"auprc: {average_precision_score(y_test, lr_prob):.4f}")

cm = confusion_matrix(y_test, lr_pred)
tn, fp, fn, tp = cm.ravel()
print(f"tp={tp} fp={fp} fn={fn} tn={tn}")

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f"LR confusion C={best_C}")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "lr_confusion.png"))
plt.show()

#coefficients - main thing LR gives you
coef_df = pd.DataFrame({
    "feature": FEAT_COLS,
    "coef": lr_best.coef_[0]
}).sort_values("coef", ascending=False)
print(coef_df.to_string(index=False))

#weeks_since_nl is negative which makes sense
#higher weeks_since = less likely to be covered (editors moved on)
#nl_rate_8wk positive and biggest = confirms EDA

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['green' if c > 0 else 'red' for c in coef_df["coef"]]
ax.barh(coef_df["feature"][::-1], coef_df["coef"][::-1], color=colors[::-1])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title(f"LR coefs (C={best_C} L2)")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "lr_coefs.png"))
plt.show()

#calibration - should be decent for LR by default
from sklearn.calibration import CalibrationDisplay
fig, ax = plt.subplots(figsize=(6, 5))
CalibrationDisplay.from_predictions(y_test, lr_prob, n_bins=8, ax=ax, name="LR")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_title("LR calibration")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "lr_calibration.png"))
plt.show()

#precision@k
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
    wp = lr_prob[mask]
    if wt.sum() == 0:
        continue
    for k, pl_, rl_ in [(3, p3, r3), (5, p5, r5), (10, p10, r10)]:
        pv = p_at_k(wt, wp, k)
        rv = r_at_k(wt, wp, k)
        if pv is not None: pl_.append(pv)
        if rv is not None: rl_.append(rv)

print(f"P@3={np.mean(p3):.3f} P@5={np.mean(p5):.3f} P@10={np.mean(p10):.3f}")
print(f"R@3={np.mean(r3):.3f} R@5={np.mean(r5):.3f} R@10={np.mean(r10):.3f}")

#similar to DT, not amazing
#going to try RF next, should be better since it handles interactions
