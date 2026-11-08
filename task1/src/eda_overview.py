# EDA pass 1: shapes, missingness, per-feature AUC/Spearman, outliers, train-vs-test KS. Run from repo root.
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score
from common import paths
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
D = paths.task_data(1)
tr = pd.read_csv(D/"train.csv"); te = pd.read_csv(D/"test.csv")
print("train", tr.shape, "test", te.shape)
print("dtypes:", tr.dtypes.value_counts().to_dict())
print("train cols not in test:", set(tr.columns)-set(te.columns))
y = tr["is_cheater"]
print("target:", y.value_counts(dropna=False).to_dict(), "rate=%.4f" % y.mean())
feats = [c for c in te.columns if c != "player_id"]
print("dup player_id train/test overlap:", tr.player_id.duplicated().sum(), len(set(tr.player_id)&set(te.player_id)))
print("dup feature rows train:", tr[feats].duplicated().sum())
mtr, mte = tr[feats].isna().mean(), te[feats].isna().mean()
print("rows w/ any NaN train %.3f test %.3f" % (tr[feats].isna().any(axis=1).mean(), te[feats].isna().any(axis=1).mean()))
print("NaN count per row (train):", tr[feats].isna().sum(axis=1).value_counts().sort_index().to_dict())
rows = []
for c in feats:
    x = tr[c]; m = x.notna() & y.notna()
    auc = roc_auc_score(y[m], x[m]) if m.sum() else np.nan
    rho = spearmanr(x[m], y[m]).statistic
    miss_rate_pos = y[x.isna()].mean()
    q = x.quantile([0, .01, .5, .99, 1]).values
    iqr = x.quantile(.75)-x.quantile(.25)
    out = ((x < x.quantile(.25)-3*iqr) | (x > x.quantile(.75)+3*iqr)).mean()
    ks = ks_2samp(x.dropna(), te[c].dropna()).statistic
    rows.append(dict(f=c, miss_tr=mtr[c], miss_te=mte[c], y_if_nan=miss_rate_pos, nuniq=x.nunique(),
        mn=q[0], p1=q[1], med=q[2], p99=q[3], mx=q[4], neg=(x<0).mean(), out3iqr=out,
        auc=max(auc, 1-auc), dir="+" if auc>=.5 else "-", spearman=rho, ks=ks,
        mean0=x[y==0].median(), mean1=x[y==1].median()))
r = pd.DataFrame(rows).set_index("f")
print(r.sort_values("auc", ascending=False).round(4).to_string())
print("\nnon-integer check on count-like:", {c: (tr[c].dropna()%1!=0).mean().round(3) for c in ["level","reports_received","device_changes_count","account_age_days","friend_network_size"]})
print("pct cols >100:", {c: (tr[c]>100).mean() for c in ["headshot_percentage","win_rate","accuracy_score","first_blood_rate","clutch_success_rate"]})
print("pct cols range check:", {c:(tr[c].min(),tr[c].max()) for c in ["night_play_ratio","headshot_percentage","win_rate"]})
# correlation among features
cm = tr[feats].corr(method="spearman").abs()
pairs = cm.where(np.triu(np.ones(cm.shape),1).astype(bool)).stack().sort_values(ascending=False)
print("\ntop feature pairs |spearman|:\n", pairs.head(12).round(3).to_string())
# missing target rows?
print("target NaN rows:", y.isna().sum())
