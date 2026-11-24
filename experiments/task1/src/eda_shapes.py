# EDA pass 2: decile target rates, boundary values, MI, unlabeled rows, derived-feature checks. Run from repo root.
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from common import paths
pd.set_option("display.width", 250)
D = paths.task_data(1)
tr = pd.read_csv(D/"train.csv"); te = pd.read_csv(D/"test.csv")
lab = tr[tr.is_cheater.notna()].copy(); y = lab.is_cheater.astype(int)
feats = [c for c in te.columns if c != "player_id"]
base = y.mean()
rows = []
for c in feats:
    x = lab[c]; m = x.notna()
    b = pd.qcut(x[m], 10, duplicates="drop")
    rate = y[m].groupby(b, observed=True).mean()
    mn, mx = x.min(), x.max()
    at_min, at_max = (x == mn), (x == mx)
    rows.append(dict(f=c, dec_lo=rate.iloc[0], dec_hi=rate.iloc[-1], dec_range=rate.max()-rate.min(),
        dec_argmax=int(np.argmax(rate.values)), at_min=at_min.mean(), y_at_min=y[at_min].mean() if at_min.any() else np.nan,
        at_max=at_max.mean(), y_at_max=y[at_max].mean() if at_max.any() else np.nan))
r = pd.DataFrame(rows).set_index("f")
Xf = lab[feats].fillna(lab[feats].median())
r["MI"] = mutual_info_classif(Xf, y, random_state=42)
print(r.sort_values("dec_range", ascending=False).round(3).to_string())
print("\nunlabeled train rows: NaN profile vs labeled:", tr[tr.is_cheater.isna()][feats].isna().mean().mean().round(3), lab[feats].isna().mean().mean().round(3))
print("unlabeled rows feature means ~ labeled? max |std diff|:",
      ((tr[tr.is_cheater.isna()][feats].mean()-lab[feats].mean())/lab[feats].std()).abs().max().round(3))
# deterministic relations
t = lab.dropna(subset=["level","account_age_days","level_progression_speed"])
for name, v in {"level/age": t.level/t.account_age_days, "(level-?)/age": t.level/t.account_age_days}.items():
    print(name, "spearman vs lps:", pd.Series(v.values).corr(pd.Series(t.level_progression_speed.values), method="spearman").round(4),
          "ratio lps/(level/age) quantiles:", (t.level_progression_speed/v).quantile([.01,.5,.99]).round(4).tolist())
# reports_received by class
print("\nreports_received target rate:", y.groupby(lab.reports_received).agg(["mean","size"]).round(3).T.to_string())
print("\ndevice_changes target rate:", y.groupby(lab.device_changes_count).mean().round(3).to_dict())
# skill composite
sk = lab[["kill_death_ratio","headshot_percentage","accuracy_score","crosshair_placement","game_sense_score"]].rank(pct=True).mean(axis=1)
print("\nskill composite decile rate:", y.groupby(pd.qcut(sk,10), observed=True).mean().round(3).tolist())
# test NaN pattern
print("\ntest NaN per row:", te[feats].isna().sum(axis=1).value_counts().sort_index().to_dict())
