"""Task 5, round 2 (2026-11-09): look for further injected anomaly types with label-free structure scans.

The 7 known types (task5_nb.py section 5) are detached clusters: groups of 200-311 rows separated from the bulk
by an empty gap in some column, with zero pairwise overlap. This script scans DERIVED features (within-row
period changes, std/mean ratios, cross-feature consistency, NaN patterns, per-period population range for
every period, not only period 4) for more groups like that, among the rows outside the known union.

player_id is dropped right after loading: no ID information is used anywhere (project rule, pending the TA).
Run: python task5/src/analysis_r2.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import paths  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

df = pd.read_csv(paths.task_data(5) / "test.csv").drop(columns=["player_id"])
N = len(df)
BASES = sorted({c.rsplit("_", 1)[0] for c in df.columns if c[-2:] in ("_1", "_2", "_3", "_4")})


def pm(f):
    return df[[f"{f}_{i}" for i in range(1, 5)]].to_numpy(float)


def out_of_range4(f):
    m = pm(f)
    return (m[:, 3] > np.nanmax(m[:, :3])) | (m[:, 3] < np.nanmin(m[:, :3]))


def below_any(f, cut):
    return (pm(f) < cut).any(axis=1)


def known_types():
    T = {
        "A_cheat": np.logical_or.reduce([out_of_range4(f) for f in
                                         ("skill_rating", "skill_trajectory", "accuracy_mean", "kda_mean",
                                          "friend_removes", "reaction_time_mean")]),
        "B_takeover": out_of_range4("currency_transfers") | out_of_range4("play_pattern_consistency"),
        "C_purchase": out_of_range4("purchase_frequency") | out_of_range4("purchase_amount_mean"),
        "bot": below_any("ip_hash_entropy", 0.25) | below_any("login_time_variance", 0.9)
               | below_any("session_duration_std", 6.0),
        "low_activity": below_any("activity_level", 0.32),
        "low_comm": below_any("communication_frequency", 0.15),
        "low_stability": (df["behavior_stability"] < 0.5).to_numpy(),
    }
    return pd.DataFrame(T).astype(int)


types = known_types()
union = types.sum(axis=1).to_numpy() > 0
rest = ~union
print(f"known union {union.sum()} rows ({union.mean():.3%}); rest {rest.sum()}")


# --------------------------------------------------------------------------------------------------
# Derived features
# --------------------------------------------------------------------------------------------------
def derived():
    D = {}
    for f in BASES:
        m = pm(f)
        prev = m[:, :3]
        D[f"{f}|d43"] = m[:, 3] - m[:, 2]
        D[f"{f}|d4prev"] = m[:, 3] - np.nanmean(prev, axis=1)
        D[f"{f}|rng"] = np.nanmax(m, axis=1) - np.nanmin(m, axis=1)
        D[f"{f}|rng123"] = np.nanmax(prev, axis=1) - np.nanmin(prev, axis=1)
        D[f"{f}|mean"] = np.nanmean(m, axis=1)
        D[f"{f}|d21"] = m[:, 1] - m[:, 0]
        D[f"{f}|d32"] = m[:, 2] - m[:, 1]
        # slope over the 4 periods (least squares, NaN-aware via mean of consecutive diffs)
        D[f"{f}|slope"] = np.nanmean(np.diff(m, axis=1), axis=1)
        D[f"{f}|nnan"] = np.isnan(m).sum(axis=1).astype(float)
    pairs = [("session_duration_std", "session_duration_mean"), ("kda_std", "kda_mean"),
             ("accuracy_std", "accuracy_mean"), ("reaction_time_std", "reaction_time_mean"),
             ("purchase_amount_std", "purchase_amount_mean"), ("friend_removes", "friend_adds"),
             ("login_count", "activity_level"), ("location_changes", "device_count"),
             ("purchase_amount_mean", "purchase_frequency"), ("refund_count", "purchase_frequency"),
             ("currency_transfers", "purchase_frequency"), ("ip_hash_entropy", "device_count"),
             ("session_duration_mean", "activity_level"), ("login_count", "session_duration_mean"),
             ("skill_rating", "kda_mean"), ("skill_rating", "accuracy_mean"), ("kda_mean", "accuracy_mean"),
             ("skill_trajectory", "skill_rating"), ("performance_stability", "kda_std"),
             ("play_pattern_consistency", "performance_stability"), ("communication_frequency", "friend_adds"),
             ("login_time_variance", "login_count"), ("guild_changes", "friend_adds")]
    for a, b in pairs:
        ma, mb = pm(a), pm(b)
        r = ma / np.where(np.abs(mb) < 1e-9, np.nan, mb)
        for i in range(4):
            D[f"{a}/{b}|{i + 1}"] = r[:, i]
        D[f"{a}/{b}|mean"] = np.nanmean(r, axis=1)
        dif = ma - mb
        D[f"{a}-{b}|mean"] = np.nanmean(dif, axis=1)
    # geography: haversine jumps between consecutive periods
    lat, lon = pm("login_lat"), pm("login_lon")
    la, lo = np.radians(lat), np.radians(lon)
    for i in range(3):
        h = (np.sin((la[:, i + 1] - la[:, i]) / 2) ** 2
             + np.cos(la[:, i]) * np.cos(la[:, i + 1]) * np.sin((lo[:, i + 1] - lo[:, i]) / 2) ** 2)
        D[f"geo_jump|{i + 1}{i + 2}"] = 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    D["nan_total"] = df.isna().sum(axis=1).to_numpy(float)
    return pd.DataFrame(D)


Dv = derived()
print("derived features:", Dv.shape[1])


# --------------------------------------------------------------------------------------------------
# Gap scan: a group of rows detached from the bulk by an empty run of histogram bins
# --------------------------------------------------------------------------------------------------
def gap_groups(x, bins=200, min_run=4, min_n=25, max_n=2500):
    """Return a list of (side, cut, n, n_rest) for every empty run of >= min_run bins whose smaller side
    holds min_n..max_n rows. side: 'low' = rows below the gap, 'high' = rows above."""
    v = x[np.isfinite(x)]
    if len(v) < 1000:
        return []
    lo, hi = np.min(v), np.max(v)
    if hi <= lo:
        return []
    cnt, edges = np.histogram(v, bins=bins, range=(lo, hi))
    out = []
    empty = cnt == 0
    i = 0
    while i < bins:
        if empty[i]:
            j = i
            while j < bins and empty[j]:
                j += 1
            if j - i >= min_run:
                cut = (edges[i] + edges[j]) / 2
                below = int((v < cut).sum())
                above = len(v) - below
                side, n = ("low", below) if below < above else ("high", above)
                if min_n <= n <= max_n:
                    out.append((side, float(cut), n, j - i))
            i = j
        else:
            i += 1
    return out


def scan(frame, label):
    rows = []
    for c in frame.columns:
        x = frame[c].to_numpy(float)
        for transform in ("raw", "log"):
            if transform == "log":
                if np.nanmin(x) < 0:
                    continue
                xx = np.log1p(x)
            else:
                xx = x
            for side, cut, n, run in gap_groups(xx):
                sel = (xx < cut) if side == "low" else (xx > cut)
                sel &= np.isfinite(xx)
                rows.append(dict(src=label, feature=c, transform=transform, side=side, cut=cut, n=n, run=run,
                                 n_rest=int((sel & rest).sum()), frac_in_union=float(union[sel].mean())))
    return pd.DataFrame(rows)


raw_cols = [c for c in df.columns]
res = pd.concat([scan(df[raw_cols], "raw"), scan(Dv, "derived")], ignore_index=True)
res = res.sort_values(["n_rest", "run"], ascending=[False, False])
pd.set_option("display.width", 220, "display.max_rows", 400, "display.max_colwidth", 60)
print("\n=== gap groups (smaller side 25..2500 rows), sorted by rows outside the known union ===")
print(res[res.n_rest >= 20].drop_duplicates(["feature", "side"]).head(80).to_string(index=False))


# --------------------------------------------------------------------------------------------------
# Population-range check for EVERY period (the known A/B/C types use period 4 vs periods 1-3)
# --------------------------------------------------------------------------------------------------
print("\n=== rows outside the other periods' population range, per base and period (rest only) ===")
rr = []
for f in BASES:
    m = pm(f)
    for k in range(4):
        others = np.delete(m, k, axis=1)
        hi, lo = np.nanmax(others), np.nanmin(others)
        o = (m[:, k] > hi) | (m[:, k] < lo)
        if o.sum():
            rr.append(dict(base=f, period=k + 1, n=int(o.sum()), n_rest=int((o & rest).sum()),
                           n_high=int((m[:, k] > hi).sum()), n_low=int((m[:, k] < lo).sum())))
rr = pd.DataFrame(rr).sort_values("n_rest", ascending=False)
print(rr.head(40).to_string(index=False))

# --------------------------------------------------------------------------------------------------
# NaN structure: is missingness random, or is there a block pattern?
# --------------------------------------------------------------------------------------------------
print("\n=== NaN count per row: known union vs rest ===")
nn = df.isna().sum(axis=1)
print(pd.crosstab(np.clip(nn, 0, 15), union, normalize="columns").round(4).T.to_string())
nan_bases = pd.DataFrame({f: np.isnan(pm(f)).sum(axis=1) for f in BASES})
print("rows with all 4 periods NaN for some base:", int((nan_bases == 4).any(axis=1).sum()),
      "| expected under independent 5% masking:", round(N * len(BASES) * 0.05 ** 4, 2))
print("rows with >=3 of 4 periods NaN for some base:", int((nan_bases >= 3).any(axis=1).sum()))

out = ROOT / "task5" / "outputs" / "scan_r2_gaps.csv"
res.to_csv(out, index=False)
print("saved", out)


# --------------------------------------------------------------------------------------------------
# Cross-base structure. Normal rows: the 29 bases are mutually independent (each base = per-player level +
# per-period noise). So a type made of MODERATE shifts in several bases at once (no empty gap anywhere)
# still shows up as excess co-occurrence of their tails.
# --------------------------------------------------------------------------------------------------
lvl = pd.DataFrame({f: np.nanmean(pm(f), axis=1) for f in BASES if f not in ("login_lat", "login_lon")})
C = lvl[rest].rank(pct=True).corr().abs().to_numpy().copy()
np.fill_diagonal(C, 0)
print(f"\nmax |Spearman| between per-row means of two different bases (rest rows): {C.max():.3f}")


def tail_cooccurrence(mask, q=0.05):
    """All cross-base pairs of 5% tails (level, within-row sd, period-4 change, period-4 value):
    observed vs expected-under-independence count among `mask` rows, z = (obs - exp) / sqrt(exp)."""
    V = {}
    for f in BASES:
        if f in ("login_lat", "login_lon"):
            continue
        m = pm(f)
        V[f"{f}|lvl"] = np.nanmean(m, axis=1)
        V[f"{f}|sd"] = np.nanstd(m, axis=1)
        V[f"{f}|d4"] = m[:, 3] - np.nanmean(m[:, :3], axis=1)
        V[f"{f}|p4"] = m[:, 3]
    V["account_age_days|lvl"] = df["account_age_days"].to_numpy(float)
    V["behavior_stability|lvl"] = df["behavior_stability"].to_numpy(float)
    V = pd.DataFrame(V)[mask].reset_index(drop=True)
    ind = {}
    for c in V.columns:
        ind[c + "-"] = (V[c] <= V[c].quantile(q)).to_numpy()
        ind[c + "+"] = (V[c] >= V[c].quantile(1 - q)).to_numpy()
    I = pd.DataFrame(ind).astype(np.float32)
    p = I.mean().to_numpy()
    O = I.T.values @ I.values
    E = len(V) * np.outer(p, p)
    Z = (O - E) / np.sqrt(E + 1e-9)
    cols = list(I.columns)
    base = [c.split("|")[0] for c in cols]
    rows = [(cols[i], cols[j], int(O[i, j]), round(float(E[i, j]), 1), round(float(Z[i, j]), 2))
            for i in range(len(cols)) for j in range(i + 1, len(cols)) if base[i] != base[j]]
    return pd.DataFrame(rows, columns=["a", "b", "obs", "exp", "z"]).sort_values("z", ascending=False)


co = tail_cooccurrence(rest)
print(f"\n=== cross-base tail co-occurrence, rest rows: {len(co)} pairs; z quantiles .5/.99/.999 = "
      f"{np.round(co.z.quantile([.5, .99, .999]).to_numpy(), 2)} ===")
print(co.head(8).to_string(index=False))

# The one outlier pair: accuracy_std+ x kda_std+. Its excess saturates at ~190 rows from the 8% tails on.
a_, b_ = lvl["accuracy_std"].to_numpy(), lvl["kda_std"].to_numpy()
ok = rest & np.isfinite(a_) & np.isfinite(b_)
for q in (0.02, 0.05, 0.08, 0.10, 0.15):
    o = int((ok & (a_ >= np.nanquantile(a_[ok], 1 - q)) & (b_ >= np.nanquantile(b_[ok], 1 - q))).sum())
    print(f"   accuracy_std top {q:.0%} x kda_std top {q:.0%}: obs {o:4d}, exp {ok.sum() * q * q:6.1f}, "
          f"excess {o - ok.sum() * q * q:6.1f}")

# Remove that corner and scan again: nothing else stands out (max z ~4.2 over ~23k pairs = noise level).
corner = ok & (a_ >= np.nanquantile(a_[ok], .9)) & (b_ >= np.nanquantile(b_[ok], .9))
co2 = tail_cooccurrence(rest & ~corner)
print(f"\nafter removing the accuracy_std/kda_std top-10% corner: max z {co2.z.max():.2f}, "
      f".999 quantile {co2.z.quantile(.999):.2f} over {len(co2)} pairs")
print(co2.head(5).to_string(index=False))
