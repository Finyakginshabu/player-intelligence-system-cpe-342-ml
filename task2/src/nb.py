"""Naive Bayes for task 2 (round 4, 2026-11-09).

Within each segment the features are uncorrelated (max |Spearman| 0.02-0.05) and missingness is independent of
the segment, so p(x | c) = prod_f p(x_f | c). Naive Bayes is then the correctly specified model, and a missing
feature is skipped (exact marginalisation), not imputed. All tables are fit on the rows passed to fit().

BinnedNB  per-class frequencies over quantile bins (continuous) / values (discrete, <= max_levels) / levels (categorical)
SmoothNB  continuous features: fine equal-mass bins (= histogram on the empirical-CDF scale) with per-class counts
          smoothed by a Gaussian kernel of `sigma` bins, i.e. a KDE in CDF space (less noise per log-ratio term)
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp


class BinnedNB:
    def __init__(self, cats, n_classes=4, n_bins=32, alpha=1.0, max_levels=64):
        self.cats, self.K = set(cats), n_classes
        self.n_bins, self.alpha, self.max_levels = n_bins, alpha, max_levels

    def _codes(self, X):
        """Integer code per column (-1 = NaN or unseen level)."""
        out = np.full((len(X), len(self.cols_)), -1, dtype=np.int32)
        for j, c in enumerate(self.cols_):
            kind, ref = self.spec_[c]
            if kind == "cat":
                out[:, j] = pd.Categorical(X[c], categories=ref).codes
            else:
                x = X[c].to_numpy(float)
                ok = np.isfinite(x)
                out[ok, j] = np.searchsorted(ref, x[ok], side="right")
        return out

    def _counts(self, codes, j, nb, y):
        T = np.zeros((nb, self.K))
        ok = codes[:, j] >= 0
        np.add.at(T, (codes[ok, j], y[ok]), 1.0)
        return T

    def fit(self, X, y):
        y = np.asarray(y)
        self.cols_, self.spec_ = list(X.columns), {}
        for c in self.cols_:
            if c in self.cats:
                self.spec_[c] = ("cat", sorted(X[c].dropna().unique()))
                continue
            x = X[c].to_numpy(float)
            x = x[np.isfinite(x)]
            u = np.unique(x)
            if len(u) <= self.max_levels:  # discrete: one bin per value
                self.spec_[c] = ("disc", (u[1:] + u[:-1]) / 2)
            else:
                self.spec_[c] = ("cont", np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)[1:-1])))
        codes = self._codes(X)
        self.prior_ = np.log(np.bincount(y, minlength=self.K) / len(y))
        self.tables_ = []
        for j, c in enumerate(self.cols_):
            kind, ref = self.spec_[c]
            nb = len(ref) if kind == "cat" else len(ref) + 1
            T = self._table(self._counts(codes, j, nb, y), kind)
            self.tables_.append(np.log(T / T.sum(0, keepdims=True)))
        return self

    def _table(self, T, kind):
        return T + self.alpha

    def predict_log_joint(self, X):
        codes = self._codes(X)
        L = np.tile(self.prior_, (len(X), 1))
        for j in range(len(self.cols_)):
            ok = codes[:, j] >= 0
            L[ok] += self.tables_[j][codes[ok, j]]
        return L

    def predict_proba(self, X):
        L = self.predict_log_joint(X)
        return np.exp(L - logsumexp(L, axis=1, keepdims=True))


class SmoothNB(BinnedNB):
    def __init__(self, cats, n_classes=4, n_bins=128, sigma=4.0, alpha=1.0, max_levels=64):
        super().__init__(cats, n_classes, n_bins, alpha, max_levels)
        self.sigma = sigma

    def _table(self, T, kind):
        if kind != "cont":
            return T + self.alpha
        return gaussian_filter1d(T, self.sigma, axis=0, mode="reflect") + self.alpha / len(T)
