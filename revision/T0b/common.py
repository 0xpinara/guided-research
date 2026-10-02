"""Shared definitions for Task 0b: feature sets, per-window preprocessing, IC and HAC.

Everything that learns parameters (winsorisation bounds, imputation medians, missingness
patterns, scaling) is fitted on the training rows of one walk-forward window only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision" / "out" / "T0b"
ETFS = ("SPY", "QQQ", "IWM")
SETS = ["A", "B", "C", "candidate_6", "D", "repr_svi", "repr_grid", "repr_grid_raw", "repr_bkm"]
MODELS = ["OLS", "ElasticNet", "XGBoost"]
SCHEMES = ["expanding", "rolling_9m_3m"]
HORIZONS = [1, 3, 5]
EXP_MIN_TRAIN, EXP_STEP = 504, 63          # BIR window geometry (run_full_matrix defaults)
ROLL_TRAIN, ROLL_TEST, ROLL_STEP = 189, 63, 63
WINSOR_Q = (0.001, 0.999)                  # BIR settings.yaml preprocessing.winsorize_quantiles


# Features that need a future event date (realised next earnings / ex-dividend date) and have
# no point-in-time announcement record: excluded from every canonical predictive set (Task 0b R9).
EXCLUDED_FUTURE_EVENT = {
    "feat_43": "days_to_next_earnings: next realised report date (Compustat rdq / Yahoo)",
    "feat_44": "earnings_flag: 1{days_to_next_earnings <= 7}",
    "feat_45": "days_to_ex_div: next realised ex-dividend date (Yahoo; no declaration dates)",
    "feat_51": "implied_earnings_move: defined only through days_to_next_earnings",
}


def corrected_sets() -> dict[str, list[str]]:
    """BIR feature sets (config/feature_definitions.yaml, unchanged) minus the excluded features."""
    fd = yaml.safe_load(open(ROOT / "config" / "feature_definitions.yaml"))
    return {fs: [str(c) for c in fd["feature_sets"][fs]["features"] if str(c) not in EXCLUDED_FUTURE_EVENT]
            for fs in SETS}


def write_feature_sets_yaml(path: Path) -> None:
    fd = yaml.safe_load(open(ROOT / "config" / "feature_definitions.yaml"))
    doc = {"note": "Canonical Task 0b feature sets = BIR sets minus future-event features (no replacements).",
           "excluded": EXCLUDED_FUTURE_EVENT, "sets": {}}
    for fs, cols in corrected_sets().items():
        doc["sets"][fs] = {"n_features_bir": len(fd["feature_sets"][fs]["features"]),
                           "n_features": len(cols), "features": cols}
    path.write_text(yaml.safe_dump(doc, sort_keys=False, width=110))


def feature_columns(fs: str, columns) -> list[str]:
    """Canonical columns of a feature set (all must exist; none may be a future-event feature)."""
    cols = corrected_sets()[fs]
    assert not set(cols) & set(EXCLUDED_FUTURE_EVENT), fs
    missing = [c for c in cols if c not in set(columns)]
    if missing:
        raise KeyError(f"set {fs}: columns missing from the panel: {missing}")
    return cols


def windows(scheme: str, n_dates: int) -> list[tuple[int, int, int, int]]:
    """(train_start, train_end, test_start, test_end) date indices; same rule as BIR."""
    out = []
    if scheme == "expanding":
        for tr_e in range(EXP_MIN_TRAIN, n_dates - EXP_STEP, EXP_STEP):
            out.append((0, tr_e, tr_e, min(tr_e + EXP_STEP, n_dates)))
    else:
        ts = ROLL_TRAIN
        while ts + ROLL_TEST <= n_dates:
            out.append((ts - ROLL_TRAIN, ts, ts, ts + ROLL_TEST))
            ts += ROLL_STEP
    return out


def demean_by_date(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    s = pd.Series(values)
    return (s - s.groupby(dates).transform("mean")).to_numpy()


class WindowPreprocessor:
    """Fitted on training rows only, then applied to any rows of the same window.

    winsorise: clip each feature to its training [0.1%, 99.9%] quantiles (NaN ignored)
    impute (linear models): training per-ticker median, then training global median;
        features that are entirely missing in the training rows are dropped for the window
    indicators (linear models): 1{missing} for each distinct training missingness pattern
    XGBoost receives the winsorised features with NaN left as missing (native handling).
    """

    def fit(self, X: np.ndarray, tickers: np.ndarray, impute: bool) -> "WindowPreprocessor":
        self.impute = impute
        with np.errstate(all="ignore"):
            self.lo, self.hi = np.nanquantile(X, WINSOR_Q, axis=0)
        self.keep = np.isfinite(self.lo) & np.isfinite(self.hi)     # drop all-missing training columns
        if not impute:
            return self
        Xw = self._clip(X)[:, self.keep]
        miss = np.isnan(Xw)
        self.global_med = np.nanmedian(Xw, axis=0)
        df = pd.DataFrame(Xw)
        df["_t"] = tickers
        self.ticker_med = df.groupby("_t").median()                # per-ticker training medians
        patterns, self.ind_cols = {}, []
        for j in np.flatnonzero(miss.any(axis=0) & ~miss.all(axis=0)):
            key = miss[:, j].tobytes()
            if key not in patterns:
                patterns[key] = j
                self.ind_cols.append(j)
        return self

    def _clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lo, self.hi)        # NaN stays NaN

    def transform(self, X: np.ndarray, tickers: np.ndarray) -> np.ndarray:
        Xw = self._clip(X)[:, self.keep]
        if not self.impute:
            return Xw
        miss = np.isnan(Xw)
        med = self.ticker_med.reindex(tickers).to_numpy()            # unseen ticker -> NaN -> global median
        out = np.where(miss, med, Xw)
        out = np.where(np.isnan(out), self.global_med, out)
        ind = miss[:, self.ind_cols].astype(float)
        return np.hstack([out, ind])


def nw_tstat(x: np.ndarray) -> tuple[float, float, float]:
    """Mean, Newey-West HAC SE (Bartlett, lag floor(n^(1/3))) and t; same formula as BIR."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (float(np.mean(x)) if n else np.nan), np.nan, np.nan
    m = float(np.mean(x))
    e = x - m
    var = float(np.mean(e * e))
    lag = max(1, int(n ** (1 / 3)))
    for k in range(1, lag + 1):
        var += 2.0 * (1 - k / (lag + 1.0)) * float(np.mean(e[k:] * e[:-k]))
    se = np.sqrt(max(var, 0.0) / n)
    return m, se, (m / se if se > 0 else np.nan)


def daily_ic(dates: np.ndarray, pred: np.ndarray, y: np.ndarray, min_names: int = 5) -> pd.DataFrame:
    """Canonical IC: cross-sectional Spearman on each date (average ranks for ties).

    A date on which the predictions are constant across names scores 0 (no ranking skill).
    Dates with fewer than min_names valid names are skipped.
    """
    d = pd.DataFrame({"date": dates, "p": pred, "y": y}).dropna()
    g = d.groupby("date")
    d["rp"], d["ry"] = g.p.rank(), g.y.rank()
    s = d.groupby("date").agg(n=("p", "size"), pmin=("p", "min"), pmax=("p", "max"))
    d["rp_c"] = d.rp - d.groupby("date").rp.transform("mean")
    d["ry_c"] = d.ry - d.groupby("date").ry.transform("mean")
    num = (d.rp_c * d.ry_c).groupby(d.date).sum()
    den = np.sqrt((d.rp_c ** 2).groupby(d.date).sum() * (d.ry_c ** 2).groupby(d.date).sum())
    ic = (num / den).where(den > 0)
    const = s.pmax - s.pmin <= 1e-14 * np.maximum(1.0, s.pmax.abs())
    ic[const] = 0.0
    out = pd.DataFrame({"ic": ic, "n": s.n, "constant": const})
    return out[out.n >= min_names]


def pooled_ic(pred: np.ndarray, y: np.ndarray) -> float:
    """BIR definition: one Spearman over all ticker-days of the window (NaN if pred is constant)."""
    p = pd.Series(pred)
    if p.nunique() < 2 or len(p) <= 10:
        return np.nan
    return float(p.corr(pd.Series(y), method="spearman"))
