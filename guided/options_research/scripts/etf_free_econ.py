"""ETF-free return economics: drop SPY/QQQ/IWM from the tradeable universe
(keep them for feature construction) and recompute the dollar-neutral long/short.
ElasticNet headline cells are re-run to get per-ticker predictions; XGBoost reuses
the saved full_matrix_oos_xgb.parquet. Writes etf_free_econ.txt.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pathlib import Path
from run_pipeline import _resolve_feature_ids, _add_ticker_encoding
from run_full_matrix import _window_ranges, _predict_window, _cross_sectional_demean
from src.trading.backtest import backtest_long_short_deciles
from src.utils.config import load_config, load_feature_defs
from src.utils.io_helpers import load_parquet, FEATURES_DIR, SPLITS_DIR, RESULTS_DIR

TAB = RESULTS_DIR / "tables"; ETFS = {"SPY", "QQQ", "IWM"}
EXP_MIN_TRAIN, EXP_STEP, ROLL_TRAIN, ROLL_TEST, ROLL_STEP = 504, 63, 189, 63, 63
out = []
def P(*a): s=" ".join(str(x) for x in a); out.append(s); print(s)

def _nw_t(x):
    """Mean, Newey-West HAC SE, t of a per-window/per-date series."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 3: return (float(np.mean(x)) if n else np.nan), np.nan
    m = float(np.mean(x)); e = x - m; g0 = float(np.mean(e*e))
    L = max(1, int(n ** (1/3))); var = g0
    for k in range(1, L+1):
        var += 2.0*(1-k/(L+1.0))*float(np.mean(e[k:]*e[:-k]))
    se = np.sqrt(max(var, 0.0)/n)
    return m, (m/se if se > 0 else np.nan)

def daily_ic(dates, preds, rets):
    """Mean per-date cross-sectional rank IC and its HAC t. Rank IC within a date
    is invariant to per-date demeaning, so raw realised returns are fine here."""
    df = pd.DataFrame({"d": dates, "p": preds, "r": rets}).dropna()
    ic = df.groupby("d").apply(
        lambda g: g["p"].corr(g["r"], method="spearman") if len(g) >= 5 else np.nan)
    ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
    return _nw_t(ic.to_numpy())

cfg = load_config(); feat_defs = load_feature_defs()
panel = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
splits = load_parquet(SPLITS_DIR / "split_indices.parquet")
for df in (panel, splits): df["date"] = pd.to_datetime(df["date"])
panel = panel.merge(splits, on=["ticker", "date"], how="inner")
rg = FEATURES_DIR / "resolution_2_surface" / "surface_features_all.parquet"
if rg.exists():
    r = load_parquet(rg); r["date"] = pd.to_datetime(r["date"])
    cols = [c for c in r.columns if c.startswith("iv_surf_") or c.startswith("surface_")]
    panel = panel.merge(r[["ticker","date"]+cols].drop_duplicates(["ticker","date"]),
                        on=["ticker","date"], how="left")
panel = panel.sort_values(["ticker","date"]).reset_index(drop=True)
_add_ticker_encoding(panel)
tcols = [c for c in panel.columns if c.startswith("ticker_")]

def en_preds(fs, scheme, h):
    ph = panel.dropna(subset=[f"ret_{h}d"]).reset_index(drop=True)
    feat = _resolve_feature_ids(feat_defs, fs, ph.columns.tolist())
    X = ph[feat+tcols].to_numpy(np.float32, na_value=np.nan)
    dates = ph["date"].to_numpy(); raw = ph[f"ret_{h}d"].to_numpy(np.float64, na_value=np.nan)
    y = _cross_sectional_demean(raw, dates); tk = ph["ticker"].to_numpy()
    np.nan_to_num(X, copy=False); np.nan_to_num(y, copy=False); np.nan_to_num(raw, copy=False)
    uniq = np.sort(np.unique(dates))
    wins = _window_ranges(scheme, uniq, EXP_MIN_TRAIN, EXP_STEP, ROLL_TRAIN, ROLL_TEST, ROLL_STEP)
    P_, R_, D_, T_ = [], [], [], []
    for ts, te, vs, ve in wins:
        tb = uniq[ts:te][:-h] if len(uniq[ts:te]) > h else uniq[ts:te]
        trm = np.isin(dates, list(set(tb))); tem = np.isin(dates, list(set(uniq[vs:ve])))
        if trm.sum()==0 or tem.sum()==0: continue
        pr, _, dt, tkk, rw = _predict_window("ElasticNet", {}, X, y, raw, tk, dates, trm, tem)
        P_.append(pr); R_.append(rw); D_.append(dt); T_.append(tkk)
    return (np.concatenate(D_), np.concatenate(T_), np.concatenate(R_), np.concatenate(P_))

xgb = pd.read_parquet(TAB / "full_matrix_oos_xgb.parquet")
xgb["date"] = pd.to_datetime(xgb["date"])

def cell(model, fs, scheme, h):
    if model == "ElasticNet":
        d, t, r, p = en_preds(fs, scheme, h)
    else:
        g = xgb[(xgb.model=="XGBoost")&(xgb.feature_set==fs)&(xgb.scheme==scheme)&(xgb.horizon==h)]
        d, t, r, p = g["date"].to_numpy(), g["ticker"].to_numpy(), g["y_raw"].to_numpy(), g["pred"].to_numpy()
    res={}
    for excl,tag in [(False,"all"),(True,"exETF")]:
        mm = ~np.isin(t, list(ETFS)) if excl else np.ones(len(t),bool)
        _, s = backtest_long_short_deciles(dates=d[mm], tickers=t[mm], actual_returns=r[mm],
            predicted_returns=p[mm], top_n=6, bottom_n=6, effective_spread_fraction=0.15,
            short_fee_bps_annual=50.0, horizon=h, rebalance_every=h)
        ic_m, ic_t = daily_ic(d[mm], p[mm], r[mm])
        res[tag]=(1000*(1+s.get("total_return",0)), s.get("sharpe",float('nan')), ic_m, ic_t)
    return res

# ---- (1) primary return IC on the ETF-free cross-section ----------------------
# Ranking SPY/QQQ against single names in one cross-section is a mild category
# error; the headline IC should be measured on the tradeable (single-name) set.
P("PRIMARY RETURN IC -- all names vs ETF-free cross-section (mean per-date rank IC, HAC t)")
P(f"{'cell':30}{'IC all':>9}{'(t)':>7}{'IC exETF':>10}{'(t)':>7}{'dIC':>9}")
for model in ["ElasticNet","XGBoost"]:
    for fs in ["D","A"]:
        for scheme,h in [("rolling_9m_3m",5),("expanding",3)]:
            r=cell(model,fs,scheme,h)
            dic = r['exETF'][2]-r['all'][2]
            P(f"{model}/{fs}/{scheme}/h{h:<3}"
              f"{r['all'][2]:>9.4f}{r['all'][3]:>7.2f}{r['exETF'][2]:>10.4f}{r['exETF'][3]:>7.2f}{dic:>+9.4f}")
            cell._cache = getattr(cell,'_cache',{}); cell._cache[(model,fs,scheme,h)] = r

# ---- (2) ETF-free economics (demoted; below the deflated-Sharpe bar) ----------
P("\nETF-free dollar-neutral long/short (drop SPY/QQQ/IWM from tradeable universe)")
P(f"{'cell':34}{'$1000 all':>12}{'Sh all':>9}{'$1000 exETF':>14}{'Sh exETF':>11}")
for model in ["ElasticNet","XGBoost"]:
    for fs in ["D","A"]:
        for scheme,h in [("rolling_9m_3m",5),("expanding",3)]:
            r=cell._cache[(model,fs,scheme,h)]
            P(f"{model}/{fs}/{scheme}/h{h:<6}"
              f"{r['all'][0]:>11.0f}{r['all'][1]:>9.2f}{r['exETF'][0]:>14.0f}{r['exETF'][1]:>11.2f}")
(TAB/"etf_free_econ.txt").write_text("\n".join(out))
print("[written] etf_free_econ.txt")
