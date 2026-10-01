"""TreeSHAP: which option-surface features drive the (weak) RETURN prediction
versus the (strong) VOLATILITY prediction. Uses XGBoost's exact built-in
TreeSHAP (pred_contribs); no GPU. Trains Set-D models on the documented train
block, attributes on the test block. Writes treeshap_importance.csv + report.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, pandas as pd, xgboost as xgb, yaml
from pathlib import Path
from run_pipeline import _resolve_feature_ids, _add_ticker_encoding
from run_full_matrix import _cross_sectional_demean
from run_vol_matrix import _build_rv
from src.utils.config import load_config, load_feature_defs
from src.utils.io_helpers import load_parquet, FEATURES_DIR, SPLITS_DIR, RESULTS_DIR

TAB = RESULTS_DIR / "tables"
TRAIN_END = pd.Timestamp("2021-05-18")
fd = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config" / "feature_definitions.yaml"))
NAME = {k: v["name"] for k, v in fd.get("features", {}).items()
        if isinstance(v, dict) and "name" in v}
# surface-feature human labels
def nice(c):
    if c in NAME: return NAME[c]
    if c.startswith("ticker_"): return "(ticker dummy)"
    return c

cfg = load_config(); feat_defs = load_feature_defs()
panel = load_parquet(FEATURES_DIR / "resolution_1_scalar" / "panel_unnormalized.parquet")
splits = load_parquet(SPLITS_DIR / "split_indices.parquet")
for df in (panel, splits): df["date"] = pd.to_datetime(df["date"])
panel = panel.merge(splits, on=["ticker", "date"], how="inner").sort_values(["ticker", "date"]).reset_index(drop=True)
_add_ticker_encoding(panel)

xpar = dict(n_estimators=400, max_depth=6, learning_rate=0.02, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=4)

def importance(target, h=5):
    feat_ids = _resolve_feature_ids(feat_defs, "D", panel.columns.tolist())
    tcols = [c for c in panel.columns if c.startswith("ticker_")]
    cols = feat_ids + tcols
    if target == "return":
        y_raw = panel[f"ret_{h}d"].to_numpy(float)
        keep = panel[f"ret_{h}d"].notna().to_numpy()
    else:
        rv = _build_rv(panel, h); y_raw = rv.to_numpy(float); keep = rv.notna().to_numpy()
    d = panel[keep].copy(); yraw = y_raw[keep]
    X = d[cols].to_numpy(np.float32, na_value=np.nan)
    np.nan_to_num(X, copy=False)
    y = _cross_sectional_demean(yraw, d["date"].to_numpy()); np.nan_to_num(y, copy=False)
    tr = (d["date"] < TRAIN_END).to_numpy(); te = ~tr
    m = xgb.XGBRegressor(**xpar).fit(X[tr], y[tr])
    contrib = m.get_booster().predict(xgb.DMatrix(X[te], feature_names=cols),
                                      pred_contribs=True)[:, :-1]  # drop bias
    imp = np.abs(contrib).mean(0)
    s = pd.Series(imp, index=cols).sort_values(ascending=False)
    # collapse all ticker dummies into one entry
    tk = s[[c for c in s.index if c.startswith("ticker_")]].sum()
    s = s[[c for c in s.index if not c.startswith("ticker_")]]
    s["(ticker dummies, sum)"] = tk
    return s.sort_values(ascending=False)

rows = []
res = {}
for tgt in ["return", "volatility"]:
    s = importance("return" if tgt == "return" else "vol", h=5)
    s = s / s.sum()   # normalise to share of attribution
    res[tgt] = s
    for feat, val in s.head(15).items():
        rows.append({"target": tgt, "feature": feat, "label": nice(feat), "share": float(val)})
imp = pd.DataFrame(rows)
imp.to_csv(TAB / "treeshap_importance.csv", index=False)

# quick text report
lines = ["TreeSHAP mean|contribution| share, Set D, h=5 (XGBoost, exact TreeSHAP)\n"]
for tgt in ["return", "volatility"]:
    lines.append(f"== {tgt} ==")
    for feat, val in res[tgt].head(12).items():
        lines.append(f"   {100*val:5.1f}%  {feat:22} {nice(feat)}")
    lines.append("")
(TAB / "treeshap.txt").write_text("\n".join(lines))
print("\n".join(lines))
print("[written] treeshap_importance.csv, treeshap.txt")
