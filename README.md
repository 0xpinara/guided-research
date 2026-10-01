# A Power-Aware Evaluation Template for Option-Implied Return Prediction, with a Volatility Positive Control

Code and saved results for the paper of the same title. We ask whether daily
option-implied information predicts short-horizon cross-sectional stock returns, and
how reliably, on a panel of 134,368 ticker-days for 60 U.S. tickers (57 large-cap
stocks and three broad ETFs) from 2016 to 2024.

## What the paper finds

- At 60 names the per-cell return signal is small (marginal surface lift of about
  0.003 rank IC) and does not clear the `t>3` detection threshold, which is about
  0.019 at this universe size. That 0.019 is the effect at which the expected HAC
  `t` reaches 3 — a significance/detection threshold, not an 80%-power minimum
  detectable effect.
- The same pipeline ranks realised volatility strongly (surface lift of about 0.036
  IC, more than ten standard errors, 158 of 162 cells clearing `t>3`). This is the
  positive control: it shows the pipeline detects signal when signal is present.
- A signal-injection test recovers a planted 0.05 factor cleanly but cannot resolve
  a 0.003 one; the empirical detection floor matches the analytic value near 0.02.
- A meta-analysis pools a small lift (+0.0029, calendar-block `p ≈ 0.05`) that does
  not clear the threshold.
- Conditionally, when the variance risk premium is high, the surface predicts
  returns with an IC of about 0.025 (the interval excludes zero in all four cells).
- The long/short economics are demoted. The headline Sharpe (1.1504) is just below
  the deflated-Sharpe bar (1.1529); that verdict is sensitive to the trial count, so
  the no-tradeable-strategy conclusion rests on the probability of backtest
  overfitting (PBO = 0.571).

## Layout

```
config/        feature_definitions.yaml (feature-set ablations), settings.yaml
run_pipeline.py
src/
  features/       scalar features, SVI surface fit, (delta,tau) grid, BKM moments, targets
  models/         OLS, ElasticNet, XGBoost; FFNN and a temporal-fusion transformer
                  (the two neural baselines used only for the deflated-Sharpe trial
                  count); TabNet and Set-Transformer are present but not used in the
                  reported results
  evaluation/     walk-forward, HAC / Harvey-Liu-Zhu t>3, deflated Sharpe, PBO, regimes
  trading/        dollar-neutral long/short decile backtest
  explainability/ TreeSHAP
scripts/          analysis and figure generation (see Reproduce)
results/
  tables/         saved metrics (.txt / .csv) — the version-controlled numbers
  figures/        paper figures (.pdf)
```

## Install

```bash
pip install -r requirements.txt   # pandas, numpy, scipy, scikit-learn, xgboost, pyarrow, matplotlib
```

## Data

End-of-day option files come from the `philippdubach/options-data` community GitHub
repository, which was public when we accessed it and has since been made private; the
files were distributed under the MIT License. Returns, market capitalization,
earnings dates, VIX and VIX3M, the Treasury-bill rate, and sector-ETF returns are
from Yahoo Finance.

The built feature panels (`data/features/**.parquet`) and the two large
out-of-sample prediction files (`results/tables/full_matrix_oos_xgb.parquet`,
`vol_oos_xgb.parquet`) are not tracked in git because of their size; the pipeline
rebuilds them. The small tables under `results/tables/*.txt` and `*.csv` are tracked
and are the canonical numbers reported in the paper.

## Reproduce

All models use `random_state=42` and are deterministic; re-running reproduces the
saved per-window IC exactly (`scripts/reproducibility_check.py`). Scripts that start
from the large out-of-sample prediction files need those files rebuilt first, since
they are not tracked. Each reported result maps to a script:

| Reported result | Script(s) |
|---|---|
| Consolidated IC benchmark (both targets, all sets x models) | `run_full_matrix.py`, `run_vol_matrix.py`, `posthoc_stats.py` |
| Returns marginal lift over the Set-A control | `run_full_matrix.py`, `posthoc_stats.py` |
| Sector-neutral robustness check | `sector_neutral_ic.py` |
| Volatility positive control and returns-vs-volatility dissociation | `run_vol_matrix.py`, `vol_dissociation.py` |
| Detection threshold and calculator | `power_analysis.py`, `power_calculator.py` |
| Signal-injection recovery | `signal_injection.py` |
| Regime conditioning and meta-analysis (joint calendar-block bootstrap) | `return_extras.py` |
| Post-hoc corrections (HAC, Harvey-Liu-Zhu t>3, deflated Sharpe, PBO) | `posthoc_stats.py` |
| TreeSHAP attribution | `treeshap.py` |
| Economics: dollar P&L, deflated Sharpe, PBO, ETF-free IC | `etf_free_econ.py`, `posthoc_stats.py` |
| Run-to-run reproducibility | `reproducibility_check.py` |
| Figures | `make_paper_figures.py` |

All scripts are in `scripts/`. `make_paper_figures.py` writes PDF figures to
`results/figures/`.

## Caveats (as stated in the paper)

Universe selection used membership and liquidity over the sample, which is a
survivorship bias that pushes reported numbers upward, so the return null is
conservative. The three ETFs (SPY, QQQ, IWM) are kept for feature construction but
dropped from the tradeable universe. We make no tradeable-strategy claim; the
reported object is the rank IC, not the dollar Sharpe.
