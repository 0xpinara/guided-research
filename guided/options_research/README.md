# How Much Do Option Metrics Predict Short-Horizon Stock Returns?

### A Power- and Cost-Aware Evaluation, with a Volatility Positive Control

Code and saved results for the paper of the same title (Pınar Aksoy, METU Computer
Engineering). We ask how much daily option-implied information predicts
short-horizon **cross-sectional stock returns**, on a reproducible panel of
134,368 ticker-days for 60 names (2016–2024).

**Headline.** At 60 names the per-cell return signal is small and statistically
indistinguishable from zero (marginal surface lift ≈ 0.003 IC vs. a 0.019 detection
bar). Three analyses make that null *informative* rather than a failure:

1. **Statistical power** — the minimum detectable IC at the Harvey–Liu–Zhu `t>3`
   hurdle is 0.019, so a ~0.004 effect needs ~1,400 names; our null *reconciles*
   with large-universe positives instead of contradicting them.
2. **Volatility positive control** — the identical pipeline detects a large signal
   on realised volatility (≈0.036 IC lift, >10 s.e., 158/162 cells surviving `t>3`).
3. **Signal-injection recovery** — injecting a *known* factor and rerunning the same
   walk-forward, the pipeline recovers a 0.05 effect at `t=8.7` but cannot resolve a
   0.003 one, and the experimental detection floor matches the analytic MDE.

A meta-analysis recovers a tiny, borderline-significant pooled lift
(+0.0029, calendar-block `p≈0.05`), and option information *does* predict returns in
a theory-motivated state — when the **variance risk premium is high** (conditional
IC ≈ 0.025, CI excludes zero in all four cells).

---

## Layout

```
config/        feature_definitions.yaml (feature-set ablations), settings.yaml
run_pipeline.py
src/
  features/    scalar features, SVI surface fit, (Δ,τ) grid, BKM moments, targets
  models/      OLS / ElasticNet / XGBoost workhorses, benchmarks, FFNN
  evaluation/  walk-forward, HAC/Harvey–Liu–Zhu, deflated Sharpe, PBO, regimes
  trading/     dollar-neutral long/short decile backtest
  explainability/ TreeSHAP
scripts/       analysis + figure generation (see "Reproduce")
results/
  tables/      saved metrics (.txt/.csv) — the canonical, version-controlled numbers
  figures/     paper figures (.pdf)
```

## Install

```bash
pip install -r requirements.txt        # pandas, numpy, scipy, scikit-learn, xgboost, pyarrow, matplotlib
```

## Data

End-of-day option files come from the public
[`philippdubach/options-data`](https://github.com/philippdubach/options-data)
repository; returns, market cap, earnings, VIX/VIX3M, the T-bill rate and sector ETF
returns from standard public sources. The built feature panels
(`data/features/**.parquet`) and the two large out-of-sample prediction parquets
(`results/tables/full_matrix_oos_xgb.parquet`, `vol_oos_xgb.parquet`) are **not**
tracked in git (size); they are rebuilt by the pipeline below. The small result
tables under `results/tables/*.txt` and `*.csv` **are** tracked and are sufficient to
regenerate every figure and number in the paper.

## Reproduce

All models are seeded (`random_state=42`) and deterministic — re-running reproduces
the saved per-window IC exactly (`scripts/reproducibility_check.py`).

Each reported number/table/figure maps to a script:

| Reported result | Script(s) |
|---|---|
| Consolidated IC benchmark (both targets, all sets × models) | `run_full_matrix.py`, `run_vol_matrix.py`, `posthoc_stats.py` |
| Returns marginal lift over the Set-A control | `run_full_matrix.py`, `posthoc_stats.py` |
| Volatility positive control + returns-vs-vol dissociation | `run_vol_matrix.py`, `vol_dissociation.py` |
| Statistical power + shippable calculator | `power_analysis.py`, `power_calculator.py` |
| **Signal-injection recovery** | `signal_injection.py` |
| Regime conditioning + meta-analysis (joint calendar-block bootstrap) | `return_extras.py` |
| Post-hoc corrections (HAC, HLZ `t>3`, deflated Sharpe, PBO) | `posthoc_stats.py` |
| TreeSHAP attribution | `treeshap.py` |
| Economics: dollar P&L, deflated Sharpe, PBO, ETF-free IC | `etf_free_econ.py`, `posthoc_stats.py` |
| Run-to-run reproducibility | `reproducibility_check.py` |
| All paper figures | `make_paper_figures.py` |

(All scripts live in `scripts/`.)

Metrics land in `results/tables/`. `make_paper_figures.py` writes PDFs to the paper's
`figures/` directory (path set at the top of the script — point it wherever you like;
the LaTeX source is not included in this code-only repo).

## Caveats (stated in the paper)

Universe selection used membership/liquidity over the sample (survivorship — an
upward bias, so the return null is *conservative*); ETFs (SPY/QQQ/IWM) are kept for
feature construction but dropped from the tradeable universe. We make **no**
tradeable-strategy claim: the robust object is the rank IC, not the dollar Sharpe.
