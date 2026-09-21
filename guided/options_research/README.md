# Statistical Resolution in Option-Implied Return Prediction

### A Power-Aware Evaluation Protocol for Constrained Cross-Sections

Code and saved results for the paper of the same title (Pınar Aksoy, METU Computer
Engineering). Large-universe studies already establish that some option
characteristics carry incremental information for future stock returns. This project
asks a different question: **when the observable option cross-section is small, which
predictive effects can a design of that size actually resolve, and how do you tell a
weak effect apart from inadequate statistical resolution?**

The worked example is a fully reproducible panel of 134,368 ticker-days over 60 U.S.
tickers, 2016–2024, with a median daily cross-section of 59 names.

**Headline.** The option surface's marginal contribution to return ranking is
0.002–0.004 rank IC. The detection threshold at the Harvey–Liu–Zhu `t>3` hurdle is
0.019, and 80% power needs about 0.025. The estimate is five to ten times below the
design's resolution, so the unconditional evidence **does not distinguish between no
incremental effect and a small positive effect of the size the broad literature
reports**. Four analyses establish that boundary rather than asserting it:

1. **Detection scale** — separates the `t>3` significance threshold (0.019) from the
   80%-power effect (0.025). These are different quantities and conflating them is
   the usual error in applied power statements.
2. **Signal injection** — planting a factor of known size: 0/3 seeds recover an
   effect at the observed 0.003, 1/3 at 0.019, 3/3 at 0.030. The transition brackets
   the analytic 80%-power figure from both sides.
3. **Scaling calibration** (`scripts/n_scaling.py`) — the standard `1/√N` assumption
   behind every power extrapolation of this kind is **tested on the data and fails**.
   The measured exponent is `N^-0.29`, not `N^-0.50`, and a two-component fit
   `SE(N)² = a + b/N` (R² = 0.96) reveals a floor of 0.015 on the detection threshold
   that no cross-section size removes, because time-variation in the cross-sectional
   IC does not shrink with the number of names. Two placebos — i.i.d. predictions and
   within-date permutations — recover `N^-0.56` with no floor, so the flattening is a
   property of the signal, not of the procedure.
4. **Volatility detectability positive control** — the identical pipeline detects a
   large signal on realised volatility (≈0.036 IC lift, >10 s.e., 158/162 cells
   clearing `t>3`, against 7/162 for returns). This shows the pipeline is *blind*
   rather than *broken*. It does not show the return estimate is correct.

Two effects lie above the measured floor and the same cross-section resolves both:
the volatility lift (0.036), and the return IC in a pre-specified
**high-variance-risk-premium** state (0.019–0.031, CI excluding zero in all four
cells, differences clearing a Bonferroni bar in three of four). A meta-analysis pools
to a small sub-threshold lift (+0.0029, calendar-block `p = 0.051`).

> **Note on an earlier version.** A previous draft extrapolated with `1/√N` and
> concluded that a 0.004 effect would become detectable at roughly 1,400 names. The
> scaling calibration retracts that. Under the measured law the same effect needs on
> the order of 18,000 names, and under the variance decomposition it is not reachable
> by adding names at all. More names alone would not resolve the unconditional
> question; more calendar time or a lower-floor estimator would.

---

## Layout

```
config/        feature_definitions.yaml (feature-set ablations), settings.yaml
run_pipeline.py
src/
  data/        WRDS / Yahoo ingestion, option + stock cleaning, source merge
  features/    scalar features, SVI surface fit, (Δ,τ) grid, BKM moments, targets
  models/      OLS / ElasticNet / XGBoost workhorses, benchmarks, FFNN
  evaluation/  walk-forward, HAC/Harvey–Liu–Zhu, deflated Sharpe, PBO, regimes
  trading/     dollar-neutral long/short backtest
  explainability/ TreeSHAP
scripts/       analysis + figure generation (see "Reproduce")
results/
  tables/      saved metrics (.txt/.csv) — the canonical, version-controlled numbers
  figures/     generated figures
```

## Install

```bash
pip install -r requirements.txt   # pandas, numpy, scipy, scikit-learn, xgboost, pyarrow, matplotlib
```

## Data

End-of-day option files come from the
[`philippdubach/options-data`](https://github.com/philippdubach/options-data)
community repository, which was public when accessed and has since been made private;
returns, market cap, earnings, VIX/VIX3M, the T-bill rate and sector ETF returns come
from standard public sources.

Not tracked in git (size): the built feature panels under `data/features/**.parquet`,
and the two large out-of-sample prediction files
`results/tables/full_matrix_oos_xgb.parquet` and `vol_oos_xgb.parquet`
(~50 MB each). The small result tables under `results/tables/` **are** tracked and
are enough to regenerate every figure and number.

⚠️ **`scripts/n_scaling.py` needs `full_matrix_oos_xgb.parquet`,** which is not
tracked. Regenerate it with `scripts/run_full_matrix.py` before running the scaling
calibration. Everything else runs from the tracked tables.

## Reproduce

All models are seeded (`random_state=42`) and deterministic — re-running reproduces
the saved per-window IC exactly (`scripts/reproducibility_check.py`), and the
per-window ICs rebuild from the saved out-of-sample predictions to machine
precision, which is what makes the subsampling calibration exact.

| Reported result | Script(s) |
|---|---|
| Consolidated IC benchmark (both targets, all sets × models) | `run_full_matrix.py`, `run_vol_matrix.py`, `posthoc_stats.py` |
| Returns marginal lift over the Set-A control | `run_full_matrix.py`, `posthoc_stats.py` |
| Volatility positive control + returns-vs-vol dissociation | `run_vol_matrix.py`, `vol_dissociation.py` |
| Detection scale + shippable calculator | `power_analysis.py`, `power_calculator.py` |
| **Scaling calibration, placebos, required-N under both criteria** | `n_scaling.py` |
| Signal-injection recovery | `signal_injection.py` |
| Regime conditioning + meta-analysis (joint calendar-block bootstrap) | `return_extras.py` |
| Post-hoc corrections (HAC, HLZ `t>3`, deflated Sharpe, PBO) | `posthoc_stats.py` |
| TreeSHAP attribution | `treeshap.py` |
| Economics: dollar P&L, deflated Sharpe, PBO, ETF-free IC | `etf_free_econ.py`, `posthoc_stats.py` |
| Run-to-run reproducibility | `reproducibility_check.py` |
| All paper figures | `make_paper_figures.py` |

Scaling calibration, including both validation placebos:

```bash
python3 scripts/n_scaling.py --seeds 40                    # canonical run
python3 scripts/n_scaling.py --seeds 40 --placebo noise    # must give ≈ -0.5, no floor
python3 scripts/n_scaling.py --seeds 20 --placebo shuffle  # must give ≈ -0.5, no floor
python3 scripts/n_scaling.py --seeds 40 --drop-etfs        # sensitivity: 57 single names
python3 scripts/n_scaling.py --seeds 40                    # re-run LAST to restore artefacts
```

The placebo and sensitivity runs overwrite the same output files, so finish with the
canonical run or the saved tables will not match the reported numbers.

## Caveats

- **Universe selection** used membership and liquidity over the sample. That is a
  survivorship bias and it pushes reported ICs upward, so the return null is
  conservative — the bias works against the paper's reading, not for it.
- **The three ETFs (SPY/QQQ/IWM) are prediction targets**, not a feature source. They
  enter every return-IC cross-section on the same footing as the single names, and
  are excluded only from the tradeable long/short book. No single-name feature is
  built from them; the market-level block uses the VIX, VIX3M, the T-bill rate and
  the ten GICS *sector* ETFs, none of which is in the prediction universe. The
  ETF-free IC is reported alongside the headline.
- **Preprocessing is fit once on a fixed 60% temporal split**, not per walk-forward
  window, so about 37% of panel rows sit in test blocks inside the window used to fit
  the winsorisation and imputation statistics. The footprint is small (0.70% of cells
  clipped, 2.31% imputed) and the direction is optimistic, so it cannot produce the
  sub-threshold estimate reported here. Refitting per window is a pending extension.
- **No tradeable-strategy claim is made.** The headline long/short does not survive
  deflation (Sharpe 1.1504 vs an `SR₀` bar of 1.1529) and the probability of backtest
  overfitting is 0.571. The robust object is the rank IC, not the dollar Sharpe.
- **Short-sale costs are not identified.** Without equity-lending fee data this study
  cannot separate information-based option signals from borrow-cost frictions, and no
  proxy is substituted for them.
