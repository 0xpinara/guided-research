# guided-research

**Statistical Resolution in Option-Implied Return Prediction**
*A power-aware evaluation protocol for constrained cross-sections.*

Pınar Aksoy · METU Computer Engineering · first author
Manuscript under review at *Borsa Istanbul Review*.

---

## The question

Large-universe studies already establish that some option characteristics carry
incremental information for future stock returns. This project asks a different
question: **when the observable option cross-section is small, which predictive effects
can a design of that size actually resolve, and how do you tell a weak effect apart from
inadequate statistical resolution?**

Worked on a fully reproducible panel of 134,368 ticker-days over 60 U.S. tickers,
2016–2024, with a median daily cross-section of 59 names.

## Headline

The option surface's marginal contribution to return ranking is **0.002–0.004 rank IC**.
The detection threshold at the Harvey–Liu–Zhu `t > 3` hurdle is **0.019**, and 80% power
needs about **0.025**. The estimate is five to ten times below the design's resolution,
so the unconditional evidence **does not distinguish between no incremental effect and a
small positive effect of the size the broad literature reports**.

Four analyses establish that boundary rather than asserting it.

| | Analysis | Result |
|---|---|---|
| 1 | **Detection scale** | Separates the `t > 3` significance threshold (0.019) from the 80%-power effect (0.025). Conflating the two is the usual error in applied power statements. |
| 2 | **Signal injection** | Planting a factor of known size: 0/3 seeds recover at the observed 0.003, 1/3 at 0.019, 3/3 at 0.030. The transition brackets the analytic figure from both sides. |
| 3 | **Scaling calibration** | The standard `1/√N` assumption behind every power extrapolation of this kind is tested on the data and **fails**. Measured exponent `N^-0.29`, not `N^-0.50`, with a floor of 0.015 that no cross-section size removes. Two placebos recover `N^-0.56` with no floor, so the flattening is a property of the signal, not the procedure. |
| 4 | **Volatility positive control** | The identical pipeline detects a large signal on realised volatility (≈0.036 IC lift, >10 s.e., 158/162 cells clearing `t > 3`, against 7/162 for returns). The pipeline is *blind*, not *broken*. It does not show the return estimate is correct. |

Two effects lie above the measured floor and the same cross-section resolves both: the
volatility lift (0.036), and the return IC in a pre-specified **high-variance-risk-premium**
state (0.019–0.031, CI excluding zero in all four cells).

## Methods

Walk-forward out-of-sample evaluation · rank IC · Harvey–Liu–Zhu multiple-testing
threshold · Deflated Sharpe Ratio · Probability of Backtest Overfitting · power analysis
specified before the fact · placebo controls using i.i.d. predictions and within-date
permutations

## Code and saved results

The full study — the walk-forward evaluation, the multiple-testing corrections, the
volatility positive control, the power analysis, and the saved result tables that
reproduce every number and figure in the paper — lives in:

### → [`guided/options_research/`](guided/options_research/)

See that folder's [README](guided/options_research/README.md) for the repository layout,
install instructions, and a table mapping each reported number to the script that
produces it.
