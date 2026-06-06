"""Strategy backtesting with transaction costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.io_helpers import RESULTS_DIR
from src.trading.frictions import effective_spread_cost, annual_short_fee_cost

log = setup_logger(__name__)


def backtest_long_flat(
    dates: np.ndarray,
    tickers: np.ndarray,
    actual_returns: np.ndarray,
    pred_direction: np.ndarray,
    transaction_cost_bps: float = 5,
    initial_capital: float = 10_000,
) -> pd.DataFrame:
    """Backtest a long/flat strategy across tickers.

    Go long when predicted up, flat when predicted down.
    Equal-weight across tickers that are long on any given day.

    Parameters
    ----------
    dates : ndarray of dates.
    tickers : ndarray of ticker strings.
    actual_returns : ndarray of actual returns.
    pred_direction : ndarray of predicted direction (1=up, 0=down).

    Returns
    -------
    DataFrame with daily portfolio returns and cumulative equity.
    """
    tc = transaction_cost_bps / 10_000

    df = pd.DataFrame({
        "date": dates,
        "ticker": tickers,
        "ret": actual_returns,
        "signal": pred_direction,
    })

    # Per-ticker strategy returns
    results = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        position = grp["signal"].values.astype(float)
        trades = np.abs(np.diff(position, prepend=0))
        strat_ret = position * grp["ret"].values - trades * tc
        grp["strategy_return"] = strat_ret
        grp["position"] = position
        results.append(grp)

    df = pd.concat(results)

    # Portfolio: equal-weight average across all active tickers per day
    portfolio = df.groupby("date").agg(
        portfolio_return=("strategy_return", "mean"),
        n_long=("position", "sum"),
        n_tickers=("ticker", "count"),
    ).reset_index().sort_values("date")

    portfolio["cumulative_return"] = (1 + portfolio["portfolio_return"]).cumprod()
    portfolio["equity"] = initial_capital * portfolio["cumulative_return"]

    # Buy-and-hold benchmark
    bnh = df.groupby("date")["ret"].mean()
    portfolio["bnh_cumulative"] = (1 + bnh).cumprod().values

    # Summary stats
    ann_ret = portfolio["portfolio_return"].mean() * 252
    ann_vol = portfolio["portfolio_return"].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    peak = portfolio["equity"].cummax()
    drawdown = (portfolio["equity"] - peak) / peak
    max_dd = drawdown.min()

    log.info(
        "Backtest: ann_return=%.2f%%, sharpe=%.2f, max_dd=%.2f%%, avg %.1f long/day",
        ann_ret * 100, sharpe, max_dd * 100, portfolio["n_long"].mean(),
    )

    return portfolio


def _performance_summary(portfolio: pd.DataFrame, return_col: str = "portfolio_return",
                         periods_per_year: float = 252.0) -> dict:
    """Compute standard strategy metrics from a daily return series."""
    r = portfolio[return_col].astype(float)
    equity = (1 + r.clip(lower=-0.99)).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    ann_ret = (1 + r.mean()) ** periods_per_year - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return {
        "n_periods": int(len(r)),
        "mean_return": float(r.mean()),
        "annualized_return": float(ann_ret),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "total_return": float(equity.iloc[-1] - 1),
        "win_rate": float((r > 0).mean()),
    }


def backtest_long_short_deciles(
    dates: np.ndarray,
    tickers: np.ndarray,
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    quoted_spread: np.ndarray | None = None,
    top_n: int = 6,
    bottom_n: int = 6,
    effective_spread_fraction: float = 0.15,
    short_fee_bps_annual: float = 50.0,
    horizon: int = 3,
    rebalance_every: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Market-neutral long/short decile strategy.

    Each rebalance date, rank tickers by predicted return.  Equal-weight the
    top ``top_n`` names long and the bottom ``bottom_n`` names short.  Deduct
    Bali-style effective spread costs on turnover and a Muravyev-style
    annualised borrow fee on short exposure.

    ``rebalance_every`` controls how often positions are reopened, in trading
    days.  For a clean non-overlapping backtest of an ``h``-day-forward target
    it must equal ``horizon`` (the default when left as ``None``): trades are
    then spaced ``h`` days apart so the realised ``h``-day returns do not
    overlap, and the Sharpe annualisation (``252 / rebalance_every``) is valid.
    Setting it to 1 reproduces daily rebalancing with overlapping returns.
    """
    if rebalance_every is None:
        rebalance_every = max(int(horizon), 1)
    rebalance_every = max(int(rebalance_every), 1)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "ticker": tickers,
        "ret": actual_returns,
        "pred": predicted_returns,
        "spread": quoted_spread if quoted_spread is not None else np.nan,
    }).dropna(subset=["date", "ticker", "ret", "pred"]).drop_duplicates(subset=["date", "ticker"])

    # Trade only on every ``rebalance_every``-th trading date so an h-day
    # target produces non-overlapping holding periods when rebalance_every == h.
    all_dates = np.sort(df["date"].unique())
    keep_dates = set(all_dates[::rebalance_every])
    df = df[df["date"].isin(keep_dates)]

    prev_pos: dict[str, float] = {}
    rows = []
    periods_per_year = 252 / max(rebalance_every, 1)

    for dt, grp in df.groupby("date", sort=True):
        grp = grp.sort_values("pred", ascending=False).copy()
        if len(grp) < top_n + bottom_n:
            continue

        longs = set(grp.head(top_n)["ticker"])
        shorts = set(grp.tail(bottom_n)["ticker"])
        curr_pos = {t: 0.0 for t in set(prev_pos) | set(grp["ticker"])}
        for t in longs:
            curr_pos[t] = 1.0 / top_n
        for t in shorts:
            curr_pos[t] = -1.0 / bottom_n

        g = grp.set_index("ticker")
        pnl = 0.0
        cost = 0.0
        turnover = 0.0
        long_ret = 0.0
        short_ret = 0.0
        for t, pos in curr_pos.items():
            old = prev_pos.get(t, 0.0)
            delta = pos - old
            turnover += abs(delta)
            if t in g.index:
                ret = float(g.loc[t, "ret"])
                spread = float(g.loc[t, "spread"]) if "spread" in g.columns else np.nan
                pnl += pos * ret
                cost += effective_spread_cost(
                    np.array([delta]), np.array([spread]),
                    effective_spread_fraction=effective_spread_fraction,
                )[0]
                if pos > 0:
                    long_ret += pos * ret
                elif pos < 0:
                    short_ret += -pos * ret

        short_exposure = sum(abs(v) for v in curr_pos.values() if v < 0)
        borrow_cost = annual_short_fee_cost(
            short_exposure, annual_fee_bps=short_fee_bps_annual,
            periods_per_year=periods_per_year,
        )
        portfolio_return = pnl - cost - borrow_cost
        rows.append({
            "date": dt,
            "portfolio_return": portfolio_return,
            "long_leg_return": long_ret,
            "short_leg_return": short_ret,
            "gross_spread_return": long_ret - short_ret,
            "spread_cost": cost,
            "short_fee_cost": borrow_cost,
            "turnover": turnover,
            "n_long": len(longs),
            "n_short": len(shorts),
        })
        prev_pos = curr_pos

    portfolio = pd.DataFrame(rows)
    if portfolio.empty:
        return portfolio, {}
    portfolio["cumulative_return"] = (1 + portfolio["portfolio_return"].clip(lower=-0.99)).cumprod()
    summary = _performance_summary(portfolio, periods_per_year=periods_per_year)
    summary.update({
        "strategy": "long_short_decile",
        "top_n": top_n,
        "bottom_n": bottom_n,
        "effective_spread_fraction": effective_spread_fraction,
        "short_fee_bps_annual": short_fee_bps_annual,
        "horizon": horizon,
        "rebalance_every": rebalance_every,
        "avg_turnover": float(portfolio["turnover"].mean()),
        "avg_spread_cost": float(portfolio["spread_cost"].mean()),
        "avg_short_fee_cost": float(portfolio["short_fee_cost"].mean()),
    })
    return portfolio, summary


def backtest_long_flat_hurdle(
    dates: np.ndarray,
    tickers: np.ndarray,
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    quoted_spread: np.ndarray | None = None,
    hurdle_bps: float = 10.0,
    effective_spread_fraction: float = 0.15,
    horizon: int = 3,
    rebalance_every: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Long/flat baseline that only trades when predicted return clears a hurdle.

    ``rebalance_every`` matches :func:`backtest_long_short_deciles`: positions
    are reopened every ``rebalance_every`` trading days (default ``horizon``),
    giving non-overlapping ``h``-day holds and a valid ``252 / rebalance_every``
    annualisation.
    """
    if rebalance_every is None:
        rebalance_every = max(int(horizon), 1)
    rebalance_every = max(int(rebalance_every), 1)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "ticker": tickers,
        "ret": actual_returns,
        "pred": predicted_returns,
        "spread": quoted_spread if quoted_spread is not None else np.nan,
    }).dropna(subset=["date", "ticker", "ret", "pred"]).drop_duplicates(subset=["date", "ticker"])

    all_dates = np.sort(df["date"].unique())
    keep_dates = set(all_dates[::rebalance_every])
    df = df[df["date"].isin(keep_dates)]

    threshold = hurdle_bps / 10_000
    prev_pos: dict[str, float] = {}
    rows = []
    periods_per_year = 252 / max(rebalance_every, 1)

    for dt, grp in df.groupby("date", sort=True):
        active = grp[grp["pred"] > threshold]
        n_active = len(active)
        active_set = set(active["ticker"])
        curr_pos = {t: 0.0 for t in set(prev_pos) | set(grp["ticker"])}
        if n_active > 0:
            for t in active_set:
                curr_pos[t] = 1.0 / n_active

        g = grp.set_index("ticker")
        pnl = 0.0
        cost = 0.0
        turnover = 0.0
        for t, pos in curr_pos.items():
            old = prev_pos.get(t, 0.0)
            delta = pos - old
            turnover += abs(delta)
            if t in g.index:
                pnl += pos * float(g.loc[t, "ret"])
                spread = float(g.loc[t, "spread"]) if "spread" in g.columns else np.nan
                cost += effective_spread_cost(
                    np.array([delta]), np.array([spread]),
                    effective_spread_fraction=effective_spread_fraction,
                )[0]

        rows.append({
            "date": dt,
            "portfolio_return": pnl - cost,
            "gross_return": pnl,
            "spread_cost": cost,
            "turnover": turnover,
            "n_long": n_active,
        })
        prev_pos = curr_pos

    portfolio = pd.DataFrame(rows)
    if portfolio.empty:
        return portfolio, {}
    portfolio["cumulative_return"] = (1 + portfolio["portfolio_return"].clip(lower=-0.99)).cumprod()
    summary = _performance_summary(portfolio, periods_per_year=periods_per_year)
    summary.update({
        "strategy": "long_flat_hurdle",
        "hurdle_bps": hurdle_bps,
        "effective_spread_fraction": effective_spread_fraction,
        "horizon": horizon,
        "rebalance_every": rebalance_every,
        "avg_turnover": float(portfolio["turnover"].mean()),
        "avg_spread_cost": float(portfolio["spread_cost"].mean()),
    })
    return portfolio, summary


def save_backtest(portfolio: pd.DataFrame, name: str) -> None:
    """Save backtest results."""
    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    portfolio.to_csv(out_dir / f"backtest_{name}.csv", index=False)
