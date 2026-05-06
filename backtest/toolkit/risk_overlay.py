"""
Rolling Sharpe Risk Overlay
===========================

Post-hoc risk management overlay that takes an existing equity curve and
simulates what would have happened with dynamic position sizing based on
rolling Sharpe and a max-drawdown circuit breaker.

The overlay scales position exposure between 0 % and 100 % using the
rolling annualised Sharpe over a lookback window:

    scale_t = clip( (RS_t - τ_lower) / (τ_upper - τ_lower),  0, 1 )

Additionally, a max-drawdown circuit breaker forces scale = 0 when the
cumulative drawdown from peak exceeds a threshold.

This module operates entirely post-hoc on equity curves — no backtests
need to be re-run.
"""

import numpy as np
import pandas as pd
from backtest.toolkit import metrics
from backtest.toolkit.psr import compute_psr_from_returns


def rolling_sharpe(returns: np.ndarray, window: int = 30,
                   risk_free_rate: float = 0.03) -> np.ndarray:
    """Compute rolling annualised Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Daily returns.
    window : int
        Lookback window in trading days.
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    np.ndarray
        Rolling Sharpe for each day.  First ``window-1`` values are NaN.
    """
    returns = np.asarray(returns, dtype=float)
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    n = len(returns)
    rs = np.full(n, np.nan)

    for i in range(window - 1, n):
        w = returns[i - window + 1: i + 1]
        excess = w - daily_rf
        std = np.std(w, ddof=1)
        if std == 0:
            rs[i] = 0.0
        else:
            rs[i] = (np.mean(excess) / std) * np.sqrt(252)
    return rs


def apply_risk_overlay(equity_curve: np.ndarray,
                       sharpe_window: int = 30,
                       sharpe_upper: float = 0.0,
                       sharpe_lower: float = -1.0,
                       max_drawdown_pct: float = 30.0,
                       risk_free_rate: float = 0.03) -> dict:
    """Apply rolling-Sharpe position scaling to an existing equity curve.

    Parameters
    ----------
    equity_curve : array-like
        Daily portfolio values (e.g. starting at 100,000).
    sharpe_window : int
        Rolling window in trading days (default 30).
    sharpe_upper : float
        Rolling Sharpe above this → full position (default 0.0).
    sharpe_lower : float
        Rolling Sharpe below this → zero position (default -1.0).
    max_drawdown_pct : float
        Circuit breaker: zero position when drawdown exceeds this (default 30 %).
    risk_free_rate : float
        Annual risk-free rate (default 0.03).

    Returns
    -------
    dict
        ``adjusted_equity``, ``scales``, ``rolling_sharpe``,
        ``raw_metrics``, ``adjusted_metrics``.
    """
    equity = np.asarray(equity_curve, dtype=float)
    daily_returns = np.diff(equity) / equity[:-1]

    # Rolling Sharpe
    rs = rolling_sharpe(daily_returns, window=sharpe_window,
                        risk_free_rate=risk_free_rate)

    # Position scales
    scales = np.ones(len(daily_returns))
    peak = equity[0]
    adjusted_equity = [equity[0]]

    for i in range(len(daily_returns)):
        # Update peak from adjusted equity
        if adjusted_equity[-1] > peak:
            peak = adjusted_equity[-1]

        # Rolling Sharpe scale
        if np.isnan(rs[i]):
            scale = 1.0  # Not enough history yet
        elif rs[i] >= sharpe_upper:
            scale = 1.0
        elif rs[i] <= sharpe_lower:
            scale = 0.0
        else:
            scale = (rs[i] - sharpe_lower) / (sharpe_upper - sharpe_lower)

        # Max drawdown circuit breaker
        if peak > 0:
            current_dd = (adjusted_equity[-1] - peak) / peak
            if current_dd <= -(max_drawdown_pct / 100):
                scale = 0.0

        scales[i] = scale
        adjusted_return = daily_returns[i] * scale
        adjusted_equity.append(adjusted_equity[-1] * (1 + adjusted_return))

    adjusted_equity = np.array(adjusted_equity)

    # Compute metrics for both raw and adjusted
    raw_daily = daily_returns
    adj_daily = np.diff(adjusted_equity) / adjusted_equity[:-1]

    raw_metrics = _compute_summary_metrics(raw_daily, equity, risk_free_rate)
    adj_metrics = _compute_summary_metrics(adj_daily, adjusted_equity, risk_free_rate)

    return {
        "adjusted_equity": adjusted_equity,
        "scales": scales,
        "rolling_sharpe": rs,
        "raw_metrics": raw_metrics,
        "adjusted_metrics": adj_metrics,
    }


def _compute_summary_metrics(daily_returns: np.ndarray,
                              equity: np.ndarray,
                              risk_free_rate: float = 0.03) -> dict:
    """Compute a standard set of summary metrics from daily returns."""
    returns_series = pd.Series(daily_returns)
    n = len(daily_returns)

    if n < 2 or np.std(daily_returns) == 0:
        return {
            "total_return": 0.0, "annual_return": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "omega": 0.0,
            "max_drawdown_pct": 0.0, "psr": 0.0, "min_trl_years": float("inf"),
        }

    total_return = (equity[-1] / equity[0]) - 1
    annual_return = (1 + total_return) ** (252 / n) - 1

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd_pct = abs(drawdowns.min()) * 100

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = daily_returns - daily_rf
    sharpe = (np.mean(excess) / np.std(daily_returns, ddof=1)) * np.sqrt(252)

    sortino = metrics.calculate_sortino_ratio(returns_series, risk_free_rate)
    calmar = metrics.calculate_calmar_ratio(annual_return, max_dd_pct)
    omega = metrics.calculate_omega_ratio(returns_series, threshold=daily_rf)
    psr_result = compute_psr_from_returns(daily_returns, benchmark_sr=0.0)

    return {
        "total_return": round(total_return, 4),
        "annual_return": round(annual_return, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "omega": round(omega, 4),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "psr": psr_result["psr"],
        "min_trl_years": psr_result["min_trl_years"],
    }
