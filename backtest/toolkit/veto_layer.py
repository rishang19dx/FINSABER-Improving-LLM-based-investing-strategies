"""
ARIMA–LLM Veto Layer
=====================

Combines an ARIMA directional forecast (price-only) with an LLM
position signal (inferred from its equity curve) to produce a hybrid
trading strategy.

The veto rules are asymmetric by design:
  • LLM=BUY  + ARIMA=UP   → BUY   (confirmed)
  • LLM=BUY  + ARIMA=DOWN → HOLD  (vetoed — ARIMA catches false buys)
  • LLM=SELL               → SELL  (always pass through)
  • LLM=HOLD               → HOLD  (always pass through)

This exploits complementary failures: ARIMA provides regime-aware
trend filtering while the LLM provides sentiment-driven sell signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Signal constants ──────────────────────────────────────────────────────
BUY = 1
HOLD = 0
SELL = -1


# ═══════════════════════════════════════════════════════════════════════════
# 1. Position Inference (return-ratio based)
# ═══════════════════════════════════════════════════════════════════════════

def infer_position_from_equity(
    equity: np.ndarray,
    prices: np.ndarray,
    ratio_threshold: float = 0.3,
    **kwargs,
) -> np.ndarray:
    """Infer daily position state (IN=1 / OUT=0) from an equity curve.

    Uses per-day return ratios: if the equity return tracks the stock
    price return (ratio > threshold), the strategy is invested.
    When in cash, equity returns are near-zero regardless of price
    movement.

    This replaces the earlier correlation-based approach which had a
    ~24% misclassification rate due to window-averaging effects.

    Parameters
    ----------
    equity : array-like
        Daily equity values (length T).
    prices : array-like
        Daily close prices for the same dates (length T).
    ratio_threshold : float
        Minimum eq_return / px_return ratio to classify as IN.

    Returns
    -------
    np.ndarray
        Binary array of length T: 1=IN, 0=OUT.
    """
    eq = np.asarray(equity, dtype=float)
    px = np.asarray(prices, dtype=float)

    n = min(len(eq), len(px))
    eq, px = eq[:n], px[:n]

    # Daily returns
    eq_ret = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1e-12)
    px_ret = np.diff(px) / np.where(px[:-1] != 0, px[:-1], 1e-12)

    position = np.zeros(n, dtype=int)

    for i in range(len(eq_ret)):
        if abs(px_ret[i]) < 1e-6:
            # Price didn't move — can't tell, carry forward
            position[i + 1] = position[i]
        elif abs(eq_ret[i]) < abs(px_ret[i]) * 0.1:
            # Equity barely moved vs price → OUT (in cash)
            position[i + 1] = 0
        else:
            # Check if equity tracked price direction
            ratio = eq_ret[i] / px_ret[i] if px_ret[i] != 0 else 0
            position[i + 1] = 1 if ratio > ratio_threshold else 0

    return position


def positions_to_signals(positions: np.ndarray) -> np.ndarray:
    """Convert position states to trading signals.

    Transitions:
        OUT → IN  = BUY
        IN  → OUT = SELL
        same      = HOLD
    """
    signals = np.full(len(positions), HOLD, dtype=int)
    for i in range(1, len(positions)):
        if positions[i] == 1 and positions[i - 1] == 0:
            signals[i] = BUY
        elif positions[i] == 0 and positions[i - 1] == 1:
            signals[i] = SELL
    return signals


# ═══════════════════════════════════════════════════════════════════════════
# 2. ARIMA Signal Reconstruction
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_arima_signals(
    prices: np.ndarray,
    order: tuple = (5, 1, 0),
    train_period: int = 756,
) -> np.ndarray:
    """Re-run ARIMA forecasting to produce daily directional signals.

    Matches the logic in ``arima_predictor.py``: for each day, forecast
    tomorrow's price and compare to today's price.

    Parameters
    ----------
    prices : array-like
        Full daily close prices (including training history).
    order : tuple
        ARIMA (p, d, q) order.
    train_period : int
        Number of historical observations to train on.

    Returns
    -------
    np.ndarray
        Array of signals: BUY(+1) if forecast > current, SELL(-1) if
        forecast < current, HOLD(0) if equal.  Length = len(prices).
    """
    px = np.asarray(prices, dtype=float)
    signals = np.full(len(px), HOLD, dtype=int)

    if len(px) < train_period + 10:
        return signals

    # Fit initial model on training period
    train = pd.Series(px[:train_period])
    try:
        model = sm.tsa.ARIMA(train, order=order).fit()
    except Exception:
        return signals

    for i in range(train_period, len(px)):
        try:
            # Forecast one step ahead
            forecast = model.forecast(steps=1).values[0]
            current = px[i]

            if forecast > current:
                signals[i] = BUY
            elif forecast < current:
                signals[i] = SELL
            else:
                signals[i] = HOLD

            # Update model with latest observation
            model = model.apply(pd.Series(px[: i + 1]))
        except Exception:
            signals[i] = HOLD

    return signals


# ═══════════════════════════════════════════════════════════════════════════
# 3. Veto Logic (position-level)
# ═══════════════════════════════════════════════════════════════════════════

def apply_veto_position(
    llm_signals: np.ndarray,
    arima_positions: np.ndarray,
    mode: str = "arima_veto",
) -> np.ndarray:
    """Combine LLM signals with ARIMA position state using the veto rule.

    Unlike signal-level veto, this checks whether the vetoing strategy
    is currently IN position (bullish) when the other fires a BUY.
    This avoids the problem of requiring two BUY transitions on the
    exact same day.

    Parameters
    ----------
    llm_signals : array
        Daily signal array (BUY/HOLD/SELL) from position transitions.
    arima_positions : array
        Daily position state (1=IN, 0=OUT) from the ARIMA equity curve.
    mode : str
        ``"arima_veto"``  — ARIMA position vetoes LLM buys
        ``"reverse_veto"`` — LLM position vetoes ARIMA buys

    Returns
    -------
    np.ndarray
        Combined signal array.
    """
    n = min(len(llm_signals), len(arima_positions))
    combined = np.full(n, HOLD, dtype=int)

    if mode == "arima_veto":
        # When LLM says BUY, check if ARIMA is currently IN position
        # (i.e., ARIMA's trend model is bullish on this stock)
        for i in range(n):
            llm = llm_signals[i]

            if llm == BUY:
                # ARIMA confirms if it's currently invested (bullish)
                combined[i] = BUY if arima_positions[i] == 1 else HOLD
            elif llm == SELL:
                combined[i] = SELL  # LLM sells always pass
            else:
                combined[i] = HOLD

    elif mode == "reverse_veto":
        # When ARIMA transitions to BUY, check if LLM is currently IN
        # Convert arima_positions to signals for transition detection
        arima_sigs = positions_to_signals(arima_positions)
        # Infer LLM position state from its signals
        llm_pos = np.zeros(n, dtype=int)
        for i in range(n):
            if llm_signals[i] == BUY:
                llm_pos[i:] = 1
            elif llm_signals[i] == SELL:
                llm_pos[i:] = 0

        for i in range(n):
            if arima_sigs[i] == BUY:
                combined[i] = BUY if llm_pos[i] == 1 else HOLD
            elif arima_sigs[i] == SELL:
                combined[i] = SELL  # ARIMA sells always pass
            else:
                combined[i] = HOLD
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return combined


# ═══════════════════════════════════════════════════════════════════════════
# 4. Equity Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_equity(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.001,
) -> np.ndarray:
    """Simulate an equity curve from daily signals and prices.

    Parameters
    ----------
    prices : array
        Daily close prices.
    signals : array
        Daily signals (BUY/HOLD/SELL).
    initial_cash : float
        Starting capital.
    commission_rate : float
        Round-trip commission as fraction of trade value.

    Returns
    -------
    np.ndarray
        Daily equity values.
    """
    n = min(len(prices), len(signals))
    equity = np.zeros(n)
    cash = initial_cash
    shares = 0
    in_position = False

    for i in range(n):
        price = prices[i]

        if signals[i] == BUY and not in_position and price > 0:
            # Buy as many shares as possible
            max_shares = int(cash / (price * (1 + commission_rate)))
            if max_shares > 0:
                cost = max_shares * price * (1 + commission_rate)
                cash -= cost
                shares = max_shares
                in_position = True

        elif signals[i] == SELL and in_position and price > 0:
            # Sell all shares
            revenue = shares * price * (1 - commission_rate)
            cash += revenue
            shares = 0
            in_position = False

        # Mark to market
        equity[i] = cash + shares * price

    return equity
