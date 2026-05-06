"""
Regime Signal Generator for LLM Prompt Conditioning
=====================================================

Generates real-time regime and volatility context signals for injection
into FinMem and FinAgent LLM prompts.  Operates on the same price data
already available during backtesting.

The signal includes:
- 20-day rolling volatility (annualised)
- Volatility regime classification (Low / Normal / High / Extreme)
- 50/200-day SMA trend regime (Bull / Bear / Sideways)
- Risk guidance text calibrated to the current regime

Usage
-----
    from backtest.toolkit.regime_signal import RegimeSignalGenerator

    rsg = RegimeSignalGenerator()
    signal = rsg.compute_signal(price_series, current_date)
    prompt_text = rsg.build_prompt_injection(signal)
"""

import numpy as np
import pandas as pd
from datetime import date
from typing import Union, Dict


class RegimeSignalGenerator:
    """Generate regime and volatility signals from price history.

    Parameters
    ----------
    vol_window : int
        Rolling window for annualised volatility (default 20 days).
    sma_short : int
        Short SMA for trend detection (default 50 days).
    sma_long : int
        Long SMA for trend detection (default 200 days).
    vol_thresholds : dict
        Percentile thresholds for volatility regimes.
    """

    # Volatility regime thresholds (annualised)
    DEFAULT_VOL_THRESHOLDS = {
        "low": 0.12,       # < 12% → Low
        "normal": 0.20,    # 12-20% → Normal
        "high": 0.35,      # 20-35% → High
        # > 35% → Extreme
    }

    # Risk guidance per regime combination
    RISK_GUIDANCE = {
        ("Bull", "Low"): (
            "Market is in a confirmed uptrend with unusually low volatility. "
            "This is historically the safest environment for long positions, "
            "but be aware that low-volatility periods can end abruptly."
        ),
        ("Bull", "Normal"): (
            "Market is trending upward with normal volatility. "
            "This is a healthy bullish environment suitable for maintaining "
            "or incrementally adding to long positions."
        ),
        ("Bull", "High"): (
            "Market trend is up but volatility is elevated. "
            "Consider reducing position sizes. "
            "Large price swings may trigger stop-losses despite the positive trend."
        ),
        ("Bull", "Extreme"): (
            "WARNING: Extreme volatility despite bullish trend. "
            "This often occurs near market tops or during panic recoveries. "
            "Strongly consider reducing exposure or hedging."
        ),
        ("Sideways", "Low"): (
            "Market is range-bound with low volatility. "
            "This is a neutral environment. Avoid large directional bets. "
            "Focus on capital preservation."
        ),
        ("Sideways", "Normal"): (
            "Market is range-bound with normal volatility. "
            "No clear trend — avoid conviction positions. "
            "This environment historically favours mean-reversion strategies."
        ),
        ("Sideways", "High"): (
            "Market has no clear trend but volatility is high. "
            "This is a difficult environment — random price swings dominate. "
            "Reduce position sizes significantly."
        ),
        ("Sideways", "Extreme"): (
            "WARNING: No trend and extreme volatility — a crisis-like environment. "
            "Strongly consider moving to cash or minimal positions."
        ),
        ("Bear", "Low"): (
            "Market is in a downtrend but volatility is low — a slow bleed. "
            "Avoid buying dips. The trend may continue for an extended period."
        ),
        ("Bear", "Normal"): (
            "Market is in a downtrend with normal volatility. "
            "Avoid long positions. Capital preservation is the priority. "
            "Consider selling any remaining holdings."
        ),
        ("Bear", "High"): (
            "Market is in a downtrend with high volatility — a sell-off phase. "
            "DO NOT buy. Rapid losses are likely. "
            "If holding positions, consider stop-losses at -5% from current levels."
        ),
        ("Bear", "Extreme"): (
            "CRITICAL WARNING: Bear market with extreme volatility — potential crash conditions. "
            "SELL all discretionary positions. Move to cash immediately. "
            "Historical data shows that LLM strategies lose significantly in this regime."
        ),
    }

    def __init__(
        self,
        vol_window: int = 20,
        sma_short: int = 50,
        sma_long: int = 200,
        vol_thresholds: dict = None,
    ):
        self.vol_window = vol_window
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.vol_thresholds = vol_thresholds or self.DEFAULT_VOL_THRESHOLDS

    def compute_signal(
        self,
        prices: Union[pd.Series, np.ndarray, list],
        current_date: Union[date, str, None] = None,
    ) -> Dict:
        """Compute regime signal from a price series.

        Parameters
        ----------
        prices : array-like
            Historical closing prices, most recent last.
            Should contain at least 200 data points for full signal.
        current_date : date or str, optional
            Date for logging/display purposes.

        Returns
        -------
        dict
            Keys: ``vol_regime``, ``trend_regime``, ``annualized_vol``,
            ``sma_short``, ``sma_long``, ``guidance``, ``date``.
        """
        prices = np.asarray(prices, dtype=float)
        n = len(prices)

        # ── Volatility ────────────────────────────────────────────────────
        if n >= self.vol_window + 1:
            daily_returns = np.diff(prices[-self.vol_window - 1:]) / prices[-self.vol_window - 1:-1]
            ann_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
        else:
            ann_vol = 0.0

        vol_regime = self._classify_vol(ann_vol)

        # ── Trend (SMA crossover) ─────────────────────────────────────────
        if n >= self.sma_long:
            sma_s = float(np.mean(prices[-self.sma_short:]))
            sma_l = float(np.mean(prices[-self.sma_long:]))
            trend_regime = self._classify_trend(prices[-1], sma_s, sma_l)
        elif n >= self.sma_short:
            sma_s = float(np.mean(prices[-self.sma_short:]))
            sma_l = None
            # Without long SMA, use price vs short SMA only
            if prices[-1] > sma_s * 1.02:
                trend_regime = "Bull"
            elif prices[-1] < sma_s * 0.98:
                trend_regime = "Bear"
            else:
                trend_regime = "Sideways"
        else:
            sma_s = sma_l = None
            trend_regime = "Sideways"  # Not enough data

        # ── Guidance ──────────────────────────────────────────────────────
        guidance = self.RISK_GUIDANCE.get(
            (trend_regime, vol_regime),
            f"Market regime: {trend_regime} trend, {vol_regime} volatility."
        )

        return {
            "date": str(current_date) if current_date else "N/A",
            "annualized_vol": round(ann_vol, 4),
            "vol_regime": vol_regime,
            "trend_regime": trend_regime,
            "sma_short": round(sma_s, 2) if sma_s else None,
            "sma_long": round(sma_l, 2) if sma_l else None,
            "guidance": guidance,
        }

    def _classify_vol(self, ann_vol: float) -> str:
        if ann_vol < self.vol_thresholds["low"]:
            return "Low"
        elif ann_vol < self.vol_thresholds["normal"]:
            return "Normal"
        elif ann_vol < self.vol_thresholds["high"]:
            return "High"
        else:
            return "Extreme"

    def _classify_trend(self, price: float, sma_s: float, sma_l: float) -> str:
        if sma_s > sma_l * 1.01 and price > sma_s:
            return "Bull"
        elif sma_s < sma_l * 0.99 and price < sma_s:
            return "Bear"
        else:
            return "Sideways"

    def build_prompt_injection(self, signal: Dict) -> str:
        """Build a natural-language regime context block for prompt injection.

        Parameters
        ----------
        signal : dict
            Output from ``compute_signal()``.

        Returns
        -------
        str
            Multi-line regime context block ready for insertion into prompts.
        """
        lines = [
            "\n=== MARKET REGIME CONTEXT (Quantitative Signals) ===",
            f"Date: {signal['date']}",
            f"Volatility Regime: {signal['vol_regime']} "
            f"(annualized volatility = {signal['annualized_vol']:.1%})",
            f"Trend Regime: {signal['trend_regime']}",
        ]

        if signal["sma_short"] is not None:
            lines.append(f"SMA({self.sma_short}) = {signal['sma_short']:.2f}")
        if signal["sma_long"] is not None:
            lines.append(f"SMA({self.sma_long}) = {signal['sma_long']:.2f}")

        lines.extend([
            "",
            f"Risk Guidance: {signal['guidance']}",
            "=== END REGIME CONTEXT ===\n",
        ])

        return "\n".join(lines)

    def build_finagent_preference(self, signal: Dict) -> str:
        """Build a compact regime clause for FinAgent's trader_preference.

        Parameters
        ----------
        signal : dict
            Output from ``compute_signal()``.

        Returns
        -------
        str
            Single-paragraph regime context for the trader preference field.
        """
        return (
            f"IMPORTANT: The current market regime is {signal['trend_regime']} "
            f"with {signal['vol_regime']} volatility "
            f"(annualized = {signal['annualized_vol']:.1%}). "
            f"{signal['guidance']} "
            f"Adjust your position sizing and risk tolerance accordingly."
        )
