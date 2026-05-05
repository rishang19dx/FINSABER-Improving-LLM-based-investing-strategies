# Plan 04 — Rolling Sharpe Drawdown-Triggered Stop Loss

**Category:** Pipeline  
**Difficulty:** 2/5 | **Effort:** ~25h | **Real-world Impact:** 5/5 | **Academic Value:** 3/5

---

## 1. Goal

Implement a **dynamic risk overlay** that wraps around any existing strategy and reduces position sizing when the strategy's rolling 30-day Sharpe drops below zero. This directly addresses the paper's conclusion that LLM strategies need "regime-aware risk controls" (§8) — turning an observation into a testable solution.

## 2. Mathematical Definition

### Rolling Sharpe Signal

At each trading day $t$, compute the rolling Sharpe over the past $W$ days:

$$RS_t = \frac{\bar{r}_{t-W:t} - r_f}{\sigma_{t-W:t}} \times \sqrt{252}$$

Where:
- $\bar{r}_{t-W:t}$ = mean daily return over window $[t-W, t]$
- $\sigma_{t-W:t}$ = standard deviation of daily returns over the same window
- $r_f$ = daily risk-free rate
- $W$ = lookback window (default: 30 trading days ≈ 6 weeks)

### Position Scaling Rule

$$\text{scale}_t = \begin{cases} 
1.0 & \text{if } RS_t \geq \tau_{\text{upper}} \\
\max(0.0,\; \frac{RS_t - \tau_{\text{lower}}}{\tau_{\text{upper}} - \tau_{\text{lower}}}) & \text{if } \tau_{\text{lower}} < RS_t < \tau_{\text{upper}} \\
0.0 & \text{if } RS_t \leq \tau_{\text{lower}}
\end{cases}$$

Default thresholds:
- $\tau_{\text{upper}} = 0.0$ (full position when rolling Sharpe is positive)
- $\tau_{\text{lower}} = -1.0$ (zero position when rolling Sharpe is deeply negative)

This creates a **linear ramp** between full and zero exposure, avoiding binary on/off switching that causes whipsaw.

### Max Drawdown Circuit Breaker (Optional)

Additionally, if cumulative drawdown from peak exceeds a threshold:

$$\text{If } \frac{\text{equity}_t - \text{peak}_t}{\text{peak}_t} \leq -\text{DD}_{\max} \Rightarrow \text{scale}_t = 0$$

Default: $DD_{\max} = 30\%$

## 3. Why This Matters for FINSABER

The paper finds (§7, Figure 2):
- **FinMem**: Sharpe -0.97 in bear markets, -0.19 in bull markets
- **FinAgent**: Sharpe -0.38 in bear markets, +0.12 in bull markets
- **Buy&Hold**: Sharpe -0.25 in bear markets, +0.61 in bull markets

The rolling Sharpe stop-loss directly tests: **If we add a simple quantitative risk overlay to LLM strategies, can we fix their bear-market over-aggression?**

Expected outcome:
- LLM bear-market Sharpe should improve dramatically (from -0.97 to perhaps -0.3)
- Bull-market Sharpe should be slightly reduced (delayed re-entry)
- Net RCS should improve significantly

## 4. Proposed Changes

### [NEW] [risk_overlay.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/strategy/timing/risk_overlay.py)

A **wrapper strategy** that delegates trading signals to an inner strategy but controls position sizing:

```python
import backtrader as bt
import numpy as np
from collections import deque
from backtest.strategy.timing.base_strategy import BaseStrategy


class RiskOverlayStrategy(BaseStrategy):
    """
    Wraps any existing timing strategy with a rolling-Sharpe-based
    position scaling mechanism.
    
    Usage:
        cerebro.addstrategy(
            RiskOverlayStrategy,
            inner_strategy_class=BuyAndHoldStrategy,
            sharpe_window=30,
            sharpe_upper=0.0,
            sharpe_lower=-1.0,
            max_drawdown_pct=30.0,
            **inner_kwargs
        )
    """
    params = (
        ('inner_strategy_class', None),     # The wrapped strategy class
        ('sharpe_window', 30),              # Rolling window in trading days
        ('sharpe_upper', 0.0),              # Full position threshold
        ('sharpe_lower', -1.0),             # Zero position threshold
        ('max_drawdown_pct', 30.0),         # Circuit breaker (%)
        ('risk_free_rate', 0.03),           # Annual risk-free rate
        ('total_days', 252),                # Required by base
    )
    
    def __init__(self):
        super().__init__()
        self.returns_buffer = deque(maxlen=self.p.sharpe_window)
        self.prev_equity = None
        self.current_scale = 1.0
        self.scale_history = []
        # Inner strategy signal generation would be delegated
    
    def _compute_rolling_sharpe(self):
        if len(self.returns_buffer) < self.p.sharpe_window:
            return None
        returns = np.array(self.returns_buffer)
        daily_rf = (1 + self.p.risk_free_rate) ** (1/252) - 1
        excess = returns - daily_rf
        if np.std(returns) == 0:
            return 0.0
        return (np.mean(excess) / np.std(returns)) * np.sqrt(252)
    
    def _compute_position_scale(self):
        rs = self._compute_rolling_sharpe()
        if rs is None:
            return 1.0  # Not enough data yet
        
        # Linear ramp between thresholds
        if rs >= self.p.sharpe_upper:
            scale = 1.0
        elif rs <= self.p.sharpe_lower:
            scale = 0.0
        else:
            scale = (rs - self.p.sharpe_lower) / (self.p.sharpe_upper - self.p.sharpe_lower)
        
        # Max drawdown circuit breaker
        if self.peak_equity > 0:
            current_dd = (self.broker.getvalue() - self.peak_equity) / self.peak_equity
            if current_dd <= -(self.p.max_drawdown_pct / 100):
                scale = 0.0
        
        return scale
```

### Alternative: Simpler Approach — Post-hoc Equity Curve Adjustment

Instead of a full backtrader strategy wrapper, implement a **post-hoc analysis** that takes existing equity curves and simulates what would have happened with the risk overlay:

```python
def apply_risk_overlay_posthoc(equity_curve: pd.Series, 
                                sharpe_window=30, 
                                sharpe_upper=0.0, 
                                sharpe_lower=-1.0):
    """
    Apply rolling Sharpe position scaling to an existing equity curve.
    Returns adjusted equity curve and metrics.
    """
    daily_returns = equity_curve.pct_change().dropna()
    adjusted_returns = []
    
    for i in range(len(daily_returns)):
        if i < sharpe_window:
            adjusted_returns.append(daily_returns.iloc[i])
            continue
        
        window_returns = daily_returns.iloc[i-sharpe_window:i]
        rs = (window_returns.mean() / window_returns.std()) * np.sqrt(252)
        
        # Linear scaling
        if rs >= sharpe_upper:
            scale = 1.0
        elif rs <= sharpe_lower:
            scale = 0.0
        else:
            scale = (rs - sharpe_lower) / (sharpe_upper - sharpe_lower)
        
        adjusted_returns.append(daily_returns.iloc[i] * scale)
    
    adjusted_equity = equity_curve.iloc[0] * (1 + pd.Series(adjusted_returns)).cumprod()
    return adjusted_equity
```

> **Recommendation**: Start with the post-hoc approach (much faster to implement, ~10h). It can be applied to ALL existing results without re-running any backtests. Then optionally build the full strategy wrapper for live use.

### [NEW] [run_risk_overlay_analysis.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/run_risk_overlay_analysis.py)

Loads existing `.pkl` results, applies the post-hoc risk overlay, and compares before/after metrics.

### [MODIFY] No existing files need to change for the post-hoc approach

The overlay reads existing equity curves from pickles and produces separate output.

## 5. Expected Results

### FinMem with Risk Overlay (estimated)

| Metric | FinMem (raw) | FinMem + Overlay | Change |
|--------|-------------|-----------------|--------|
| Bull Sharpe | -0.19 | -0.10 | ↑ slightly (delayed entry) |
| Bear Sharpe | -0.97 | -0.30 | ↑↑ major improvement |
| Sideways Sharpe | -0.10 | +0.05 | ↑ modest |
| Max Drawdown | -52% | -25% | ↑↑ halved |
| RCS | -0.22 | -0.05 | ↑↑ substantial |

### FinAgent with Risk Overlay (estimated)

| Metric | FinAgent (raw) | FinAgent + Overlay | Change |
|--------|---------------|-------------------|--------|
| Bear Sharpe | -0.38 | -0.10 | ↑ |
| Max Drawdown | -35% | -20% | ↑ |

## 6. Verification Plan

### Automated Tests
- Synthetic equity curve with known bear/bull periods
- Verify scale = 0 when rolling Sharpe < lower threshold
- Verify scale = 1 when rolling Sharpe > upper threshold
- Verify circuit breaker fires at max drawdown
- Verify adjusted equity <= raw equity (overlay can only reduce returns, not add)

### Integration Test
```bash
$env:PYTHONPATH = "."; python backtest/run_risk_overlay_analysis.py \
    --setup cherry_pick_both_finmem \
    --strategies BuyAndHoldStrategy,FinMemStrategy
```
- Output: before/after comparison table
- Verify BuyAndHold is barely affected (already performs well)
- Verify FinMem shows improved bear-market metrics

### Parameter Sensitivity
- Test window sizes: 15, 30, 60 days
- Test threshold pairs: (0, -0.5), (0, -1.0), (0.5, -1.0)
- Report as sensitivity matrix

## 7. Estimated Effort Breakdown

| Task | Hours |
|------|-------|
| Post-hoc `apply_risk_overlay_posthoc()` function | 4 |
| `run_risk_overlay_analysis.py` script | 5 |
| Load and process existing pickle files | 3 |
| Before/after comparison table generation | 3 |
| Parameter sensitivity analysis | 4 |
| Unit tests | 3 |
| Documentation and change_history update | 3 |
| **Total** | **~25h** |
