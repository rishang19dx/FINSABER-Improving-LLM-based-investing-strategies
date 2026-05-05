# Plan 05 — Regime-Conditioned Prompting with Volatility Signal

**Category:** LLM Design  
**Difficulty:** 2/5 | **Effort:** ~30h | **Real-world Impact:** 4/5 | **Academic Value:** 5/5

---

## 1. Goal

Inject a **quantitative market regime signal** (VIX percentile, rolling volatility, or HMM-derived regime probability) directly into the LLM's system prompt, so the model is explicitly told the current market regime. This tests a high-value research question: **If an LLM is mathematically informed that it is in a high-volatility bear regime, does its trading behaviour actually change?**

## 2. Research Hypothesis

**H₀**: Regime-conditioned prompting does not significantly change LLM trading behaviour (Sharpe, turnover, drawdown).

**H₁**: Regime-conditioned prompting improves LLM performance, particularly in bear markets, by reducing excessive aggression when the model is told volatility is high.

### Expected Mechanism

The paper finds (§7) that LLM strategies are:
- Too **conservative** in bull markets (missing upside)
- Too **aggressive** in bear markets (incurring heavy losses)

This suggests the LLMs lack **regime awareness** — they can't distinguish "now is a good time to be aggressive" from "now is a time to be defensive." By injecting this signal explicitly, we test whether the failure is in **signal detection** (the LLM can't detect regimes from text alone) or in **decision policy** (the LLM ignores regime signals even when given them).

## 3. Signal Design

### Option A: VIX Percentile (Recommended — simplest)

```
MARKET REGIME CONTEXT:
Current VIX: 28.5 (85th percentile of historical values)
Regime Classification: HIGH VOLATILITY
Historical context: When VIX is above 25, markets have declined 
an average of -12% in the following month over the past 20 years.
Risk guidance: Consider reducing position sizes and being more 
conservative with buy signals.
```

### Option B: Rolling Volatility Z-Score

$$z_{\sigma,t} = \frac{\sigma_{30d,t} - \bar{\sigma}_{252d}}{\text{std}(\sigma_{252d})}$$

```
MARKET REGIME CONTEXT:
30-day realised volatility: 32.1% (annualised)
Volatility Z-score: +2.1 (significantly above average)
Regime: STRESSED MARKET — elevated uncertainty detected.
```

### Option C: Simple Return-Based Regime (matches paper's ±20% rule)

```
MARKET REGIME CONTEXT:
Year-to-date S&P 500 return: -18.4%
Classification: BEAR MARKET (approaching -20% threshold)
Historical precedent: In bear years, passive holding averaged -28% annual return.
Recommendation: Exercise extreme caution with buy signals.
```

> **Recommendation**: Start with **Option A (VIX)** — it's the industry standard regime proxy, readily available, and most interpretable by LLMs.

## 4. Architecture

### Current LLM Strategy Flow (FinMem/FinAgent)

```
Financial News + Price Data + Filings
        │
        ▼
    LLM Prompt (system + user)
        │
        ▼
    LLM Response: BUY / SELL / HOLD
        │
        ▼
    Backtrader executes trade
```

### Proposed Flow with Regime Conditioning

```
Financial News + Price Data + Filings
        │
        ▼
  ┌─────────────────────────────┐
  │  Regime Signal Generator    │ ← NEW
  │  (VIX percentile / vol z)  │
  └─────────┬───────────────────┘
            │
            ▼
    LLM Prompt (system + user + REGIME CONTEXT)  ← MODIFIED
        │
        ▼
    LLM Response: BUY / SELL / HOLD
        │
        ▼
    Backtrader executes trade
```

## 5. Proposed Changes

### [NEW] [regime_signal.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/toolkit/regime_signal.py)

Module to generate regime context strings from market data:

```python
import pandas as pd
import numpy as np

class RegimeSignalGenerator:
    """Generate human-readable regime context for LLM prompts."""
    
    def __init__(self, price_data: pd.DataFrame, vix_data: pd.DataFrame = None):
        self.price_data = price_data
        self.vix_data = vix_data
        self._precompute_vol_stats()
    
    def _precompute_vol_stats(self):
        """Compute rolling volatility statistics from price data."""
        returns = self.price_data['close'].pct_change().dropna()
        self.rolling_vol_30d = returns.rolling(30).std() * np.sqrt(252)
        self.rolling_vol_252d_mean = returns.rolling(252).std().mean() * np.sqrt(252)
        self.rolling_vol_252d_std = returns.rolling(252).std().std() * np.sqrt(252)
    
    def get_regime_context(self, date: str, ticker: str = "SPY") -> str:
        """
        Generate regime context string for a given date.
        
        Returns a formatted string to inject into the LLM system prompt.
        """
        date = pd.to_datetime(date)
        
        # Current rolling volatility
        if date in self.rolling_vol_30d.index:
            current_vol = self.rolling_vol_30d.loc[:date].iloc[-1]
        else:
            return ""  # No data available
        
        # Volatility z-score
        hist_vol = self.rolling_vol_30d.loc[:date]
        if len(hist_vol) > 252:
            vol_mean = hist_vol.iloc[-252:].mean()
            vol_std = hist_vol.iloc[-252:].std()
            z_score = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0
        else:
            z_score = 0
        
        # YTD return
        year_start = pd.Timestamp(date.year, 1, 1)
        year_prices = self.price_data.loc[year_start:date, 'close']
        if len(year_prices) > 1:
            ytd_return = (year_prices.iloc[-1] / year_prices.iloc[0]) - 1
        else:
            ytd_return = 0
        
        # Classify regime
        if z_score > 1.5:
            regime = "HIGH VOLATILITY / STRESSED"
            guidance = "Exercise caution. Consider reducing position sizes."
        elif z_score > 0.5:
            regime = "ELEVATED VOLATILITY"
            guidance = "Be selective with new positions. Monitor closely."
        elif z_score < -0.5:
            regime = "LOW VOLATILITY / CALM"
            guidance = "Conditions favour trend-following. Normal sizing."
        else:
            regime = "NORMAL VOLATILITY"
            guidance = "Standard market conditions. Normal sizing."
        
        context = f"""MARKET REGIME CONTEXT (as of {date.strftime('%Y-%m-%d')}):
- 30-day realised volatility: {current_vol*100:.1f}% (annualised)
- Volatility Z-score: {z_score:+.2f} (vs 1-year average)
- Year-to-date market return: {ytd_return*100:+.1f}%
- Regime classification: {regime}
- Risk guidance: {guidance}"""
        
        return context
    
    def get_regime_label(self, date: str) -> str:
        """Return just the regime label (bull/bear/sideways)."""
        date = pd.to_datetime(date)
        year_start = pd.Timestamp(date.year, 1, 1)
        year_prices = self.price_data.loc[year_start:date, 'close']
        if len(year_prices) > 1:
            ytd_return = (year_prices.iloc[-1] / year_prices.iloc[0]) - 1
            if ytd_return >= 0.20:
                return "bull"
            elif ytd_return <= -0.20:
                return "bear"
        return "sideways"
```

### [MODIFY] LLM Strategy Prompt Construction

The key integration point is where FinMem and FinAgent construct their prompts. These are in the `llm_traders/` directory:

#### FinMem Integration Point
```
llm_traders/finmem/  ← prompt construction happens here
```

The system prompt needs a regime context block injected before the trading instruction:

```python
# In the prompt construction method:
regime_context = regime_signal_gen.get_regime_context(current_date, ticker)

system_prompt = f"""You are a financial trading agent.

{regime_context}

Based on the above market regime and the following information, 
decide whether to BUY, SELL, or HOLD...
"""
```

#### FinAgent Integration Point
```
llm_traders/finagent/  ← similar prompt construction
```

### [NEW] [run_regime_prompt_experiment.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/run_regime_prompt_experiment.py)

Experiment runner that compares:
1. Baseline LLM strategy (no regime context)
2. LLM + VIX percentile prompt
3. LLM + volatility z-score prompt
4. LLM + simple regime label prompt

## 6. Experimental Design

### Controlled Comparison

| Variant | Prompt Modification | What It Tests |
|---------|-------------------|---------------|
| Baseline | No change | Current behaviour |
| + Regime Label | "Market is in BEAR regime" | Does the LLM respond to simple labels? |
| + Vol Signal | Full quantitative context (z-score, VIX) | Does the LLM use numerical signals? |
| + Risk Guidance | "Reduce position sizes" | Does explicit instruction change behaviour? |

### Metrics to Compare (per variant)

- Sharpe Ratio (overall and per-regime)
- Trading turnover (are trades reduced in bear markets?)
- Max Drawdown
- RCS (Regime-Conditional Sharpe from Plan 02)
- Action distribution: % BUY vs SELL vs HOLD per regime

### Statistical Test

- Paired t-test of Sharpe across rolling windows (baseline vs regime-conditioned)
- Action distribution chi-squared test (does the distribution change?)

## 7. Data Requirements

### VIX Data Source
- Can use `^VIX` from Yahoo Finance (free, available via `yfinance`)
- Or compute implied volatility proxy from S&P 500 options data
- Or use realised volatility as a simpler proxy (already available from price data)

### S&P 500 Price Data
- Already available: `data/price/all_sp500_prices_2000_2024_delisted_include.csv`
- Can compute SPY-equivalent returns from this

## 8. Prerequisites

> [!IMPORTANT]  
> This plan requires:
> 1. **OPENAI_API_KEY** set in `.env` (FinMem/FinAgent use GPT-4o)
> 2. **LLM dependencies** installed: `toml`, `sentence_transformers`, `chromadb`
> 3. **VIX data** downloaded (or use realised vol proxy)
>
> Cost estimate: Each full LLM backtest costs ~$198 in API fees (Appendix G of the paper). Running 4 variants × 2 strategies = ~$1,584 total.

## 9. Verification Plan

### Automated Tests
- Unit test `RegimeSignalGenerator`: known dates → expected regime labels
- Unit test prompt injection: verify regime context appears in final prompt string

### Integration Test
- Run a single-ticker, short-period test (e.g., TSLA, 1 month) with and without regime context
- Compare LLM outputs (BUY/SELL/HOLD) — expect different actions in high-vol periods

### Academic Validation
- If H₁ holds: regime-conditioned LLMs should show improved bear-market Sharpe
- If H₀ holds: LLMs ignore the signal → proves the failure is in decision policy, not signal detection
- Either outcome is publishable

## 10. Estimated Effort Breakdown

| Task | Hours |
|------|-------|
| Implement `RegimeSignalGenerator` | 5 |
| Download/prepare VIX data | 2 |
| Identify and modify FinMem prompt construction point | 5 |
| Identify and modify FinAgent prompt construction point | 5 |
| Create `run_regime_prompt_experiment.py` | 4 |
| Run experiments (waiting for API) | 3 |
| Analysis: before/after comparison tables | 3 |
| Documentation and change_history update | 3 |
| **Total** | **~30h** |

## 11. Why Both Outcomes Are Publishable

**If regime prompting works** (H₁):
- "We show that LLM trading agents can be significantly improved by injecting quantitative regime signals into their prompts, reducing bear-market drawdowns by X% and improving Regime-Conditional Sharpe by Y."
- Actionable design recommendation for future LLM investors.

**If regime prompting doesn't work** (H₀):
- "We demonstrate that LLM trading agents fail to utilise explicit quantitative regime information even when provided in-context, suggesting the performance gap is rooted in fundamental limitations of LLM decision policies rather than information access."
- Strengthens the paper's argument that the problem is deeper than architecture complexity.
