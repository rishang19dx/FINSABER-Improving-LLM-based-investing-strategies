# Plan 01 — Calmar & Omega Ratios as Primary Metrics

**Category:** Quant Math  
**Difficulty:** 1/5 | **Effort:** ~20h | **Real-world Impact:** 4/5 | **Academic Value:** 3/5

---

## 1. Goal

Replace or supplement Sharpe with **Calmar Ratio** (AR / MDD) and **Omega Ratio** (probability-weighted gains vs losses). These metrics are far more sensitive to tail risk and regime shifts — exactly where LLM strategies fail.

## 2. Mathematical Definitions

### Calmar Ratio

$$\text{Calmar} = \frac{\text{Annualised Return}}{\text{Maximum Drawdown}}$$

- Already available: `annual_return` and `max_drawdown` are computed in `_analyze_results()` ([finsaber_bt.py:320-327](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/finsaber_bt.py#L320-L327))
- Simple division — no new data needed
- Penalises strategies with high drawdowns (LLM strategies in bear markets)

### Omega Ratio

$$\Omega(\tau) = \frac{\int_{\tau}^{\infty} [1 - F(r)] \, dr}{\int_{-\infty}^{\tau} F(r) \, dr}$$

Where $F(r)$ is the CDF of daily returns and $\tau$ is the minimum acceptable return threshold (default: risk-free rate / 252).

Discrete approximation:

```python
def calculate_omega_ratio(daily_returns, threshold=0.0):
    excess = daily_returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess <= 0].sum()
    if losses == 0:
        return float('inf')
    return gains / losses
```

- Evaluates the **entire return distribution** — superior for non-normal returns
- Omega > 1.0 means gains outweigh losses at the chosen threshold
- Does not penalise upside volatility (unlike Sharpe)

## 3. Why This Matters for FINSABER

The paper's central finding is that LLM strategies have **asymmetric risk**: they bleed heavily in bear markets but are too conservative in bull markets. Sharpe penalises both upside and downside variance equally, which actually *flatters* LLMs by hiding their asymmetry. Calmar and Omega expose this directly:

- **Calmar**: FinMem's massive drawdowns (-52% on TSLA) will produce terrible Calmar scores even when its Sharpe looks moderate
- **Omega**: The probability-weighted gain/loss split will show that LLM strategies have thin right tails (missing bull runs) and fat left tails (bear losses)

## 4. Proposed Changes

### [MODIFY] [metrics.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/toolkit/metrics.py)

Add two new functions:

```python
def calculate_calmar_ratio(annual_return, max_drawdown):
    """Calmar Ratio = Annualised Return / Maximum Drawdown"""
    if max_drawdown == 0:
        return 0.0
    return annual_return / (max_drawdown / 100)  # max_drawdown stored as percentage

def calculate_omega_ratio(daily_returns, threshold=0.0):
    """Omega Ratio: sum of gains above threshold / sum of losses below threshold"""
    excess = daily_returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess <= 0].sum()
    if losses == 0 or len(daily_returns) < 5:
        return 0.0
    return gains / losses
```

### [MODIFY] [finsaber_bt.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/finsaber_bt.py)

**`_calculate_annualized_metrics()` (line 330-382):**
- Compute `calmar_ratio` from existing `annual_return` and `max_drawdown`
- Compute `omega_ratio` from `daily_returns` using daily risk-free rate as threshold
- Add both to the returned dict

**`_analyze_results()` (line 269-327):**
- Accept `max_drawdown` as input to pass to `_calculate_annualized_metrics()`, OR compute Calmar after metrics are returned
- Add `calmar_ratio` and `omega_ratio` to the returned eval_metrics dict (line 320-327)
- Add print lines for the new metrics (line 300-305)

### [MODIFY] [operation_utils.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/toolkit/operation_utils.py)

**`aggregate_results_one_strategy()` (line 495-614):**
- Add `calmar_ratio` and `omega_ratio` columns to `results_df_by_tickers` (line 512-514)
- Add accumulation variables: `avg_calmar_ratio`, `avg_omega_ratio` (line 516-521)
- Add per-window aggregation (line 539-544, 546-551)
- Add per-ticker average rows (line 573-578, 580-591)
- Add "All" average row (line 593-611)
- Add columns to CSV output

### No new files needed

## 5. Data Flow

```
daily_returns (pd.Series from equity curve)
    │
    ├── metrics.calculate_omega_ratio(daily_returns, threshold=daily_rf)
    │       → omega_ratio (float)
    │
    ├── metrics.calculate_calmar_ratio(annual_return, max_drawdown)
    │       → calmar_ratio (float)
    │
    └── returned in eval_metrics dict
            │
            └── aggregated in operation_utils → results.csv
```

## 6. Verification Plan

### Automated Tests
- Unit test `calculate_calmar_ratio()`: known AR=0.10, MDD=20% → Calmar = 0.5
- Unit test `calculate_omega_ratio()`: synthetic returns [0.01, -0.02, 0.015, -0.005] with threshold=0 → verify manually
- Edge cases: MDD=0 (never lost money), all-negative returns, single-day returns

### Integration Test
```bash
$env:PYTHONPATH = "."; python backtest/run_baselines_exp.py \
    --setup cherry_pick_both_finmem \
    --include BuyAndHoldStrategy \
    --date_from 2022-10-06 --date_to 2023-04-10
```
- Verify `results.csv` now contains `calmar_ratio` and `omega_ratio` columns
- Cross-check: Calmar for TSLA should be very negative (large drawdown), NFLX should be positive

### Backward Compatibility
- Existing pickle files will lack `calmar_ratio` and `omega_ratio` keys — aggregation code must handle `KeyError` gracefully with default 0

## 7. Estimated Effort Breakdown

| Task | Hours |
|------|-------|
| Implement `calculate_calmar_ratio()` and `calculate_omega_ratio()` in metrics.py | 2 |
| Wire into `_calculate_annualized_metrics()` and `_analyze_results()` | 4 |
| Update `aggregate_results_one_strategy()` for new columns | 4 |
| Handle backward compatibility with old pickle files | 3 |
| Unit tests + integration test | 4 |
| Documentation and change_history update | 3 |
| **Total** | **~20h** |
