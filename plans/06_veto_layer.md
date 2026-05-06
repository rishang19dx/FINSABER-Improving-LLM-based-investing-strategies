# ARIMA-LLM Veto Layer Hybrid Strategy

## Goal

Test whether ARIMA's trend signal can filter out LLM buy signals during adverse regimes, improving bear-market Sharpe while preserving bull-market performance.

## Architecture Challenge

ARIMA runs in **Backtrader** (`bt.Strategy`), while FinMem/FinAgent run in the **ISO framework** (`BaseStrategyIso`). They cannot run simultaneously in the same engine.

**Solution**: Post-hoc signal reconstruction from existing result pickles. Both strategies already have daily equity curves stored. We reconstruct implied signals and combine them offline.

## Proposed Changes

### [NEW] [veto_layer.py](file:///home/rishang/Desktop/FINSABER/FINSABER/backtest/toolkit/veto_layer.py)

Core logic module containing:

1. **`infer_position_from_equity(equity_curve, price_series)`** — Reconstruct daily position state (IN/OUT) by comparing equity curve movements to the underlying price. When equity tracks the stock price, position is IN. When equity is flat while price moves, position is OUT.

2. **`reconstruct_arima_signal(price_series, order=(5,1,0), train_period=756)`** — Re-run ARIMA forecasting on the same price data to get daily directional forecasts (UP/DOWN/FLAT). This is deterministic and matches the existing strategy exactly.

3. **`apply_veto_logic(llm_signals, arima_signals)`** — The veto rules:
   - LLM=BUY + ARIMA=UP → **BUY** (confirmed)
   - LLM=BUY + ARIMA=DOWN/FLAT → **HOLD** (vetoed)
   - LLM=SELL → **SELL** (always pass through)
   - LLM=HOLD → **HOLD** (always pass through)

4. **`simulate_veto_equity(price_series, combined_signals, initial_cash=100000)`** — Simulate the equity curve that would result from the combined signals, with proper commission modeling.

---

### [NEW] [run_veto_experiment.py](file:///home/rishang/Desktop/FINSABER/FINSABER/backtest/run_veto_experiment.py)

Experiment runner that:

1. Loads existing LLM pickle results (FinMem, FinAgent)
2. Loads price data for the same tickers/windows
3. Runs ARIMA forecast reconstruction on price data
4. Infers LLM position signals from equity curves  
5. Applies veto logic to produce hybrid signals
6. Simulates hybrid equity curves
7. Computes all metrics (Sharpe, Calmar, Omega, PSR, MinTRL)
8. Produces comparison table: LLM-only vs ARIMA-only vs Hybrid vs Reverse-Veto

### Four Variants

| Variant | Description |
|---------|-------------|
| **LLM-only** | Existing FinMem/FinAgent results (baseline) |
| **ARIMA-only** | Existing ARIMA results (benchmark) |
| **Veto Hybrid** | ARIMA vetoes LLM buys when trend is down |
| **Reverse Veto** | LLM vetoes ARIMA buys when sentiment is negative (control) |

## Verification Plan

### Automated Tests
```bash
PYTHONPATH=. python3 backtest/run_veto_experiment.py --setup lowvol_sp500_5
```

- Compare hybrid RCS vs LLM-only RCS vs ARIMA-only RCS
- Check that bear-market Sharpe improves for hybrid
- Check that bull-market Sharpe is preserved (not significantly degraded)
- Run PSR on hybrid equity curves

### Sanity Checks
- Verify that ARIMA signal reconstruction matches existing ARIMA equity curves
- Verify that position inference from LLM equity curves is consistent with known trade history

## Open Questions

> [!IMPORTANT]
> **Signal granularity**: The existing pickles only store daily equity values, not explicit BUY/SELL/HOLD signals. Position inference from equity curve changes is an approximation — small price movements may be ambiguous. Should we use a threshold (e.g., equity change > 0.1% of stock price move = position is IN)?

> [!IMPORTANT]  
> **Which setups to run**: The `lowvol_sp500_5` setup has both FinMem and FinAgent results with 20 years of data (strongest statistical power). The `cherry_pick_both_finmem` setup has only 6 months but includes more LLM strategies. Recommend running both but leading with `lowvol_sp500_5`.
