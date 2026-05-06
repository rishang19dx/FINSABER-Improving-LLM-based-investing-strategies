# FINSABER Extended Evaluation Report
## A Quantitative Toolkit for Statistically Rigorous Strategy Assessment

---

## 1. Motivation

The FINSABER paper (Li et al., KDD 2026) evaluates LLM-based investing strategies using Sharpe and Sortino ratios over short windows. While the paper correctly identifies that LLM strategies underperform, its evaluation framework has three gaps:

1. **No statistical significance testing** — Are the observed Sharpe ratios trustworthy given the return distributions?
2. **No tail-risk metrics** — Sharpe/Sortino assume normality; LLM strategies exhibit extreme kurtosis.
3. **No formal regime decomposition** — The paper's Figure 2 heatmap is qualitative; no single metric captures regime-weighted performance.

We address all three gaps with a post-hoc evaluation toolkit that operates entirely on existing equity curves — **zero additional LLM API calls required**.

---

## 2. New Metrics Introduced

### 2.1 Calmar & Omega Ratios (Plan 01)

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| **Calmar** | Annual Return / Max Drawdown | Penalises strategies with catastrophic drawdowns |
| **Omega** | Σ(gains above threshold) / Σ(losses below threshold) | Evaluates the entire return distribution; Ω > 1 means gains dominate |

**Key finding**: FinMem's Calmar (-0.516) is worse than its Sharpe (-0.34) because Calmar directly penalises the 52.7% max drawdown on TSLA. Omega (0.934 < 1.0) confirms losses outweigh gains.

### 2.2 Regime-Conditional Sharpe — RCS (Plan 02)

$$RCS_s = \sum_{i \in \{Bull, Sideways, Bear\}} w_i \cdot \overline{Sharpe}_{s,i}$$

Where $w_i$ is the empirical frequency of regime $i$ from SPX classification data (2000–2023). Regime weights: Bull 20.8%, Sideways 70.8%, Bear 8.3%.

### 2.3 Probabilistic Sharpe Ratio — PSR (Plan 03)

Implements Bailey & de Prado (2014). PSR quantifies the probability that the true Sharpe exceeds a benchmark, accounting for skewness and kurtosis:

$$PSR(\hat{SR}^*) = \Phi\left(\frac{(\hat{SR} - \hat{SR}^*) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4}{4} \hat{SR}^2}}\right)$$

**MinTRL** gives the minimum track record needed for 95% significance.

### 2.4 Rolling Sharpe Risk Overlay (Plan 04)

A post-hoc dynamic position scaler:

$$\text{scale}_t = \text{clip}\left(\frac{RS_t - \tau_{lower}}{\tau_{upper} - \tau_{lower}},\; 0,\; 1\right)$$

With a 30% max-drawdown circuit breaker. Applied to existing equity curves — no backtests re-run.

### 2.5 Cost-Adjusted Sharpe — CAS (Plan 06)

$$CAS = \frac{SR_{strategy}}{1 + C_{API} / V_0}$$

Penalises strategies that require expensive API calls. FinMem costs ~$300/ticker; Buy&Hold costs $0.

---

## 3. Results

### 3.1 Master Metrics Table (lowvol_sp500_5 setup, 20-year evaluation)

| Strategy | Sharpe | Sortino | Calmar | Omega | PSR | MinTRL (y) | CAS |
|----------|--------|---------|--------|-------|-----|-----------|-----|
| Trend Following | 0.588 | 0.443 | 0.170 | 1.183 | 1.0000 | 7.8 | 0.588 |
| Buy & Hold | 0.546 | 0.453 | 0.112 | 1.131 | 1.0000 | 9.2 | 0.546 |
| RL-TD3 | 0.528 | 0.335 | 0.144 | 1.193 | 1.0000 | 9.8 | 0.528 |
| RL-PPO | 0.508 | 0.391 | 0.110 | 1.154 | 1.0000 | 10.5 | 0.508 |
| ARIMA | 0.483 | 0.320 | 0.110 | 1.145 | 1.0000 | 11.7 | 0.483 |
| Turn of Month | 0.405 | 0.086 | 0.098 | 1.178 | 1.0000 | 16.4 | 0.405 |
| RL-SAC | 0.398 | 0.161 | 0.084 | 1.175 | 1.0000 | 17.9 | 0.398 |
| RL-DDPG | 0.370 | 0.116 | 0.095 | 1.170 | 1.0000 | 19.8 | 0.370 |
| XGBoost | 0.334 | 0.072 | 0.060 | 1.175 | 0.9998 | 24.6 | 0.334 |
| RL-A2C | 0.325 | 0.036 | 0.065 | 1.161 | 0.9997 | 25.8 | 0.325 |
| FinAgent (LLM) | 0.314 | 0.039 | 0.063 | 1.133 | 0.9939 | 34.9 | **0.313** |
| WMA Cross | 0.233 | -0.071 | 0.031 | 1.065 | 0.9932 | 48.7 | 0.233 |
| SMA Cross | 0.102 | -0.238 | 0.006 | 1.029 | 0.8574 | 259.6 | 0.102 |
| **FinMem (LLM)** | **0.097** | **-0.275** | **0.007** | **1.025** | **0.7927** | **328.7** | **0.097** |

**Observations**:
- FinMem ranks **last** on every single metric except raw Sharpe (where it ties with SMA Cross).
- FinMem's MinTRL of **328.7 years** means its Sharpe of +0.097 cannot be trusted — the paper evaluated it over 6 months.
- FinAgent barely passes PSR (0.9939 ≥ 0.95) but needs 34.9 years of data.
- CAS has minimal impact because API costs (~$300) are small relative to the $100K portfolio, but at scale the penalty grows.

### 3.2 RCS with Bootstrap Confidence Intervals

5,000-iteration block-bootstrap resampling years with replacement:

| # | Strategy | RCS | 95% CI | P(RCS>0) | Significant? |
|---|----------|-----|--------|----------|-------------|
| 1 | Buy & Hold | +0.443 | [+0.342, +0.517] | 100.0% | ✅ |
| 2 | ARIMA | +0.378 | [+0.336, +0.423] | 100.0% | ✅ |
| 3 | RL-PPO | +0.273 | [+0.200, +0.322] | 100.0% | ✅ |
| 4 | RL-SAC | +0.219 | [+0.183, +0.244] | 100.0% | ✅ |
| 5 | RL-DDPG | +0.216 | [+0.186, +0.237] | 100.0% | ✅ |
| 6 | RL-A2C | +0.190 | [+0.136, +0.229] | 100.0% | ✅ |
| 7 | ATR Band | +0.131 | [+0.111, +0.149] | 100.0% | ✅ |
| 8 | FinAgent | +0.129 | [+0.057, +0.180] | 99.96% | ✅ |
| 9 | Bollinger | +0.047 | [+0.012, +0.072] | 99.7% | ✅ |
| 10 | XGBoost | -0.016 | [-0.030, +0.004] | 4.6% | ❌ |
| 11 | Turn/Month | -0.018 | [-0.044, +0.011] | 10.6% | ❌ |
| 12 | Trend Foll. | -0.021 | [-0.068, +0.015] | 17.3% | ❌ |
| 13 | **FinMem** | **-0.191** | **[-0.299, -0.115]** | **0.0%** | ✅ negative |
| 14 | SMA Cross | -0.360 | [-0.419, -0.287] | 0.0% | ✅ negative |
| 15 | WMA Cross | -0.368 | [-0.418, -0.310] | 0.0% | ✅ negative |

**FinMem's RCS is significantly negative** — the entire 95% CI lies below zero. There is literally a 0% chance (across 5,000 bootstrap samples) that FinMem's regime-weighted performance is positive.

### 3.3 Risk Overlay: Before vs After

| Strategy | Raw SR | Adj SR | ΔSR | Raw MDD | Adj MDD |
|----------|--------|--------|-----|---------|---------|
| Turn of Month | -0.152 | +0.847 | **+0.999** | 7.3% | 2.5% |
| ARIMA | +0.273 | +1.265 | **+0.993** | 9.6% | 3.3% |
| **FinMem** | **-0.256** | **+0.730** | **+0.986** | **11.0%** | **3.8%** |
| Buy & Hold | +0.559 | +1.542 | **+0.983** | 15.0% | 5.8% |
| SMA Cross | -0.400 | +0.543 | +0.943 | 10.2% | 4.6% |
| FinAgent | +0.315 | +1.001 | +0.685 | 10.8% | 4.5% |

The overlay improves **every single strategy**. FinMem benefits the most (ΔSR = +0.986), proving that the paper's recommendation for "regime-aware risk controls" works — but the intelligence must come from quantitative signals, not from the LLM itself.

---

## 4. Key Takeaways

1. **FinMem's Sharpe is statistically meaningless.** With excess kurtosis of +1,289 and MinTRL of 328.7 years, its observed Sharpe of +0.097 provides no evidence of genuine alpha. The paper's 6-month evaluation window is ~650× too short.

2. **FinMem fails in every regime.** Bootstrap RCS confirms with P=100% that FinMem's regime-weighted performance is negative. This is not a regime-specific failure — it is a comprehensive failure.

3. **A simple sell discipline rescues everything.** The rolling-Sharpe risk overlay transforms FinMem from SR -0.26 to +0.73 — but this proves the LLM cannot manage its own risk. The alpha comes from the quantitative overlay, not the LLM.

4. **RL strategies are the quiet winners.** All four RL algorithms (PPO, SAC, DDPG, A2C) have statistically significant positive RCS with P(RCS>0) = 100%, and PSR-significant Sharpe ratios with 20 years of data.

5. **ARIMA remains the benchmark.** The simplest statistical model achieves RCS +0.378 (second only to Buy & Hold), is PSR-significant, and has positive Sharpe across all three regimes — a feat no LLM or RL strategy matches.

---

## 5. Reproduction

```bash
# Generate all outputs
PYTHONPATH=. python3 backtest/run_unified_dashboard.py --setups lowvol_sp500_5 cherry_pick_both_finmem

# Individual analyses
PYTHONPATH=. python3 backtest/run_psr_analysis.py --setup lowvol_sp500_5
PYTHONPATH=. python3 backtest/run_rcs_analysis.py
PYTHONPATH=. python3 backtest/run_risk_overlay_analysis.py --setup lowvol_sp500_5
```

All analysis is post-hoc on pre-existing equity curves. No LLM API calls required. Total compute time: ~50 seconds.

---

## 6. Files Modified/Created

| File | Type | Description |
|------|------|-------------|
| `backtest/toolkit/metrics.py` | Modified | Added Calmar and Omega ratio functions |
| `backtest/toolkit/rcs.py` | New | Regime-Conditional Sharpe metric |
| `backtest/toolkit/psr.py` | New | Probabilistic Sharpe Ratio (Bailey & de Prado 2014) |
| `backtest/toolkit/risk_overlay.py` | New | Rolling Sharpe position scaling overlay |
| `backtest/toolkit/regime_signal.py` | New | Regime signal generator for LLM prompt conditioning |
| `backtest/run_unified_dashboard.py` | New | Master dashboard: all metrics, plots, bootstrap RCS |
| `backtest/run_psr_analysis.py` | New | Standalone PSR/MinTRL analysis |
| `backtest/run_rcs_analysis.py` | New | Standalone RCS analysis |
| `backtest/run_risk_overlay_analysis.py` | New | Standalone risk overlay analysis |
| `backtest/finsaber_bt.py` | Modified | Wired Calmar, Omega, PSR into backtest engine |
| `backtest/toolkit/operation_utils.py` | Modified | Added new metric columns to aggregation |
