# FINSABER — Change History

A virtual diary of all modifications made to this project since cloning.

---

## 2026-05-05 22:00 IST — Initial Setup & Dependency Installation

**Context**: Cloned repo, no conda available. System has Python 3.13.4 (repo recommends 3.10).

### Actions:
- Created `.env` from `.env.example` (API keys left blank)
- Installed core pip dependencies since conda was unavailable:
  - `backtrader`, `python-dotenv`, `tqdm`, `tabulate`, `huggingface_hub`
  - `datasets`, `pandas_datareader`, `statsmodels`, `xgboost`
  - `openai`, `colorlog`
  - (numpy, pandas, matplotlib, requests were already present)

### Note:
- `pip install finsaber` failed due to NumPy compilation issues (needs GCC on Python 3.13)
- `pwb_toolbox` not installed (optional, shows harmless warning)
- RL deps (`stable_baselines3`) and full LLM deps (`sentence_transformers`, `toml`, `chromadb`) **not** installed

---

## 2026-05-05 22:20 IST — Import Guard Fixes (4 files modified)

**Problem**: The codebase has hard imports of optional heavy dependencies (RL, LLM packages) at the module level. This prevents non-LLM baseline experiments from running when those packages aren't installed.

**Fix**: Wrapped optional imports in `try/except ImportError` blocks so the rest of the framework loads cleanly.

### File: `backtest/strategy/timing/__init__.py`

```diff
-from .finrl import FinRLStrategy
+try:
+    from .finrl import FinRLStrategy
+except ImportError:
+    pass
```

**Reason**: `FinRLStrategy` imports `stable_baselines3` and `rl_traders`, which are heavy RL dependencies not needed for baseline strategies.

### File: `backtest/strategy/selection/__init__.py`

```diff
-from .fincon_agent_selector import FinConSP500Selector
+try:
+    from .fincon_agent_selector import FinConSP500Selector
+except ImportError:
+    pass
```

**Reason**: `FinConSP500Selector` transitively imports `sentence_transformers` and other LLM packages via `llm_traders.fincon_selector`.

### File: `backtest/strategy/timing_llm/__init__.py`

```diff
 from .base_strategy_iso import BaseStrategyIso
-from .finmem import FinMemStrategy
-from .finagent import FinAgentStrategy
+try:
+    from .finmem import FinMemStrategy
+except ImportError:
+    pass
+try:
+    from .finagent import FinAgentStrategy
+except ImportError:
+    pass
```

**Reason**: `FinMemStrategy` requires `toml`; `FinAgentStrategy` requires additional LLM deps. Both are unnecessary for non-LLM baseline experiments.

### File: `backtest/experiment_runner.py`

```diff
-from backtest.strategy.selection import RandomSP500Selector, MomentumSP500Selector, LowVolatilitySP500Selector, FinConSP500Selector
+from backtest.strategy.selection import RandomSP500Selector, MomentumSP500Selector, LowVolatilitySP500Selector
+try:
+    from backtest.strategy.selection import FinConSP500Selector
+except ImportError:
+    FinConSP500Selector = None
```

**Reason**: `experiment_runner.py` explicitly imports `FinConSP500Selector`, which fails when LLM deps are missing. Setting it to `None` allows non-FinCon setups to work.

---

## 2026-05-05 22:35 IST — Dataset Downloads

### Downloaded from HuggingFace (`waylonli/FINSABER-data`):

1. **Cherry-pick dataset** (~86MB)
   - Source: `data/finmem_data/stock_data_cherrypick_2000_2024_v2.pkl`
   - Saved as: `data/finmem_data/stock_data_cherrypick_2000_2024_v2.pkl`
   - Copied to: `data/finmem_data/stock_data_cherrypick_2000_2024.pkl`
   - (Copy needed because code references the filename without `_v2`)

2. **S&P500 price CSV** (~253MB)
   - Source: `data/price/all_sp500_prices_2000_2024_delisted_include.csv`
   - Saved as: `data/price/all_sp500_prices_2000_2024_delisted_include.csv`

### NOT downloaded:
- `stock_data_sp500_2000_2024.pkl` (~13GB) — needed for composite setups

---

## 2026-05-05 22:45 IST — First Successful Run

**Command**:
```bash
$env:PYTHONPATH = "."; python backtest/run_baselines_exp.py --setup cherry_pick_both_finmem --include BuyAndHoldStrategy --date_from 2022-10-06 --date_to 2023-04-10
```

**Result**: ✅ Success — BuyAndHoldStrategy ran on TSLA, NFLX, AMZN, MSFT (COIN skipped due to insufficient prior data).

### Output files overwritten:
- `backtest/output/cherry_pick_both_finmem/BuyAndHoldStrategy/results.csv`
- `backtest/output/cherry_pick_both_finmem/BuyAndHoldStrategy/2022-10-06_2023-04-10.pkl`

### Results:
| Ticker | Total Return | Annual Return | Sharpe | Max Drawdown |
|--------|-------------|---------------|--------|-------------|
| TSLA   | -20.48%     | -27.19%       | -0.34  | 52.73%      |
| NFLX   | +43.08%     | +64.21%       | 1.33   | 20.18%      |
| AMZN   | -13.25%     | -17.87%       | -0.46  | 31.55%      |
| MSFT   | +21.13%     | +30.40%       | 0.97   | 14.16%      |
| **Avg**| **+7.62%**  | **+12.39%**   |**0.37**| **29.66%**  |

---

## Summary of All Modified Files

| File | Type of Change |
|------|---------------|
| `backtest/strategy/timing/__init__.py` | Import guard for FinRLStrategy |
| `backtest/strategy/selection/__init__.py` | Import guard for FinConSP500Selector |
| `backtest/strategy/timing_llm/__init__.py` | Import guards for FinMemStrategy, FinAgentStrategy |
| `backtest/experiment_runner.py` | Import guard for FinConSP500Selector |
| `backtest/output/cherry_pick_both_finmem/BuyAndHoldStrategy/results.csv` | Overwritten by fresh run |
| `backtest/output/cherry_pick_both_finmem/BuyAndHoldStrategy/2022-10-06_2023-04-10.pkl` | Overwritten by fresh run |
| `.env` | Created from `.env.example` (untracked, in .gitignore) |
| `data/finmem_data/stock_data_cherrypick_2000_2024.pkl` | Downloaded (untracked) |
| `data/finmem_data/stock_data_cherrypick_2000_2024_v2.pkl` | Downloaded (untracked) |
| `data/price/all_sp500_prices_2000_2024_delisted_include.csv` | Downloaded (untracked) |

---

## 2026-05-06 00:35 IST — Implementation Plans Created

**Context**: Analysed the paper's experimental gaps and created 5 detailed implementation plans for high-ROI extensions, biased toward mathematical contributions.

### Files Created:
All under `plans/` directory (new):

| File | Title | Category | Effort |
|------|-------|----------|--------|
| `01_calmar_omega_ratios.md` | Calmar & Omega Ratios as Primary Metrics | Quant Math | ~20h |
| `02_regime_conditional_sharpe.md` | Regime-Conditional Sharpe (RCS) Metric | Quant Math | ~30h |
| `03_probabilistic_sharpe_ratio.md` | Probabilistic Sharpe Ratio & Min Track Record | Quant Math | ~45h |
| `04_rolling_sharpe_stop_loss.md` | Rolling Sharpe Drawdown-Triggered Stop Loss | Pipeline | ~25h |
| `05_regime_conditioned_prompting.md` | Regime-Conditioned Prompting with Volatility Signal | LLM Design | ~30h |

### Key integration points identified per plan:
- **Metrics**: `backtest/toolkit/metrics.py` (currently 38 lines with just Sortino + annual vol)
- **Metric wiring**: `backtest/finsaber_bt.py` → `_calculate_annualized_metrics()` (L330) and `_analyze_results()` (L269)
- **Results aggregation**: `backtest/toolkit/operation_utils.py` → `aggregate_results_one_strategy()` (L495)
- **LLM prompts**: `llm_traders/finmem/` and `llm_traders/finagent/`
- **Regime data**: `backtest/output/SPX_Classification.csv` and `sharpe_records.json` (pre-existing)

---

## 2026-05-06 01:20 IST — Plan 01: Calmar & Omega Ratios Implemented

**Context**: First planned extension. Added Calmar Ratio (AR / MDD) and Omega Ratio (probability-weighted gains vs losses) as primary evaluation metrics alongside existing Sharpe and Sortino.

### Mathematical Definitions

- **Calmar Ratio** = Annualised Return / Maximum Drawdown — penalises strategies with high drawdowns
- **Omega Ratio** = Σ(gains above threshold) / Σ(losses below threshold) — evaluates the entire return distribution, superior for non-normal returns. Omega > 1.0 means gains outweigh losses.

### Files Modified

| File | Change |
|------|--------|
| `backtest/toolkit/metrics.py` | Added `calculate_calmar_ratio()` and `calculate_omega_ratio()` functions |
| `backtest/finsaber_bt.py` | Wired new metrics into `_calculate_annualized_metrics()` (computation), `_analyze_results()` (print + return dict) |
| `backtest/toolkit/operation_utils.py` | Added `calmar_ratio` and `omega_ratio` columns to both `aggregate_results()` and `aggregate_results_one_strategy()`, with `.get()` fallbacks for backward compatibility with old pickle files |

### Verification Run

```bash
PYTHONPATH=. python backtest/run_baselines_exp.py \
    --setup cherry_pick_both_finmem \
    --include BuyAndHoldStrategy \
    --date_from 2022-10-06 --date_to 2023-04-10
```

### Results Comparison: Baseline vs Plan 01

**Baseline (before — only Sharpe/Sortino):**

| Ticker | Total Return | Annual Return | Sharpe | Sortino | Max Drawdown |
|--------|-------------|---------------|--------|---------|-------------|
| TSLA   | -20.48%     | -27.19%       | -0.34  | -0.47   | 52.73%      |
| NFLX   | +43.08%     | +64.21%       | 1.33   | 2.32    | 20.18%      |
| AMZN   | -13.25%     | -17.87%       | -0.46  | -0.68   | 31.55%      |
| MSFT   | +21.13%     | +30.40%       | 0.97   | 1.50    | 14.16%      |
| **Avg**| **+7.62%**  | **+12.39%**   |**0.37**|**0.67** | **29.66%**  |

**Plan 01 (after — with Calmar & Omega):**

| Ticker | Total Return | Annual Return | Sharpe | Sortino | **Calmar** | **Omega** | Max Drawdown |
|--------|-------------|---------------|--------|---------|-----------|----------|-------------|
| TSLA   | -20.48%     | -27.19%       | -0.34  | -0.47   | **-0.516**| **0.934**| 52.73%      |
| NFLX   | +43.08%     | +64.21%       | 1.33   | 2.32    | **3.181** | **1.326**| 20.18%      |
| AMZN   | -13.25%     | -17.87%       | -0.46  | -0.68   | **-0.566**| **0.911**| 31.55%      |
| MSFT   | +21.13%     | +30.40%       | 0.97   | 1.50    | **2.146** | **1.217**| 14.16%      |
| **Avg**| **+7.62%**  | **+12.39%**   |**0.37**|**0.67** |**1.061**  |**1.097** | **29.66%**  |

### Key Observations

1. **Calmar exposes drawdown severity**: TSLA's Calmar (-0.516) is worse than its Sharpe (-0.34) because Calmar directly penalises the 52.73% drawdown. NFLX's Calmar (3.181) is much better than its Sharpe (1.33) because its drawdown was moderate relative to its return.
2. **Omega reveals asymmetry**: TSLA's Omega (0.934 < 1.0) confirms losses outweigh gains at the risk-free threshold. NFLX's Omega (1.326 > 1.0) confirms gains dominate.
3. **Backward compatible**: Old pickle files without `calmar_ratio`/`omega_ratio` keys default to 0 via `.get()` fallbacks — no breakage.

---

## 2026-05-06 02:55 IST — Plan 02: Regime-Conditional Sharpe (RCS) Implemented

**Context**: Second planned extension. Formalised the paper's Figure 2 regime heatmap into a single, comparable scalar per strategy — the **Regime-Conditional Sharpe (RCS)**.

### Mathematical Definition

$$RCS_s = \sum_{i \in \{Bull, Sideways, Bear\}} w_i \cdot \overline{Sharpe}_{s,i}$$

Where $w_i$ = empirical frequency of regime $i$ from `SPX_Classification.csv` (2000–2023).

### Regime Weights (from SPX_Classification.csv, 2000–2023)

| Regime | Count | Weight |
|--------|-------|--------|
| Sideways | 17/24 | 70.83% |
| Bull | 5/24 | 20.83% |
| Bear | 2/24 | 8.33% |

### Files Created

| File | Change |
|------|--------|
| `backtest/toolkit/rcs.py` | **[NEW]** `RegimeClassifier` (parses SPX CSV, computes regime weights) + `compute_rcs()` (sharpe records × weights → RCS scores) + `compute_rcs_from_results()` (end-to-end loader) |
| `backtest/run_rcs_analysis.py` | **[NEW]** Standalone CLI script: prints regime weights, ranked RCS table, per-strategy breakdown with weighted contributions |

### Verification

```bash
PYTHONPATH=. python backtest/run_rcs_analysis.py
```

Hand-verified Buy And Hold: 0.610×0.2083 + 0.479×0.7083 + (-0.283)×0.0833 = **+0.4429** ✓

### RCS Rankings (all 16 strategies)

| # | Strategy | Bull Sharpe | Sideways Sharpe | Bear Sharpe | RCS |
|---|----------|------------|----------------|-------------|-----|
| 1 | Buy And Hold | +0.610 | +0.479 | -0.283 | **+0.4429** |
| 2 | ARIMA Predictor | +0.577 | +0.341 | +0.193 | **+0.3776** |
| 3 | RL-PPO | +0.299 | +0.327 | -0.252 | **+0.2728** |
| 4 | RL-SAC | +0.221 | +0.249 | -0.041 | **+0.2190** |
| 5 | RL-DDPG | +0.223 | +0.240 | -0.004 | **+0.2160** |
| 6 | RL-A2C | +0.257 | +0.217 | -0.202 | **+0.1903** |
| 7 | ATR Band | +0.063 | +0.161 | +0.042 | **+0.1307** |
| 8 | FinAgent | +0.121 | +0.191 | -0.383 | **+0.1287** |
| 9 | Bollinger Bands | +0.050 | +0.076 | -0.204 | **+0.0474** |
| 10 | XGBoost Predictor | -0.041 | -0.025 | +0.126 | **-0.0156** |
| 11 | Turn Of The Month | +0.075 | -0.061 | +0.119 | **-0.0176** |
| 12 | Trend Following | +0.086 | -0.014 | -0.350 | **-0.0214** |
| 13 | **FinMem** | -0.187 | -0.100 | -0.967 | **-0.1906** |
| 14 | SMA Cross | -0.559 | -0.361 | +0.153 | **-0.3596** |
| 15 | WMA Cross | -0.564 | -0.356 | +0.022 | **-0.3676** |

### Key Observations

1. **Sideways dominates** (70.8% weight): Strategies that perform well in sideways markets dominate the RCS ranking. Buy&Hold's +0.479 sideways Sharpe contributes +0.3394 to its RCS — more than its bull and bear contributions combined.
2. **FinMem is deeply negative** (RCS = -0.1906): Negative across *all three* regimes means the frequency weighting cannot rescue it. The -0.967 bear Sharpe contributes -0.0806, but the sideways -0.100 contributes -0.0712 — nearly as damaging because sideways markets are 8.5× more frequent than bear markets.
3. **FinAgent survives** (RCS = +0.1287): Unlike FinMem, FinAgent's positive sideways Sharpe (+0.191) contributes +0.1353, which offsets its bear-market losses (-0.0319). This confirms the paper's finding that FinAgent is the less catastrophic LLM strategy.
4. **ARIMA beats all RL strategies**: ARIMA's consistent positive Sharpe across all three regimes (including +0.193 in bear markets) gives it the second-highest RCS. No RL strategy matches this regime resilience.
5. **Bear weight is only 8.3%**: This means bear-market performance matters far less than sideways performance in the RCS metric — a realistic reflection of historical market frequency but a reminder that RCS is optimistic about tail risk.

---

## 2026-05-06 03:10 IST — Plan 03: Probabilistic Sharpe Ratio (PSR) & MinTRL Implemented

**Context**: Third planned extension. Implements Bailey & de Prado (2014) to statistically test whether observed Sharpe Ratios are significant and how much data is needed to trust them.

### Mathematical Definitions

**PSR** — the probability that the true Sharpe exceeds a benchmark:

$$PSR(\hat{SR}^*) = \Phi\left(\frac{(\hat{SR} - \hat{SR}^*) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4}{4} \hat{SR}^2}}\right)$$

**MinTRL** — minimum observations for significance at 95% confidence:

$$T^* = 1 + \left(1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4}{4} \hat{SR}^2\right) \left(\frac{z_\alpha}{\hat{SR} - \hat{SR}^*}\right)^2$$

### Files Created / Modified

| File | Change |
|------|--------|
| `backtest/toolkit/psr.py` | **[NEW]** `probabilistic_sharpe_ratio()`, `minimum_track_record_length()`, `compute_psr_from_returns()` |
| `backtest/run_psr_analysis.py` | **[NEW]** Standalone CLI: loads pickles → extracts equity curves → PSR/MinTRL per strategy + per ticker detail |
| `backtest/finsaber_bt.py` | **[MOD]** Wired PSR + MinTRL into `_calculate_annualized_metrics()` and `_analyze_results()` — now printed after every backtest |
| `backtest/toolkit/operation_utils.py` | **[MOD]** Added `psr` and `min_trl_years` columns to both `aggregate_results()` and `aggregate_results_one_strategy()`, with `.get()` fallbacks + `inf` guards |

### Verification

#### Integration test (backtest engine):

```bash
PYTHONPATH=. python backtest/run_baselines_exp.py --setup cherry_pick_both_finmem \
    --include BuyAndHoldStrategy --date_from 2022-10-06 --date_to 2023-04-10
```

Output now includes:
```
PSR (vs SR=0): 0.4035 ✗  |  Min track record: ∞ years    (TSLA, negative Sharpe)
PSR (vs SR=0): 0.8933 ✗  |  Min track record: 1.3 years  (NFLX, positive but < 0.95)
```

#### Standalone analysis (cherry_pick, 6-month window):

| Strategy | Obs SR | Skew | Kurt | T (days) | PSR | MinTRL | Signif? |
|----------|--------|------|------|----------|-----|--------|---------|
| XGBoostPredictor | +0.990 | +0.636 | +17.02 | 728 | 0.9555 | 2.7y | **YES** |
| ARIMAPredictor | +0.928 | +0.236 | +6.45 | 728 | 0.9432 | 3.1y | no |
| FinAgent | +0.508 | +1.189 | +9.03 | 630 | 0.7929 | 10.1y | no |
| BuyAndHold | +0.355 | +0.406 | +4.12 | 728 | 0.7277 | 21.3y | no |
| Bollinger Bands | -0.230 | +0.122 | +12.14 | 815 | 0.3396 | ∞ | no |

#### Standalone analysis (lowvol_sp500_5, 20-year window — the killer table):

| Strategy | Obs SR | Skew | Kurt | T (days) | PSR | MinTRL | Signif? |
|----------|--------|------|------|----------|-----|--------|---------|
| TrendFollowing | +0.588 | +0.356 | +24.86 | 27699 | 1.0000 | **7.8y** | **YES** |
| BuyAndHold | +0.546 | -0.169 | +28.12 | 29504 | 1.0000 | **9.2y** | **YES** |
| ARIMAPredictor | +0.483 | +0.314 | +79.25 | 29504 | 1.0000 | **11.7y** | **YES** |
| TurnOfTheMonth | +0.405 | +0.447 | +52.78 | 29504 | 1.0000 | **16.4y** | **YES** |
| ATRBand | +0.336 | +0.285 | +119.47 | 28174 | 0.9998 | **24.1y** | **YES** |
| FinAgent | +0.314 | -10.594 | +628.91 | 20376 | 0.9939 | **34.9y** | **YES** |
| **FinMem** | **+0.097** | **-20.651** | **+1289.06** | 20376 | **0.7927** | **328.7y** | **no** |
| SMA Cross | +0.102 | +1.731 | +46.35 | 27604 | 0.8574 | 259.6y | no |

### Key Observations

1. **FinMem needs 328.7 years** to validate its Sharpe of +0.097 at 95% confidence. This is the single most devastating number in the entire analysis — the paper evaluated it over 6 months. The extreme excess kurtosis (+1289.06) and negative skewness (-20.651) make the Sharpe estimator wildly unreliable.
2. **Buy&Hold needs only 9.2 years** — already achieved in the 20-year dataset → PSR = 1.0000, statistically significant.
3. **FinAgent barely passes** (PSR = 0.9939, MinTRL = 34.9y) — its extreme kurtosis (+628.91) from occasional large losses makes its Sharpe estimate noisy, but 20 years of data just barely suffices.
4. **Short evaluations are dangerous**: In the cherry_pick (6-month) setup, only XGBoost passes at 95%. Even Buy&Hold with Sharpe +0.355 needs 21.3 years! This quantitatively validates the paper's qualitative argument.
5. **Kurtosis is the enemy**: The strategies with the highest kurtosis (FinMem +1289, FinAgent +629) have the longest MinTRL. This is because fat-tailed returns make the Sharpe estimator imprecise — exactly the non-normality that PSR was designed to detect.

