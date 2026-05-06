# MA546 Course Project: Evaluating LLM-based Financial Investing Strategies (FINSABER)

> **Course:** MA546 - Introduction to Quantitative Finance, IIT Mandi  
> **Based on Paper:** "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?" (KDD 2026)  

This repository contains our course project implementation based on the official **FINSABER** framework. We reproduce the baseline experiments and extend the codebase with advanced quantitative metrics, a dynamic risk management overlay, and regime-conditioned LLM prompting.

---

## 📈 Project Roadmap — All 5 Extensions Complete ✅

Five extensions to the baseline framework, biased toward mathematical contributions in quantitative finance. Detailed plans are in `plans/`:

| # | Extension | Status | Key Result |
|---|-----------|--------|------------|
| 1 | [Calmar & Omega Ratios](plans/01_calmar_omega_ratios.md) | ✅ Done | Calmar exposes TSLA's -52% drawdown; Omega confirms asymmetric losses |
| 2 | [Regime-Conditional Sharpe (RCS)](plans/02_regime_conditional_sharpe.md) | ✅ Done | FinMem RCS = **-0.1906** — negative across all three regimes |
| 3 | [Probabilistic Sharpe Ratio & MinTRL](plans/03_probabilistic_sharpe_ratio.md) | ✅ Done | FinMem needs **328.7 years** to validate its Sharpe at 95% confidence |
| 4 | [Rolling Sharpe Stop-Loss](plans/04_rolling_sharpe_stop_loss.md) | ✅ Done | Risk overlay improves **every** strategy; FinMem SR: -0.256 → +0.730 |
| 5 | [Regime-Conditioned Prompting](plans/05_regime_conditioned_prompting.md) | ✅ Done | 12-class regime signal injected into FinMem + FinAgent prompts |

---

## 🧮 Extended Metrics & Analysis Tools

### Metrics integrated into the backtest engine

Every backtest run now automatically computes and reports these metrics alongside the original Sharpe/Sortino:

| Metric | Formula | What it reveals |
|--------|---------|-----------------|
| **Calmar Ratio** | Annual Return / Max Drawdown | Penalises high-drawdown strategies (LLMs in bear markets) |
| **Omega Ratio** | Σ(gains > τ) / Σ(losses ≤ τ) | Full return distribution — superior for non-normal returns |
| **PSR** | P(true SR > 0 \| data) | Statistical significance of the observed Sharpe |
| **MinTRL** | Min observations for 95% confidence | How long you need to trust the Sharpe |

### Standalone analysis scripts

```bash
# Regime-Conditional Sharpe — ranks all strategies by frequency-weighted regime Sharpe
PYTHONPATH=. python backtest/run_rcs_analysis.py

# Probabilistic Sharpe Ratio — loads pickle results and computes PSR/MinTRL
PYTHONPATH=. python backtest/run_psr_analysis.py --setup lowvol_sp500_5
PYTHONPATH=. python backtest/run_psr_analysis.py --setup cherry_pick_both_finmem

# Rolling Sharpe Risk Overlay — before/after comparison with dynamic position scaling
PYTHONPATH=. python backtest/run_risk_overlay_analysis.py --setup lowvol_sp500_5
PYTHONPATH=. python backtest/run_risk_overlay_analysis.py --setup cherry_pick_both_finmem --window 60
```

### Headline results

> **FinMem needs 328.7 years** of track record to validate its Sharpe of +0.097 at 95% confidence (excess kurtosis = +1289). The paper evaluated it over 6 months.

**Statistical significance (PSR & RCS)**

| Strategy | RCS | PSR | MinTRL | Verdict |
|----------|-----|-----|--------|---------|
| Buy & Hold | +0.4429 | 1.0000 | 9.2y | ✅ Statistically validated |
| ARIMA | +0.3776 | 1.0000 | 11.7y | ✅ Statistically validated |
| FinAgent | +0.1287 | 0.9939 | 34.9y | ⚠️ Barely significant |
| **FinMem** | **-0.1906** | **0.7927** | **328.7y** | ❌ Not trustworthy |

**Risk overlay impact (before → after)**

| Strategy | Raw Sharpe → Adjusted | Raw DD → Adjusted | Raw Return → Adjusted |
|----------|-----------------------|-------------------|-----------------------|
| Buy & Hold | +0.559 → **+1.542** | 15.0% → **5.8%** | +11.8% → **+32.3%** |
| FinMem | -0.256 → **+0.730** | 11.0% → **3.8%** | +1.6% → **+13.2%** |
| FinAgent | +0.315 → **+1.001** | 10.8% → **4.5%** | +5.1% → **+15.5%** |

---

## 🌡️ Regime-Conditioned Prompting (Plan 05)

When enabled, both FinMem and FinAgent receive real-time quantitative regime context in their prompts:

```
=== MARKET REGIME CONTEXT (Quantitative Signals) ===
Volatility Regime: Extreme (annualized volatility = 39.1%)
Trend Regime: Bear
Risk Guidance: CRITICAL WARNING: Bear market with extreme volatility —
potential crash conditions. SELL all discretionary positions.
=== END REGIME CONTEXT ===
```

**Activation:**
```bash
export FINSABER_REGIME_CONDITIONING=1    # enable (off by default)
PYTHONPATH=. python backtest/run_llm_traders_exp.py \
    --setup cherry_pick_both_finmem --strategy FinMemStrategy \
    --strat_config_path strats_configs/finmem_gpt_config.toml
```

> **Note:** Full experimental validation requires `OPENAI_API_KEY` to be set. The implementation is complete and module-tested; LLM experiments are deferred until the API key is configured.

---

## 🛠️ Current Progress & Modifications

Detailed logs are available in [change_history.md](change_history.md).

**Infrastructure:**
- **Environment Setup**: Initialized on Python 3.13 with `pip` (bypassing conda); configured `.env`
- **Framework Decoupling**: Import guards across `backtest/strategy/` to isolate heavy RL/LLM dependencies
- **Data Acquisition**: S&P 500 pricing (~253MB) + cherry-picked stocks (~86MB) from HuggingFace

**Completed Extensions:**

| Plan | New/Modified Files | Key Module |
|------|--------------------|------------|
| 01 Calmar & Omega | `backtest/toolkit/metrics.py`, `finsaber_bt.py`, `operation_utils.py` | `calculate_calmar_ratio()`, `calculate_omega_ratio()` |
| 02 RCS | `backtest/toolkit/rcs.py`, `backtest/run_rcs_analysis.py` | `RegimeClassifier`, `compute_rcs()` |
| 03 PSR & MinTRL | `backtest/toolkit/psr.py`, `backtest/run_psr_analysis.py`, `finsaber_bt.py`, `operation_utils.py` | `probabilistic_sharpe_ratio()`, `minimum_track_record_length()` |
| 04 Risk Overlay | `backtest/toolkit/risk_overlay.py`, `backtest/run_risk_overlay_analysis.py` | `apply_risk_overlay()`, `rolling_sharpe()` |
| 05 Regime Prompting | `backtest/toolkit/regime_signal.py`, `timing_llm/finmem.py`, `timing_llm/finagent.py`, `puppy/reflection.py` | `RegimeSignalGenerator` |

---

## 📖 Original FINSABER Overview (Baseline)

FINSABER is a comprehensive framework for evaluating trading strategies with a specific focus on comparing traditional technical analysis approaches with modern machine learning and large language model (LLM) based strategies.

<img src="https://github.com/waylonli/FINSABER/blob/main/figs/framework.png" width="900">

### 1. Environment Setup (Original Instructions)

To set up the full environment for running LLM experiments:

```bash
git clone https://github.com/waylonli/FINSABER
cd FINSABER
conda create -n finsaber python=3.10
conda activate finsaber
pip install -r requirements-complete.txt --no-deps
```

*Note: For our project, we are using a lightweight pip environment. See `change_history.md` for our specific dependency setup.*

Rename `.env.example` to `.env` and set the environment variables.
- `OPENAI_API_KEY` is required to run LLM-based strategies.
- `HF_ACCESS_TOKEN` is optional.

### 2. Data

We provide aggregated datasets on [HuggingFace](https://huggingface.co/datasets/waylonli/FINSABER-data). Datasets are auto-downloaded when running experiments.

| Dataset | Content | Size |
| :--- | :--- | :--- |
| **S&P500 Full** | Aggregated data (Price + News + Filings) | ~13 GB |
| **Price Only** | CSV format price-only data | ~253 MB |
| **Selected Symbols** | Aggregated data for TSLA, AMZN, MSFT, NFLX, COIN | ~60 MB |

### 3. Reproducing Results

The paper contains three experimental setups: *selective (cherry picking) setup*, *selected-4 setup*, and *composite setup*.

**Baselines (non-LLM):**
```bash
python backtest/run_baselines_exp.py \
    --setup <setup_name> \
    --include <strategy_name> \
    --date_from 2004-01-01 \
    --date_to 2024-01-01 \
    --training_years 2 \
    --rolling_window_size 1 \
    --rolling_window_step 1
```

**LLM Strategies:**
```bash
python backtest/run_llm_traders_exp.py \
    --setup <setup_name> \
    --strategy <strategy_name> \
    --strat_config_path <config_path> \
    --date_from 2004-01-01 \
    --date_to 2024-01-01 \
    --rolling_window_size 1 \
    --rolling_window_step 1
```

### 4. Extending the Framework

You can plug in your own datasets and strategies by subclassing the helpers under `backtest/strategy` and `backtest/data_util`. 
- **Custom Timing**: Subclass `backtest.strategy.timing.base_strategy.BaseStrategy`.
- **Custom LLM Timing**: Inherit from `backtest.strategy.timing_llm.base_strategy_iso.BaseStrategyIso`.
- **Custom Selection**: Subclass `backtest.strategy.selection.base_selector.BaseSelector`.
- **Custom Dataset**: Derive from `backtest.data_util.backtest_dataset.BacktestDataset`.

### Citation

```
@misc{li2025llmbasedfinancialinvestingstrategies,
      title={Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?}, 
      author={Weixian Waylon Li and Hyeonjun Kim and Mihai Cucuringu and Tiejun Ma},
      year={2025},
      eprint={2505.07078},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2505.07078}, 
}
```
