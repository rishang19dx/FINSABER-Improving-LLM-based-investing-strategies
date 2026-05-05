# MA546 Course Project: Evaluating LLM-based Financial Investing Strategies (FINSABER)

> **Course:** MA546 - Introduction to Quantitative Finance, IIT Mandi  
> **Based on Paper:** "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?" (KDD 2026)  

This repository contains our course project implementation based on the official **FINSABER** framework. We are reproducing the baseline experiments and extending the codebase to introduce advanced quantitative metrics, risk management mechanisms, and regime-conditioned LLM prompting.

---

## 📈 Project Roadmap & Implementation Plans

As part of our course project, we have outlined five key extensions to the baseline framework, biased toward mathematical contributions in quantitative finance. Detailed implementation plans are located in the `plans/` directory:

1. **[Calmar & Omega Ratios as Primary Metrics](plans/01_calmar_omega_ratios.md)** (~20h): Integrating Calmar and Omega ratios to better assess tail risk and drawdowns compared to standard Sharpe/Sortino ratios.
2. **[Regime-Conditional Sharpe (RCS) Metric](plans/02_regime_conditional_sharpe.md)** (~30h): Evaluating strategy robustness by computing Sharpe ratios conditionally based on underlying market regimes (e.g., high vs. low volatility).
3. **[Probabilistic Sharpe Ratio (PSR) & Min Track Record](plans/03_probabilistic_sharpe_ratio.md)** (~45h): Applying PSR to account for non-normal return distributions (skewness/kurtosis) and calculating the minimum track record required for statistical significance.
4. **[Rolling Sharpe Drawdown-Triggered Stop Loss](plans/04_rolling_sharpe_stop_loss.md)** (~25h): Implementing a dynamic, portfolio-level risk management pipeline that triggers stop-losses when rolling Sharpe drops below historical thresholds.
5. **[Regime-Conditioned Prompting with Volatility Signal](plans/05_regime_conditioned_prompting.md)** (~30h): Enhancing the `FinMem` and `FinAgent` LLM strategies by injecting volatility and regime-state signals directly into the prompt context.

## 🛠️ Current Progress & Modifications

We have bootstrapped the repository and successfully run initial baselines. Detailed logs are available in [change_history.md](change_history.md).

**Key Accomplishments:**
- **Environment Setup**: Initialized the project on Python 3.13 using `pip` (bypassing conda requirements) and configured `.env`.
- **Framework Decoupling**: Implemented import guards across `backtest/strategy/` to isolate heavy RL (`stable_baselines3`) and LLM (`sentence_transformers`) dependencies. This allows pure quantitative baselines to run instantly without requiring GPU-heavy package installations.
- **Data Acquisition**: Downloaded required S&P 500 pricing datasets (~253MB) and cherry-picked stock datasets (~86MB) from HuggingFace.
- **Successful Baseline Run**: Executed the `BuyAndHoldStrategy` under the `cherry_pick_both_finmem` setup (TSLA, NFLX, AMZN, MSFT) for the 2022-2023 timeframe, establishing a working pipeline and verifying output generation.

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
