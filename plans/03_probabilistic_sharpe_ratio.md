# Plan 03 — Probabilistic Sharpe Ratio (PSR) & Minimum Track Record Length

**Category:** Quant Math  
**Difficulty:** 3/5 | **Effort:** ~45h | **Real-world Impact:** 3/5 | **Academic Value:** 5/5

---

## 1. Goal

Apply **Bailey & de Prado's Probabilistic Sharpe Ratio (PSR)** to quantify exactly how many years of data are needed to trust a given Sharpe ratio at a chosen confidence level. This directly formalises the paper's core qualitative argument — that 6-month evaluations inflate LLM performance — into a mathematically rigorous statistical test.

## 2. Mathematical Definitions

### Probabilistic Sharpe Ratio (PSR)

Given an observed Sharpe Ratio $\hat{SR}$ computed from $T$ observations:

$$PSR(\hat{SR}^*) = \Phi\left(\frac{(\hat{SR} - \hat{SR}^*) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \hat{SR}^2}}\right)$$

Where:
- $\hat{SR}^*$ = benchmark Sharpe Ratio (e.g., 0 or Buy&Hold's Sharpe)
- $T$ = number of return observations
- $\hat{\gamma}_3$ = skewness of returns
- $\hat{\gamma}_4$ = kurtosis of returns
- $\Phi$ = standard normal CDF

**Interpretation**: PSR gives the probability that the true Sharpe exceeds $\hat{SR}^*$. If PSR < 0.95, the observed Sharpe is **not statistically significant** at 95% confidence.

### Minimum Track Record Length (MinTRL)

The minimum number of observations $T^*$ needed for the observed Sharpe to be significant:

$$T^* = 1 + \left(1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \hat{SR}^2\right) \left(\frac{z_\alpha}{\hat{SR} - \hat{SR}^*}\right)^2$$

Where $z_\alpha$ is the critical value for confidence level $\alpha$ (e.g., $z_{0.05} = 1.645$).

**Key insight**: For FinMem's TSLA Sharpe of 2.679 over ~126 trading days, MinTRL reveals whether 126 days is enough to trust that number — spoiler: it almost certainly isn't, given equity returns have kurtosis >> 3.

### Deflated Sharpe Ratio (DSR) — Optional Extension

Adjusts for multiple testing (data-snooping). When evaluating $K$ strategies:

$$DSR = PSR(\hat{SR}^*_K)$$

Where $\hat{SR}^*_K = \sqrt{V[{\hat{SR}}]} \cdot \left((1 - \gamma) z^{-1}(1 - \frac{1}{K}) + \gamma z^{-1}(1 - \frac{1}{K} e^{-1})\right)$

This penalises the "best" Sharpe for the number of strategies tested — highly relevant since the paper tests 16 strategies.

## 3. Why This Matters for FINSABER

The paper argues qualitatively that:
1. Short evaluation periods inflate LLM performance (§6.1)
2. Extending the period degrades LLM Sharpe (Table 2 vs Table 3)
3. Data-snooping across strategies is a concern (§4)

PSR and MinTRL turn these arguments into **quantitative proofs**:
- "FinMem needs at least X years of data to claim significance" — devastating if X >> 0.5 years
- "Given 16 strategies tested, the DSR-adjusted best Sharpe is Y" — quantifies the multiple-testing penalty

## 4. Proposed Changes

### [NEW] [psr.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/toolkit/psr.py)

Core PSR/MinTRL computation module:

```python
import numpy as np
from scipy.stats import norm

def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_observations: int,
    skewness: float,
    kurtosis: float
) -> float:
    """
    Compute the Probabilistic Sharpe Ratio (Bailey & de Prado, 2014).
    
    Returns the probability that the true Sharpe exceeds benchmark_sr.
    """
    if n_observations <= 1:
        return 0.0
    
    sr_diff = observed_sr - benchmark_sr
    
    # Denominator: accounts for non-normality of returns
    denom = np.sqrt(
        1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr**2
    )
    
    if denom == 0:
        return 0.0
    
    z_score = sr_diff * np.sqrt(n_observations - 1) / denom
    return norm.cdf(z_score)


def minimum_track_record_length(
    observed_sr: float,
    benchmark_sr: float,
    skewness: float,
    kurtosis: float,
    confidence: float = 0.95
) -> float:
    """
    Compute the Minimum Track Record Length (MinTRL).
    
    Returns the minimum number of observations needed for the
    observed Sharpe to be significant at the given confidence level.
    """
    z_alpha = norm.ppf(confidence)
    sr_diff = observed_sr - benchmark_sr
    
    if sr_diff <= 0:
        return float('inf')  # Can never be significant
    
    variance_factor = 1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr**2
    
    min_trl = 1 + variance_factor * (z_alpha / sr_diff) ** 2
    return min_trl


def compute_psr_from_returns(
    daily_returns: np.ndarray,
    benchmark_sr: float = 0.0,
    annualisation_factor: int = 252
) -> dict:
    """
    End-to-end PSR computation from a daily returns series.
    
    Returns dict with PSR, MinTRL, observed Sharpe, skewness, kurtosis.
    """
    from scipy.stats import skew, kurtosis as kurt
    
    n = len(daily_returns)
    if n < 10:
        return {"psr": 0, "min_trl": float('inf'), "observed_sr": 0}
    
    # Annualised Sharpe
    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1)
    if std_r == 0:
        return {"psr": 0, "min_trl": float('inf'), "observed_sr": 0}
    
    observed_sr = (mean_r / std_r) * np.sqrt(annualisation_factor)
    benchmark_sr_ann = benchmark_sr
    
    # Non-annualised SR for PSR formula (uses per-period SR)
    sr_per_period = mean_r / std_r
    benchmark_per_period = benchmark_sr_ann / np.sqrt(annualisation_factor)
    
    gamma3 = skew(daily_returns)
    gamma4 = kurt(daily_returns, fisher=False)  # excess=False → raw kurtosis
    
    psr = probabilistic_sharpe_ratio(
        sr_per_period, benchmark_per_period, n, gamma3, gamma4
    )
    
    min_trl = minimum_track_record_length(
        sr_per_period, benchmark_per_period, gamma3, gamma4, confidence=0.95
    )
    
    # Convert MinTRL from trading days to years
    min_trl_years = min_trl / annualisation_factor
    
    return {
        "psr": round(psr, 4),
        "min_trl_days": round(min_trl, 0),
        "min_trl_years": round(min_trl_years, 2),
        "observed_sr": round(observed_sr, 4),
        "skewness": round(gamma3, 4),
        "kurtosis": round(gamma4, 4),
        "n_observations": n
    }
```

### [NEW] [run_psr_analysis.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/run_psr_analysis.py)

Standalone script that loads existing pickle results and computes PSR/MinTRL:

```python
"""
Compute PSR and MinTRL from pre-existing backtest results.
Directly quantifies: "How many years of data does FinMem need
for its Sharpe to be statistically significant?"

Usage: python backtest/run_psr_analysis.py --setup cherry_pick_both_finmem
"""
# Loads .pkl files from backtest/output/<setup>/<strategy>/
# Extracts equity curves → daily returns → PSR + MinTRL
# Outputs a comparison table
```

### [MODIFY] [finsaber_bt.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/finsaber_bt.py)

**`_calculate_annualized_metrics()` (line 330-382):**
- After computing `daily_returns`, call `compute_psr_from_returns(daily_returns)`
- Add `psr`, `min_trl_years` to returned dict

**`_analyze_results()` (line 269-327):**
- Add `psr` and `min_trl_years` to eval_metrics dict
- Print: `f"PSR (vs Sharpe=0): {psr:.4f}, Min Track Record: {min_trl_years:.1f} years"`

### [MODIFY] [operation_utils.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/toolkit/operation_utils.py)

- Add `psr` and `min_trl_years` columns to results CSV (same pattern as other metrics)

## 5. Expected Killer Results

### Cherry-Pick Setup (6 months, Table 2)

| Strategy | Sharpe | T (days) | Skew | Kurt | PSR (vs 0) | MinTRL (years) |
|----------|--------|----------|------|------|------------|----------------|
| FinMem (TSLA) | 2.679 | ~126 | est. -0.3 | est. 5.0 | ~0.98 | ~0.3 |
| FinMem (AMZN) | -0.46 | ~126 | est. -0.2 | est. 4.5 | ~0.28 | ∞ |
| Buy&Hold (NFLX) | 1.33 | ~126 | est. 0.1 | est. 3.5 | ~0.92 | ~0.8 |

### Extended Setup (20 years, Table 3)

| Strategy | Sharpe | T (days) | PSR (vs 0) | MinTRL (years) |
|----------|--------|----------|------------|----------------|
| FinMem (Avg) | ~0.3 | ~5040 | ~0.99 | ~3.2 |
| Buy&Hold (Avg) | ~0.5 | ~5040 | ~1.00 | ~1.1 |

**The devastating comparison**: FinMem needs ~3.2 years of data to validate its 0.3 Sharpe. The original paper evaluated it over 6 months. This single number proves the paper's central thesis.

## 6. Verification Plan

### Automated Tests
- Reproduce Table 1 from Bailey & de Prado (2014) with known inputs
- Verify: SR=1.0, T=252, Normal returns (skew=0, kurt=3) → PSR ≈ Φ(√251) ≈ 1.0
- Verify: SR=0.5, T=60, skew=-0.5, kurt=5 → compute by hand
- Edge: SR < 0 → PSR < 0.5, MinTRL = ∞

### Integration Test
```bash
$env:PYTHONPATH = "."; python backtest/run_psr_analysis.py --setup cherry_pick_both_finmem
```
- Table output with PSR and MinTRL per strategy per ticker

## 7. Dependencies

- `scipy` (for `scipy.stats.norm`, `scipy.stats.skew`, `scipy.stats.kurtosis`) — likely already installed

## 8. Estimated Effort Breakdown

| Task | Hours |
|------|-------|
| Implement PSR, MinTRL, DSR in `psr.py` | 10 |
| Unit tests with known values from Bailey & de Prado | 6 |
| `run_psr_analysis.py` script (load pickles, compute, tabulate) | 8 |
| Wire into `finsaber_bt.py` and `operation_utils.py` | 8 |
| Analysis notebook: PSR comparison across setups | 6 |
| Documentation: mathematical derivation + interpretation guide | 4 |
| Change_history update | 3 |
| **Total** | **~45h** |

## 9. References

- Bailey, D.H. and de Prado, M.L. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management*, 40(5), pp. 94-107.
- Bailey, D.H., Borwein, J.M., de Prado, M.L. and Zhu, Q.J. (2015). "The Probability of Backtest Overfitting." *ERN: Econometric Modeling in Financial Economics*.
