# Plan 02 — Regime-Conditional Sharpe (RCS) Metric

**Category:** Quant Math  
**Difficulty:** 2/5 | **Effort:** ~30h | **Real-world Impact:** 4/5 | **Academic Value:** 5/5

---

## 1. Goal

Formalise the paper's Figure 2 heatmap into a single, rigorous, comparable scalar: the **Regime-Conditional Sharpe (RCS)**. This collapses the 3×N strategy-regime matrix into one number per strategy that accounts for the empirical frequency of bull, bear, and sideways markets.

## 2. Mathematical Definition

### Regime-Conditional Sharpe (RCS)

$$RCS_s = \sum_{i \in \{bull, bear, sideways\}} w_i \cdot \overline{Sharpe}_{s,i}$$

Where:
- $s$ = strategy
- $i$ = market regime (bull, bear, sideways)
- $w_i$ = empirical frequency of regime $i$ over the evaluation period
- $\overline{Sharpe}_{s,i}$ = average Sharpe ratio of strategy $s$ across all rolling windows classified as regime $i$

### Regime Classification (from the paper, §7)

A year $y$ is classified by the annual return of S&P 500:

$$R_y = \frac{P_T - P_0}{P_0}$$

- **Bull**: $R_y \geq +20\%$
- **Bear**: $R_y \leq -20\%$
- **Sideways**: $-20\% < R_y < +20\%$

The ±20% threshold follows industry convention (Zweig, 2019).

### Regime Weights

From `SPX_Classification.csv` over 2004–2024:
- Bull years: ~8/20 → $w_{bull} \approx 0.40$
- Sideways years: ~10/20 → $w_{sideways} \approx 0.50$
- Bear years: ~2/20 → $w_{bear} \approx 0.10$

### Why RCS > Plain Sharpe

Consider two strategies:
- **Strategy A**: Sharpe 0.8 in bull, 0.5 in sideways, -0.3 in bear → RCS = 0.40(0.8) + 0.50(0.5) + 0.10(-0.3) = **0.54**
- **Strategy B (LLM)**: Sharpe -0.2 in bull, 0.2 in sideways, -1.0 in bear → RCS = 0.40(-0.2) + 0.50(0.2) + 0.10(-1.0) = **-0.08**

The LLM strategy gets annihilated once regime frequency is accounted for — a fact hidden by the simple average Sharpe.

## 3. Why This Matters for FINSABER

The paper's Figure 2 heatmap is visually compelling but:
- Requires the reader to mentally weight the cells by regime frequency
- Cannot be used in automated comparisons or rankings
- Cannot be tested for statistical significance

RCS solves all three. It's a novel metric contribution that is directly publishable and makes the paper's core argument mathematically rigorous.

## 4. Proposed Changes

### [NEW] [rcs.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/toolkit/rcs.py)

New module dedicated to Regime-Conditional Sharpe computation:

```python
import pandas as pd
import numpy as np
import json
import os

class RegimeClassifier:
    """Classify years into bull/bear/sideways based on S&P 500 annual returns."""
    
    BULL_THRESHOLD = 0.20
    BEAR_THRESHOLD = -0.20
    
    def __init__(self, spx_classification_path: str = None):
        if spx_classification_path:
            self.regimes = pd.read_csv(spx_classification_path)
        else:
            self.regimes = None
    
    def classify_year(self, annual_return: float) -> str:
        if annual_return >= self.BULL_THRESHOLD:
            return "Bull"
        elif annual_return <= self.BEAR_THRESHOLD:
            return "Bear"
        else:
            return "Sideways"
    
    def get_regime_weights(self, years: list = None) -> dict:
        """Compute empirical regime frequencies over the evaluation period."""
        if self.regimes is not None:
            if years:
                filtered = self.regimes[self.regimes['Year'].isin(years)]
            else:
                filtered = self.regimes
            total = len(filtered)
            return {
                "Bull": len(filtered[filtered['Regime'] == 'Bull']) / total,
                "Sideways": len(filtered[filtered['Regime'] == 'Sideways']) / total,
                "Bear": len(filtered[filtered['Regime'] == 'Bear']) / total,
            }
        return {"Bull": 0.4, "Sideways": 0.5, "Bear": 0.1}  # fallback


def compute_rcs(sharpe_records: list, regime_weights: dict) -> dict:
    """
    Compute Regime-Conditional Sharpe for each strategy.
    
    Args:
        sharpe_records: list of dicts with keys 'Strategy', 'Bull', 'Sideways', 'Bear'
                       (same format as sharpe_records.json)
        regime_weights: dict with keys 'Bull', 'Sideways', 'Bear' and frequency values
    
    Returns:
        dict mapping strategy name -> RCS score
    """
    rcs_scores = {}
    for record in sharpe_records:
        strategy = record['Strategy']
        rcs = sum(
            regime_weights[regime] * record[regime]
            for regime in ['Bull', 'Sideways', 'Bear']
        )
        rcs_scores[strategy] = round(rcs, 4)
    return rcs_scores


def compute_rcs_from_results(results_dir: str, spx_path: str) -> dict:
    """
    End-to-end: load pre-computed sharpe_records.json and SPX classification,
    compute RCS for all strategies.
    """
    with open(os.path.join(results_dir, "sharpe_records.json")) as f:
        sharpe_records = json.load(f)
    
    classifier = RegimeClassifier(spx_path)
    weights = classifier.get_regime_weights()
    
    return compute_rcs(sharpe_records, weights)
```

### [NEW] [run_rcs_analysis.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/run_rcs_analysis.py)

Standalone script to compute and display RCS from existing results:

```python
"""
Compute Regime-Conditional Sharpe from pre-existing sharpe_records.json.
Usage: python backtest/run_rcs_analysis.py
"""
from backtest.toolkit.rcs import compute_rcs_from_results
import os

results_dir = os.path.join("backtest", "output")
spx_path = os.path.join(results_dir, "SPX_Classification.csv")

rcs_scores = compute_rcs_from_results(results_dir, spx_path)

# Print ranked results
print("\n" + "=" * 50)
print("Regime-Conditional Sharpe (RCS) Rankings")
print("=" * 50)
for strategy, score in sorted(rcs_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {strategy:<25} RCS = {score:+.4f}")
```

### [MODIFY] [operation_utils.py](file:///c:/Users/mhtgt/OneDrive/Desktop/FINSABER/FINSABER/backtest/toolkit/operation_utils.py)

Add optional RCS column to results CSV when regime data is available.

### [MODIFY] sharpe_records.json generation

Ensure the existing `market_regime_analysis.ipynb` pipeline also emits RCS scores alongside the heatmap.

## 5. Data Dependencies

```
backtest/output/SPX_Classification.csv     ← already exists in repo
backtest/output/sharpe_records.json        ← already exists (98 lines)
    │
    ├── RegimeClassifier.get_regime_weights()
    │       → {"Bull": 0.40, "Sideways": 0.50, "Bear": 0.10}
    │
    └── compute_rcs(sharpe_records, weights)
            → {"Buy&Hold": 0.54, "FinMem": -0.08, ...}
```

## 6. Expected Results

Using the existing `sharpe_records.json` data:

| Strategy | Bull Sharpe | Sideways Sharpe | Bear Sharpe | RCS (estimated) |
|----------|------------|----------------|-------------|-----------------|
| Buy&Hold | 0.611 | 0.480 | -0.252 | ~0.459 |
| ARIMA | 0.489 | 0.445 | 0.191 | ~0.438 |
| ATR Band | 0.521 | 0.454 | 0.240 | ~0.459 |
| FinMem | -0.187 | -0.100 | -0.967 | ~-0.222 |
| FinAgent | 0.121 | 0.191 | -0.383 | ~0.106 |

This quantitatively proves: FinMem's RCS is deeply negative, confirming the paper's qualitative claim with a single number.

## 7. Verification Plan

### Automated Tests
- Unit test: known Sharpe values + known weights → expected RCS (hand-calculated)
- Edge case: all regimes have same Sharpe → RCS = that Sharpe
- Edge case: bear weight = 0 → RCS ignores bear performance

### Integration Test
```bash
$env:PYTHONPATH = "."; python backtest/run_rcs_analysis.py
```
- Verify output matches hand-calculated values from `sharpe_records.json`
- Cross-check that strategy rankings are consistent with paper's conclusions

### Academic Validation
- Verify that RCS ranking matches or strengthens the paper's Table 4 conclusions
- RCS should rank Buy&Hold and ATR Band above FinMem and FinAgent

## 8. Estimated Effort Breakdown

| Task | Hours |
|------|-------|
| Implement `RegimeClassifier` and `compute_rcs()` | 6 |
| Create `run_rcs_analysis.py` standalone script | 3 |
| Parse `SPX_Classification.csv` format | 2 |
| Wire into existing results pipeline (optional) | 6 |
| Unit tests | 4 |
| Visualisation: RCS bar chart + comparison with plain Sharpe | 5 |
| Documentation and change_history update | 4 |
| **Total** | **~30h** |
