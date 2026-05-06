"""
Regime-Conditional Sharpe (RCS) Metric
======================================

Formalises the paper's Figure 2 heatmap into a single, rigorous, comparable
scalar per strategy.  RCS accounts for the empirical frequency of bull, bear,
and sideways markets so that strategies are evaluated proportionally to how
often each regime actually occurs.

    RCS_s = Σ_i  w_i · mean_Sharpe_{s,i}

where i ∈ {Bull, Sideways, Bear} and w_i is the fraction of years classified
as regime i in SPX_Classification.csv.

References
----------
- FINSABER paper §7, Figure 2 (regime heatmap)
- Zweig (2019) for ±20 % bull/bear thresholds
"""

import json
import os
import pandas as pd


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

class RegimeClassifier:
    """Classify years into bull / bear / sideways based on S&P 500 annual
    returns and compute empirical regime frequencies.

    Parameters
    ----------
    spx_classification_path : str, optional
        Path to ``SPX_Classification.csv``.  The CSV must contain at least
        the columns ``Year`` and ``Market`` (regime label).
    """

    BULL_THRESHOLD  =  0.20   # ≥ +20 %
    BEAR_THRESHOLD  = -0.20   # ≤ −20 %

    def __init__(self, spx_classification_path: str = None):
        if spx_classification_path and os.path.exists(spx_classification_path):
            self.regimes = pd.read_csv(spx_classification_path)
        else:
            self.regimes = None

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def classify_year(annual_return: float) -> str:
        """Return 'Bull', 'Bear', or 'Sideways' for a single year."""
        if annual_return >= RegimeClassifier.BULL_THRESHOLD:
            return "Bull"
        elif annual_return <= RegimeClassifier.BEAR_THRESHOLD:
            return "Bear"
        return "Sideways"

    def get_regime_weights(self, years: list = None) -> dict:
        """Compute empirical regime frequencies over selected years.

        Parameters
        ----------
        years : list of int, optional
            Subset of years to consider.  If *None* use all years in the CSV.

        Returns
        -------
        dict
            ``{'Bull': float, 'Sideways': float, 'Bear': float}`` summing
            to 1.0.
        """
        if self.regimes is not None:
            df = self.regimes
            if years:
                df = df[df["Year"].isin(years)]
            total = len(df)
            if total == 0:
                return {"Bull": 0.0, "Sideways": 0.0, "Bear": 0.0}
            return {
                "Bull":     len(df[df["Market"] == "Bull"])     / total,
                "Sideways": len(df[df["Market"] == "Sideways"]) / total,
                "Bear":     len(df[df["Market"] == "Bear"])     / total,
            }
        # Fallback defaults (approximate 2004-2024 distribution)
        return {"Bull": 0.40, "Sideways": 0.50, "Bear": 0.10}


# ---------------------------------------------------------------------------
# RCS computation
# ---------------------------------------------------------------------------

def compute_rcs(sharpe_records: list, regime_weights: dict) -> dict:
    """Compute Regime-Conditional Sharpe for every strategy.

    Parameters
    ----------
    sharpe_records : list[dict]
        Each dict must contain ``'Strategy'``, ``'Bull'``, ``'Sideways'``,
        ``'Bear'`` keys (same schema as ``sharpe_records.json``).
    regime_weights : dict
        ``{'Bull': w1, 'Sideways': w2, 'Bear': w3}`` where w_i sum to 1.

    Returns
    -------
    dict
        ``{strategy_name: rcs_score}``
    """
    rcs_scores = {}
    for record in sharpe_records:
        strategy = record["Strategy"]
        rcs = sum(
            regime_weights[regime] * record[regime]
            for regime in ["Bull", "Sideways", "Bear"]
        )
        rcs_scores[strategy] = round(rcs, 4)
    return rcs_scores


def compute_rcs_from_results(results_dir: str, spx_path: str,
                              years: list = None) -> dict:
    """End-to-end: load ``sharpe_records.json`` + ``SPX_Classification.csv``
    and return RCS scores for all strategies.

    Parameters
    ----------
    results_dir : str
        Directory containing ``sharpe_records.json``.
    spx_path : str
        Path to ``SPX_Classification.csv``.
    years : list of int, optional
        Restrict regime weights to these years.

    Returns
    -------
    dict
        ``{strategy_name: rcs_score}``
    """
    sharpe_path = os.path.join(results_dir, "sharpe_records.json")
    with open(sharpe_path) as f:
        sharpe_records = json.load(f)

    classifier = RegimeClassifier(spx_path)
    weights = classifier.get_regime_weights(years)

    return compute_rcs(sharpe_records, weights), weights
