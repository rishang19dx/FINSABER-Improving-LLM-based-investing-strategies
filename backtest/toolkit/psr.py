"""
Probabilistic Sharpe Ratio (PSR) & Minimum Track Record Length (MinTRL)
======================================================================

Implements Bailey & de Prado (2014) to quantify how many observations are
needed before an observed Sharpe Ratio can be trusted at a given confidence
level.  This directly formalises the FINSABER paper's qualitative argument
that short evaluation windows inflate LLM strategy performance.

Key functions
-------------
- ``probabilistic_sharpe_ratio``  — P(true SR > benchmark | data)
- ``minimum_track_record_length`` — min T for significance at α
- ``compute_psr_from_returns``    — end-to-end from daily returns

References
----------
Bailey, D.H. and de Prado, M.L. (2014).  "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
*Journal of Portfolio Management*, 40(5), pp. 94–107.
"""

import numpy as np
from scipy.stats import norm, skew, kurtosis as _kurtosis


# ---------------------------------------------------------------------------
# Core PSR / MinTRL
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Probability that the *true* Sharpe exceeds *benchmark_sr*.

    Uses the per-period (non-annualised) Sharpe ratio and adjusts for
    skewness and kurtosis of the return distribution.

    Parameters
    ----------
    observed_sr : float
        Observed Sharpe Ratio (per period, **not** annualised).
    benchmark_sr : float
        Benchmark / threshold Sharpe Ratio (per period).
    n_observations : int
        Number of return observations (T).
    skewness : float
        Sample skewness (γ₃) of returns.
    kurtosis : float
        Sample **excess** kurtosis (γ₄) of returns.
        (Normal distribution → γ₄ = 0.)

    Returns
    -------
    float
        PSR ∈ [0, 1].  Values ≥ 0.95 are significant at the 5 % level.
    """
    if n_observations <= 1:
        return 0.0

    sr_diff = observed_sr - benchmark_sr

    # Variance of the Sharpe estimator (Bailey & de Prado eq. 2)
    denom_sq = (
        1
        - skewness * observed_sr
        + ((kurtosis) / 4) * observed_sr ** 2
    )

    if denom_sq <= 0:
        return 0.0

    z = sr_diff * np.sqrt(n_observations - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def minimum_track_record_length(
    observed_sr: float,
    benchmark_sr: float,
    skewness: float,
    kurtosis: float,
    confidence: float = 0.95,
) -> float:
    """Minimum number of observations for significance at *confidence*.

    Parameters
    ----------
    observed_sr : float
        Observed Sharpe Ratio (per period).
    benchmark_sr : float
        Benchmark / threshold Sharpe Ratio (per period).
    skewness : float
        Sample skewness of returns.
    kurtosis : float
        Sample **excess** kurtosis of returns.
    confidence : float
        Significance level (default 0.95 → 5 % test).

    Returns
    -------
    float
        Minimum T* in periods.  Returns ``inf`` when the observed SR
        does not exceed the benchmark.
    """
    z_alpha = norm.ppf(confidence)
    sr_diff = observed_sr - benchmark_sr

    if sr_diff <= 0:
        return float("inf")

    variance_factor = (
        1
        - skewness * observed_sr
        + ((kurtosis) / 4) * observed_sr ** 2
    )

    return 1 + variance_factor * (z_alpha / sr_diff) ** 2


# ---------------------------------------------------------------------------
# End-to-end helper
# ---------------------------------------------------------------------------

def compute_psr_from_returns(
    daily_returns,
    benchmark_sr: float = 0.0,
    annualisation_factor: int = 252,
    confidence: float = 0.95,
) -> dict:
    """Compute PSR and MinTRL from a daily returns series.

    Parameters
    ----------
    daily_returns : array-like
        Daily (or per-period) returns.
    benchmark_sr : float
        Annualised benchmark Sharpe (default 0 = "is this strategy
        better than holding cash?").
    annualisation_factor : int
        Trading days per year (default 252).
    confidence : float
        Confidence level for MinTRL (default 0.95).

    Returns
    -------
    dict
        Keys: ``psr``, ``min_trl_days``, ``min_trl_years``,
        ``observed_sr``, ``skewness``, ``kurtosis``, ``n_observations``.
    """
    returns = np.asarray(daily_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = len(returns)

    if n < 10:
        return {
            "psr": 0.0,
            "min_trl_days": float("inf"),
            "min_trl_years": float("inf"),
            "observed_sr": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "n_observations": n,
        }

    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)

    if std_r == 0:
        return {
            "psr": 0.0,
            "min_trl_days": float("inf"),
            "min_trl_years": float("inf"),
            "observed_sr": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "n_observations": n,
        }

    # ── Per-period (daily) Sharpe ────────────────────────────────────────
    sr_per_period = mean_r / std_r
    benchmark_per_period = benchmark_sr / np.sqrt(annualisation_factor)

    # ── Higher moments ───────────────────────────────────────────────────
    gamma3 = float(skew(returns))
    # excess kurtosis (Normal → 0)
    gamma4 = float(_kurtosis(returns, fisher=True))

    # ── PSR ───────────────────────────────────────────────────────────────
    psr = probabilistic_sharpe_ratio(
        sr_per_period, benchmark_per_period, n, gamma3, gamma4
    )

    # ── MinTRL ────────────────────────────────────────────────────────────
    min_trl = minimum_track_record_length(
        sr_per_period, benchmark_per_period, gamma3, gamma4, confidence
    )
    min_trl_years = min_trl / annualisation_factor

    # ── Annualised observed SR (for display) ──────────────────────────────
    observed_sr_ann = sr_per_period * np.sqrt(annualisation_factor)

    return {
        "psr": round(psr, 4),
        "min_trl_days": round(min_trl, 1) if np.isfinite(min_trl) else float("inf"),
        "min_trl_years": round(min_trl_years, 2) if np.isfinite(min_trl_years) else float("inf"),
        "observed_sr": round(observed_sr_ann, 4),
        "skewness": round(gamma3, 4),
        "kurtosis": round(gamma4, 4),
        "n_observations": n,
    }
