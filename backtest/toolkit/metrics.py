import numpy as np

def calculate_sortino_ratio(daily_returns, risk_free_rate=0):
    """
    Calculate the Sortino Ratio of a strategy
    :param daily_returns: pd.Series, daily returns of a strategy
    :param risk_free_rate: float, risk-free rate
    :return: float, Sortino Ratio
    """
    # check if non-zero daily returns are enough for calculation
    non_zero_daily_returns = daily_returns[daily_returns != 0]
    if len(non_zero_daily_returns) < 5:
        return 0

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    excess_returns = daily_returns - daily_rf

    # Calculate downside deviation (for Sortino Ratio)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) < 2:
        return 0

    downside_deviation = downside_returns.std()

    if downside_deviation == 0:
        return 0

    return (excess_returns.mean() ) / downside_deviation * np.sqrt(252)

def calculate_annual_volatility(daily_returns):
    """
    Calculate the annualized volatility of a strategy
    :param daily_returns: pd.Series, daily returns of a strategy
    :return: float, annualized volatility
    """
    return daily_returns.std() * np.sqrt(252)

def calculate_calmar_ratio(annual_return, max_drawdown):
    """
    Calculate the Calmar Ratio of a strategy.
    Calmar = Annualised Return / Maximum Drawdown
    Penalises strategies with high drawdowns (e.g. LLM strategies in bear markets).
    :param annual_return: float, annualised return (as decimal, e.g. 0.10 for 10%)
    :param max_drawdown: float, maximum drawdown as percentage (e.g. 20.0 for 20%)
    :return: float, Calmar Ratio
    """
    if max_drawdown == 0:
        return 0.0
    return annual_return / (max_drawdown / 100)

def calculate_omega_ratio(daily_returns, threshold=0.0):
    """
    Calculate the Omega Ratio of a strategy.
    Omega = sum of gains above threshold / sum of losses below threshold
    Evaluates the entire return distribution — superior for non-normal returns.
    Omega > 1.0 means gains outweigh losses at the chosen threshold.
    :param daily_returns: pd.Series, daily returns of a strategy
    :param threshold: float, minimum acceptable return threshold (default: 0)
    :return: float, Omega Ratio
    """
    excess = daily_returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess <= 0].sum()
    if losses == 0 or len(daily_returns) < 5:
        return 0.0
    return gains / losses