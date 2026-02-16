"""Risk metrics calculation utilities."""

import numpy as np
from typing import Dict, Tuple, Optional


def trapezoidal_integration(
    y: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Perform trapezoidal numerical integration.

    Parameters
    ----------
    y : np.ndarray
        Function values
    x : np.ndarray
        Integration points

    Returns
    -------
    float
        Integral value
    """
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))


def calculate_moments(
    strikes: np.ndarray,
    density: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Calculate moments of a density function.

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices
    density : np.ndarray
        Density values

    Returns
    -------
    Tuple[float, float, float, float]
        Mean, standard deviation, skewness, kurtosis
    """
    # Mean
    mean = trapezoidal_integration(strikes * density, strikes)

    # Variance
    variance = trapezoidal_integration((strikes - mean) ** 2 * density, strikes)
    std = np.sqrt(variance)

    # Skewness
    skewness = trapezoidal_integration(
        ((strikes - mean) / std) ** 3 * density,
        strikes,
    )

    # Kurtosis
    kurtosis = trapezoidal_integration(
        ((strikes - mean) / std) ** 4 * density,
        strikes,
    )

    return mean, std, skewness, kurtosis


def calculate_tail_metrics(
    strikes: np.ndarray,
    density: np.ndarray,
    current_price: float,
    tail_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Calculate basic tail risk metrics.

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices
    density : np.ndarray
        Density values
    current_price : float
        Current asset price
    tail_threshold : float, optional
        Threshold for tail definition (default: 0.1 for 10%)

    Returns
    -------
    Dict[str, float]
        Dictionary containing tail risk metrics
    """
    metrics = {}

    # Downside tail probability (probability of threshold drop or more)
    downside_threshold = current_price * (1 - tail_threshold)
    downside_mask = strikes <= downside_threshold
    metrics['downside_prob'] = trapezoidal_integration(
        density[downside_mask],
        strikes[downside_mask],
    )

    # Upside tail probability (probability of threshold increase or more)
    upside_threshold = current_price * (1 + tail_threshold)
    upside_mask = strikes >= upside_threshold
    metrics['upside_prob'] = trapezoidal_integration(
        density[upside_mask],
        strikes[upside_mask],
    )

    # Expected Shortfall (Conditional VaR)
    if metrics['downside_prob'] > 0:
        metrics['expected_shortfall'] = trapezoidal_integration(
            (current_price - strikes[downside_mask]) *
            density[downside_mask] / metrics['downside_prob'],
            strikes[downside_mask],
        )
    else:
        metrics['expected_shortfall'] = 0

    # Crash probability (20% drop)
    crash_threshold = current_price * 0.8
    crash_mask = strikes <= crash_threshold
    metrics['crash_prob'] = trapezoidal_integration(
        density[crash_mask],
        strikes[crash_mask],
    )

    return metrics


def calculate_advanced_tail_metrics(
    strikes: np.ndarray,
    density: np.ndarray,
    current_price: float,
    risk_free_rate: float = 0.02,
    time_to_expiry: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate advanced tail risk metrics.

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices
    density : np.ndarray
        Density values
    current_price : float
        Current asset price
    risk_free_rate : float, optional
        Risk-free interest rate (default: 0.02)
    time_to_expiry : float, optional
        Time to expiration in years

    Returns
    -------
    Dict[str, float]
        Dictionary containing advanced tail risk metrics
    """
    metrics = {}

    # Basic tail metrics
    basic_metrics = calculate_tail_metrics(strikes, density, current_price)
    metrics.update(basic_metrics)

    # Tail Value at Risk (TVaR) at different levels
    for level in [0.05, 0.01]:
        # Find VaR level
        cdf = np.cumsum(density) * (strikes[1] - strikes[0])
        var_idx = np.argmax(cdf >= level)
        var_level = strikes[var_idx] if var_idx < len(strikes) else strikes[-1]

        # Calculate Expected Shortfall beyond VaR
        tail_mask = strikes <= var_level
        tail_prob = trapezoidal_integration(density[tail_mask], strikes[tail_mask])

        if tail_prob > 0:
            expected_shortfall = trapezoidal_integration(
                (current_price - strikes[tail_mask]) *
                density[tail_mask] / tail_prob,
                strikes[tail_mask],
            )
        else:
            expected_shortfall = 0

        metrics[f'var_{int(level * 100)}'] = current_price - var_level
        metrics[f'es_{int(level * 100)}'] = expected_shortfall

    # Scenario probabilities
    scenarios = {
        'mild_correction': (0.95, 0.90),  # 5-10% drop
        'significant_drop': (0.80, 0.70),  # 20-30% drop
        'crash': (0.00, 0.50),             # >50% drop
        'bull_market': (1.10, 1.50),       # 10-50% gain
        'bubble': (1.50, float('inf')),    # >50% gain
    }

    for name, (lower, upper) in scenarios.items():
        if upper == float('inf'):
            mask = strikes >= current_price * lower
        else:
            mask = (strikes >= current_price * lower) & (strikes < current_price * upper)

        if np.any(mask):
            metrics[f'prob_{name}'] = trapezoidal_integration(
                density[mask],
                strikes[mask],
            )
        else:
            metrics[f'prob_{name}'] = 0

    # Tail risk ratio (left tail / right tail probability)
    left_tail = trapezoidal_integration(
        density[strikes <= current_price * 0.9],
        strikes[strikes <= current_price * 0.9],
    )
    right_tail = trapezoidal_integration(
        density[strikes >= current_price * 1.1],
        strikes[strikes >= current_price * 1.1],
    )
    metrics['tail_ratio'] = left_tail / right_tail if right_tail > 0 else float('inf')

    return metrics


def calculate_variance_risk_premium(
    implied_variance: float,
    realized_variance: float,
) -> Dict[str, float]:
    """
    Calculate variance risk premium metrics.

    Parameters
    ----------
    implied_variance : float
        Implied variance from options
    realized_variance : float
        Expected realized variance from historical data

    Returns
    -------
    Dict[str, float]
        Dictionary containing VRP metrics
    """
    metrics = {
        'implied_variance': implied_variance,
        'realized_variance': realized_variance,
        'vrp': implied_variance - realized_variance,
        'vrp_ratio': implied_variance / realized_variance if realized_variance > 0 else float('inf'),
        'implied_vol': np.sqrt(implied_variance),
        'realized_vol': np.sqrt(realized_variance),
        'vol_premium': np.sqrt(implied_variance) - np.sqrt(realized_variance),
    }

    return metrics
