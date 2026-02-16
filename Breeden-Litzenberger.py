"""Breeden-Litzenberger RND extraction implementation."""

import numpy as np
from scipy import interpolate
from typing import Tuple, Optional


class BreedenLitzenbergerRND:
    """
    Extract risk-neutral density using Breeden-Litzenberger theorem.

    The risk-neutral density is proportional to the second derivative
    of call prices with respect to strike:
    f(K) = e^{rτ} * ∂²C/∂K²
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the Breeden-Litzenberger extractor.

        Parameters
        ----------
        risk_free_rate : float, optional
            Risk-free interest rate (default: 0.02)
        """
        self.risk_free_rate = risk_free_rate

    def extract_rnd(
        self,
        strikes: np.ndarray,
        call_prices: np.ndarray,
        spot: float,
        time_to_expiry: Optional[float] = None,
        interpolate_grid: bool = True,
        n_points: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract risk-neutral density from call option prices.

        Parameters
        ----------
        strikes : np.ndarray
            Strike prices
        call_prices : np.ndarray
            Call option prices
        spot : float
            Current spot price
        time_to_expiry : float, optional
            Time to expiration in years. If None, extracted from data.
        interpolate_grid : bool, optional
            Whether to interpolate to a finer grid (default: True)
        n_points : int, optional
            Number of points for interpolation grid (default: 1000)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Strike prices and corresponding risk-neutral densities
        """
        # Sort by strike
        idx = np.argsort(strikes)
        strikes_sorted = strikes[idx]
        prices_sorted = call_prices[idx]

        # Remove duplicates
        strikes_unique, idx_unique = np.unique(strikes_sorted, return_index=True)
        prices_unique = prices_sorted[idx_unique]

        # Interpolate for smoother density
        f = interpolate.CubicSpline(
            strikes_unique,
            prices_unique,
            bc_type="natural",
            extrapolate=False,
        )

        # Create grid for differentiation
        if interpolate_grid:
            k_min = max(strikes_unique.min(), spot * 0.5)
            k_max = min(strikes_unique.max(), spot * 1.5)
            k_fine = np.linspace(k_min, k_max, n_points)
        else:
            k_fine = strikes_unique

        # Get interpolated prices
        c_fine = f(k_fine)

        # Calculate first derivative (delta)
        dC_dK = np.gradient(c_fine, k_fine)

        # Calculate second derivative (gamma)
        d2C_dK2 = np.gradient(dC_dK, k_fine)

        # Apply Breeden-Litzenberger formula
        if time_to_expiry is None:
            # Use default if not provided (will be overridden if data available)
            rnd = d2C_dK2
        else:
            rnd = np.exp(self.risk_free_rate * time_to_expiry) * d2C_dK2

        # Ensure non-negative density
        rnd = np.maximum(rnd, 0)

        # Normalize to integrate to 1
        rnd = self._normalize_density(k_fine, rnd)

        return k_fine, rnd

    def _normalize_density(
        self,
        strikes: np.ndarray,
        density: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize density to integrate to 1.

        Parameters
        ----------
        strikes : np.ndarray
            Strike prices
        density : np.ndarray
            Risk-neutral density values

        Returns
        -------
        np.ndarray
            Normalized density
        """
        integral = self._trapezoidal_integration(density, strikes)

        if integral > 0:
            density = density / integral

        return density

    @staticmethod
    def _trapezoidal_integration(
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
