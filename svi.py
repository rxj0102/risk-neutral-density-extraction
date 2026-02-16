"""SVI (Stochastic Volatility Inspired) model implementation."""

import numpy as np
from scipy import optimize
from typing import Tuple, Optional, List


class SVI:
    """
    SVI parameterization for implied volatility surface.

    Total variance: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    where k = log(K/F) is the log-moneyness
    """

    def __init__(self):
        """Initialize the SVI model."""
        self.params = None
        self.sse = None

    def volatility(self, k: np.ndarray, a: float, b: float, rho: float,
                   m: float, sigma: float) -> np.ndarray:
        """
        Calculate SVI total variance.

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness values
        a, b, rho, m, sigma : float
            SVI parameters

        Returns
        -------
        np.ndarray
            Total variance values
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def calibrate(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        forward: float,
        initial_guess: Optional[List[float]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[List[float], float]:
        """
        Calibrate SVI parameters to market implied volatilities.

        Parameters
        ----------
        strikes : np.ndarray
            Strike prices
        ivs : np.ndarray
            Implied volatilities
        forward : float
            Forward price
        initial_guess : list, optional
            Initial parameter guesses [a, b, rho, m, sigma]
        bounds : list, optional
            Parameter bounds [(a_min, a_max), ...]

        Returns
        -------
        Tuple[List[float], float]
            Calibrated parameters and sum of squared errors
        """
        if initial_guess is None:
            initial_guess = [0.04, 0.1, -0.7, 0.0, 0.1]

        if bounds is None:
            bounds = [
                (0.001, 0.5),   # a
                (0.001, 1.0),   # b
                (-0.99, 0.99),  # rho
                (-1.0, 1.0),    # m
                (0.001, 0.5),   # sigma
            ]

        # Calculate log-moneyness
        k = np.log(strikes / forward)

        def objective(params):
            a, b, rho, m, sigma = params
            w_pred = self.volatility(k, a, b, rho, m, sigma)

            # Add penalties for constraint violations
            penalty = 0
            if b <= 0:
                penalty += 1000 * (abs(b) + 0.01)
            if sigma <= 0:
                penalty += 1000 * (abs(sigma) + 0.01)
            if abs(rho) >= 1:
                penalty += 1000 * (abs(rho) - 0.99)

            return np.sum((w_pred - ivs ** 2) ** 2) + penalty

        result = optimize.minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
        )

        self.params = result.x
        self.sse = result.fun

        return self.params.tolist(), self.sse

    def get_volatility_curve(
        self,
        strikes: np.ndarray,
        forward: float,
    ) -> np.ndarray:
        """
        Get implied volatility curve from calibrated parameters.

        Parameters
        ----------
        strikes : np.ndarray
            Strike prices
        forward : float
            Forward price

        Returns
        -------
        np.ndarray
            Implied volatilities

        Raises
        ------
        ValueError
            If model hasn't been calibrated
        """
        if self.params is None:
            raise ValueError("Model must be calibrated first")

        a, b, rho, m, sigma = self.params
        k = np.log(strikes / forward)
        w = self.volatility(k, a, b, rho, m, sigma)

        return np.sqrt(w)

    def extract_rnd(
        self,
        strikes: np.ndarray,
        forward: float,
        time_to_expiry: float,
        n_points: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract risk-neutral density from calibrated SVI parameters.

        Parameters
        ----------
        strikes : np.ndarray
            Strike prices (used for range)
        forward : float
            Forward price
        time_to_expiry : float
            Time to expiration in years
        n_points : int, optional
            Number of points for density grid (default: 1000)

        Returns
        -------
        Tuple containing:
            strikes_fine : np.ndarray
                Fine strike grid
            rnd : np.ndarray
                Risk-neutral density
            w : np.ndarray
                Total variance
            dw_dk : np.ndarray
                First derivative of total variance
            d2w_dk2 : np.ndarray
                Second derivative of total variance
        """
        if self.params is None:
            raise ValueError("Model must be calibrated first")

        a, b, rho, m, sigma = self.params

        # Generate fine grid in log-moneyness
        k_min = np.log(min(strikes) / forward)
        k_max = np.log(max(strikes) / forward)
        k_fine = np.linspace(k_min, k_max, n_points)

        # Calculate total variance and derivatives
        w = self.volatility(k_fine, a, b, rho, m, sigma)
        sqrt_w = np.sqrt(w)

        # First derivative dw/dk
        dw_dk = b * (rho + (k_fine - m) / np.sqrt((k_fine - m) ** 2 + sigma ** 2))

        # Second derivative d²w/dk²
        d2w_dk2 = b * sigma ** 2 / ((k_fine - m) ** 2 + sigma ** 2) ** 1.5

        # RND formula from SVI
        term1 = 1 - (k_fine / (2 * w)) * dw_dk
        term2 = 0.25 * (-0.25 - 1 / w + k_fine ** 2 / w ** 2) * dw_dk ** 2
        term3 = 0.5 * d2w_dk2

        density = (1 / (np.sqrt(2 * np.pi) * sqrt_w)) * (term1 + term2 + term3)

        # Ensure positivity
        density = np.maximum(density, 0)

        # Convert back to strike space
        strikes_fine = forward * np.exp(k_fine)

        # Normalize density
        integral = self._trapezoidal_integration(density, strikes_fine)
        if integral > 0:
            density = density / integral

        return strikes_fine, density, w, dw_dk, d2w_dk2

    @staticmethod
    def _trapezoidal_integration(
        y: np.ndarray,
        x: np.ndarray,
    ) -> float:
        """Perform trapezoidal numerical integration."""
        return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
