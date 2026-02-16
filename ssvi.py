"""SSVI (Surface SVI) model implementation."""

import numpy as np
from scipy import optimize
from typing import Tuple, Optional, List


class SSVI:
    """
    Surface SVI parameterization with no-arbitrage constraints.

    Total variance: w(k,θ) = θ/2 * (1 + ρφ(θ)k + sqrt((φ(θ)k + ρ)^2 + (1-ρ^2)))
    where φ(θ) = η / θ^γ (power law) or φ(θ) = η (constant)
    """

    def __init__(self, power_law: bool = True):
        """
        Initialize SSVI model.

        Parameters
        ----------
        power_law : bool, optional
            Whether to use power law for φ(θ) (default: True)
        """
        self.power_law = power_law
        self.params = None
        self.sse = None

    def total_variance(
        self,
        k: np.ndarray,
        theta: float,
        rho: float,
        phi: float,
        gamma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Calculate SSVI total variance.

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness values
        theta : float
            ATM total variance
        rho : float
            Correlation parameter
        phi : float
        gamma : float, optional
            Power law parameter (required if power_law=True)

        Returns
        -------
        np.ndarray
            Total variance values
        """
        if self.power_law:
            if gamma is None:
                raise ValueError("gamma required when power_law=True")
            phi_theta = phi / (theta ** gamma)
        else:
            phi_theta = phi

        return 0.5 * theta * (
            1 + rho * phi_theta * k +
            np.sqrt((phi_theta * k + rho) ** 2 + (1 - rho ** 2))
        )

    def calibrate(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        forward: float,
        time_to_expiry: float,
        initial_guess: Optional[List[float]] = None,
    ) -> Tuple[List[float], float]:
        """
        Calibrate SSVI parameters to market implied volatilities.

        Parameters
        ----------
        strikes : np.ndarray
            Strike prices
        ivs : np.ndarray
            Implied volatilities
        forward : float
            Forward price
        time_to_expiry : float
            Time to expiration in years
        initial_guess : list, optional
            Initial parameter guesses [theta, rho, phi, gamma] if power_law,
            otherwise [theta, rho, phi]

        Returns
        -------
        Tuple[List[float], float]
            Calibrated parameters and sum of squared errors
        """
        # Calculate log-moneyness and total variance
        k = np.log(strikes / forward)
        w_market = ivs ** 2 * time_to_expiry

        # Set initial guess
        if initial_guess is None:
            atm_idx = np.argmin(np.abs(k))
            theta_init = w_market[atm_idx] if atm_idx < len(w_market) else 0.04

            if self.power_law:
                initial_guess = [theta_init, -0.7, 2.0, 0.5]
            else:
                initial_guess = [theta_init, -0.7, 2.0]

        def objective(params):
            if self.power_law:
                theta, rho, phi, gamma = params
                w_pred = self.total_variance(k, theta, rho, phi, gamma)
            else:
                theta, rho, phi = params
                w_pred = self.total_variance(k, theta, rho, phi)

            # Add penalty for arbitrage violations
            penalty = self._arbitrage_penalty(params)

            return np.sum((w_pred - w_market) ** 2) + penalty

        # Parameter bounds
        if self.power_law:
            bounds = [
                (0.001, 0.5),   # theta
                (-0.99, 0.99),  # rho
                (0.1, 10.0),    # phi
                (0.1, 1.0),     # gamma
            ]
        else:
            bounds = [
                (0.001, 0.5),   # theta
                (-0.99, 0.99),  # rho
                (0.1, 10.0),    # phi
            ]

        result = optimize.minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
        )

        self.params = result.x
        self.sse = result.fun

        return self.params.tolist(), self.sse

    def _arbitrage_penalty(self, params: List[float]) -> float:
        """
        Calculate penalty for no-arbitrage violations.

        Parameters
        ----------
        params : List[float]
            Model parameters

        Returns
        -------
        float
            Penalty value
        """
        penalty = 0

        if self.power_law:
            theta, rho, phi, gamma = params

            # Butterfly arbitrage condition
            if theta * (1 + abs(rho)) >= 4:
                penalty += 1000 * (theta * (1 + abs(rho)) - 4)

            # Calendar spread arbitrage condition
            if gamma <= 0 or gamma >= 1:
                penalty += 1000 * (abs(gamma - 0.5) + 0.1)

        else:
            theta, rho, phi = params

            # Butterfly arbitrage condition
            if theta * (1 + abs(rho)) >= 4:
                penalty += 1000 * (theta * (1 + abs(rho)) - 4)

        return penalty

    def get_volatility_curve(
        self,
        strikes: np.ndarray,
        forward: float,
        time_to_expiry: float,
    ) -> np.ndarray:
        """
        Get implied volatility curve from calibrated parameters.

        Parameters
        ----------
        strikes : np.ndarray
            Strike prices
        forward : float
            Forward price
        time_to_expiry : float
            Time to expiration in years

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

        k = np.log(strikes / forward)

        if self.power_law:
            theta, rho, phi, gamma = self.params
            w = self.total_variance(k, theta, rho, phi, gamma)
        else:
            theta, rho, phi = self.params
            w = self.total_variance(k, theta, rho, phi)

        return np.sqrt(w / time_to_expiry)

    def extract_rnd(
        self,
        strikes: np.ndarray,
        forward: float,
        time_to_expiry: float,
        n_points: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract risk-neutral density from calibrated SSVI parameters.

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

        # Generate fine grid in log-moneyness
        k_min = np.log(min(strikes) / forward)
        k_max = np.log(max(strikes) / forward)
        k_fine = np.linspace(k_min, k_max, n_points)

        # Calculate total variance and derivatives
        if self.power_law:
            theta, rho, phi, gamma = self.params
            w = self.total_variance(k_fine, theta, rho, phi, gamma)
        else:
            theta, rho, phi = self.params
            w = self.total_variance(k_fine, theta, rho, phi)

        sqrt_w = np.sqrt(w)

        # Numerical derivatives
        dw_dk = np.gradient(w, k_fine)
        d2w_dk2 = np.gradient(dw_dk, k_fine)

        # RND formula
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
