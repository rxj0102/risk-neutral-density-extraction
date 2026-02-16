"""Unit tests for RND extraction models."""

import pytest
import numpy as np
from src.models.breeden_litzenberger import BreedenLitzenbergerRND
from src.models.svi import SVI


class TestBreedenLitzenberger:
    """Test Breeden-Litzenberger RND extraction."""

    def test_extraction(self):
        """Test basic extraction functionality."""
        bl = BreedenLitzenbergerRND(risk_free_rate=0.02)

        # Create simple test data
        strikes = np.linspace(80, 120, 41)
        # Simulate call prices (simplified)
        call_prices = np.maximum(100 - strikes, 0) + 1

        k_fine, rnd = bl.extract_rnd(
            strikes=strikes,
            call_prices=call_prices,
            spot=100,
        )

        assert len(k_fine) > len(strikes)
        assert np.all(rnd >= 0)
        assert np.isclose(np.sum(rnd) * (k_fine[1] - k_fine[0]), 1.0, rtol=0.1)

    def test_normalization(self):
        """Test density normalization."""
        bl = BreedenLitzenbergerRND()

        x = np.linspace(0, 10, 1000)
        y = np.exp(-(x - 5) ** 2 / 2)

        normalized = bl._normalize_density(x, y)
        integral = bl._trapezoidal_integration(normalized, x)

        assert np.isclose(integral, 1.0)


class TestSVI:
    """Test SVI model."""

    def test_calibration(self):
        """Test SVI calibration."""
        svi = SVI()

        # Generate synthetic data
        strikes = np.linspace(80, 120, 21)
        forward = 100
        k = np.log(strikes / forward)

        # True parameters
        true_params = [0.04, 0.1, -0.7, 0.0, 0.1]
        w_true = svi.volatility(k, *true_params)
        iv_true = np.sqrt(w_true)

        # Calibrate
        params, sse = svi.calibrate(strikes, iv_true, forward)

        # Check calibration accuracy
        assert sse < 1e-4
        for p_true, p_est in zip(true_params, params):
            assert np.abs(p_true - p_est) < 0.01

    def test_rnd_extraction(self):
        """Test RND extraction from SVI."""
        svi = SVI()

        # Calibrate first
        strikes = np.linspace(80, 120, 21)
        forward = 100
        k = np.log(strikes / forward)
        params = [0.04, 0.1, -0.7, 0.0, 0.1]
        w = svi.volatility(k, *params)
        iv = np.sqrt(w)
        svi.calibrate(strikes, iv, forward)

        # Extract RND
        k_fine, rnd, w_fine, dw_dk, d2w_dk2 = svi.extract_rnd(
            strikes,
            forward,
            time_to_expiry=0.25,
        )

        assert len(k_fine) > len(strikes)
        assert np.all(rnd >= 0)
        assert np.all(w_fine >= 0)
        assert np.all(dw_dk != 0)
        assert np.all(d2w_dk2 != 0)
