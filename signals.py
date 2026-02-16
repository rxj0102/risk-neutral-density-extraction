"""Trading signal generation based on risk premia."""

from typing import Dict, Any, Optional


class TradingSignalGenerator:
    """Generate trading signals based on risk premia analysis."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize signal generator.

        Parameters
        ----------
        thresholds : Dict[str, float], optional
            Signal thresholds for different metrics
        """
        self.thresholds = thresholds or {
            'vrp_high': 1.5,      # Sell signal when VRP ratio > 1.5
            'vrp_low': 0.8,        # Buy signal when VRP ratio < 0.8
            'crash_high': 2.0,     # Expensive protection when ratio > 2
            'crash_low': 0.5,       # Cheap protection when ratio < 0.5
            'skew_high': 0.3,       # Negative skew overpriced
            'skew_low': -0.3,       # Positive skew undervalued
        }

    def generate_signals(
        self,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on risk metrics.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Dictionary of risk metrics

        Returns
        -------
        Dict[str, Any]
            Dictionary containing trading signals and explanations
        """
        signals = {
            'volatility': self._volatility_signal(metrics),
            'tail_risk': self._tail_risk_signal(metrics),
            'skew': self._skew_signal(metrics),
            'overall': 'NEUTRAL',
        }

        # Determine overall signal
            signal_values = [
                signals['volatility']['signal'],
                signals['tail_risk']['signal'],
                signals['skew']['signal'],
            ]

        if 'BUY' in signal_values and 'SELL' not in signal_values:
            signals['overall'] = 'BUY'
        elif 'SELL' in signal_values and 'BUY' not in signal_values:
            signals['overall'] = 'SELL'
        elif 'BUY' in signal_values and 'SELL' in signal_values:
            signals['overall'] = 'MIXED'

        return signals

    def _volatility_signal(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate volatility trading signal.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Risk metrics dictionary

        Returns
        -------
        Dict[str, str]
            Signal and explanation
        """
        vrp_ratio = metrics.get('vrp_ratio', 1.0)

        if vrp_ratio > self.thresholds['vrp_high']:
            return {
                'signal': 'SELL',
                'explanation': f'Implied vol {vrp_ratio:.2f}x historical - consider selling options',
            }
        elif vrp_ratio < self.thresholds['vrp_low']:
            return {
                'signal': 'BUY',
                'explanation': f'Implied vol {vrp_ratio:.2f}x historical - consider buying options',
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'explanation': 'Volatility fairly priced',
            }

    def _tail_risk_signal(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate tail risk trading signal.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Risk metrics dictionary

        Returns
        -------
        Dict[str, str]
            Signal and explanation
        """
        crash_ratio = metrics.get('crash_risk_ratio', 1.0)

        if crash_ratio > self.thresholds['crash_high']:
            return {
                'signal': 'SELL',
                'explanation': f'Crash probability {crash_ratio:.2f}x historical - consider selling puts',
            }
        elif crash_ratio < self.thresholds['crash_low']:
            return {
                'signal': 'BUY',
                'explanation': f'Crash probability {crash_ratio:.2f}x historical - consider buying puts',
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'explanation': 'Tail risk fairly priced',
            }

    def _skew_signal(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate skew trading signal.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Risk metrics dictionary

        Returns
        -------
        Dict[str, str]
            Signal and explanation
        """
        skew_diff = metrics.get('skewness_diff', 0)

        if skew_diff > self.thresholds['skew_high']:
            return {
                'signal': 'SELL',
                'explanation': 'Negative skew overpriced - consider put spreads',
            }
        elif skew_diff < self.thresholds['skew_low']:
            return {
                'signal': 'BUY',
                'explanation': 'Positive skew undervalued - consider call spreads',
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'explanation': 'Skew fairly priced',
            }
