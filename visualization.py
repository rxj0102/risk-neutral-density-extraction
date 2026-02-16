"""Visualization utilities for RND analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import seaborn as sns


class RNDVisualizer:
    """Visualization tools for risk-neutral density analysis."""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.

        Parameters
        ----------
        style : str, optional
            Matplotlib style (default: 'seaborn-v0_8-darkgrid')
        """
        plt.style.use(style)

    def plot_density_comparison(
        self,
        densities: Dict[str, np.ndarray],
        strikes: Dict[str, np.ndarray],
        current_price: float,
        forward_price: Optional[float] = None,
        title: str = "Risk-Neutral Density Comparison",
        log_scale: bool = False,
        xlim: Optional[tuple] = None,
    ) -> plt.Figure:
        """
        Plot comparison of multiple densities.

        Parameters
        ----------
        densities : Dict[str, np.ndarray]
            Dictionary of density arrays with labels as keys
        strikes : Dict[str, np.ndarray]
            Dictionary of strike arrays with matching keys
        current_price : float
            Current asset price
        forward_price : float, optional
            Forward price
        title : str, optional
            Plot title
        log_scale : bool, optional
            Whether to use log scale for y-axis
        xlim : tuple, optional
            x-axis limits

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(densities)))

        for (label, density), color in zip(densities.items(), colors):
            ax.plot(
                strikes[label],
                density,
                label=label,
                linewidth=2,
                alpha=0.7,
                color=color,
            )
            ax.fill_between(
                strikes[label],
                0,
                density,
                alpha=0.2,
                color=color,
            )

        ax.axvline(current_price, color='k', linestyle='--', alpha=0.5, label='Spot')
        if forward_price is not None:
            ax.axvline(forward_price, color='b', linestyle=':', alpha=0.7, label='Forward')

        if log_scale:
            ax.set_yscale('log')

        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Density' + (' (log scale)' if log_scale else ''))
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if xlim:
            ax.set_xlim(xlim)

        return fig

    def plot_cdf_comparison(
        self,
        densities: Dict[str, np.ndarray],
        strikes: Dict[str, np.ndarray],
        current_price: float,
        title: str = "Cumulative Distribution Comparison",
    ) -> plt.Figure:
        """
        Plot comparison of cumulative distribution functions.

        Parameters
        ----------
        densities : Dict[str, np.ndarray]
            Dictionary of density arrays with labels as keys
        strikes : Dict[str, np.ndarray]
            Dictionary of strike arrays with matching keys
        current_price : float
            Current asset price
        title : str, optional
            Plot title

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(densities)))

        for (label, density), color in zip(densities.items(), colors):
            cdf = np.cumsum(density) * (strikes[label][1] - strikes[label][0])
            ax.plot(
                strikes[label],
                cdf,
                label=label,
                linewidth=2,
                color=color,
            )

        ax.axvline(current_price, color='k', linestyle='--', alpha=0.5, label='Spot')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Median')

        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_tail_risk_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Tail Risk Metrics Comparison",
    ) -> plt.Figure:
        """
        Plot comparison of tail risk metrics.

        Parameters
        ----------
        metrics : Dict[str, Dict[str, float]]
            Dictionary of metric dictionaries with labels as keys
        title : str, optional
            Plot title

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        metric_names = ['downside_prob', 'upside_prob', 'crash_prob']
        display_names = ['Downside\n(10%)', 'Upside\n(10%)', 'Crash\n(20%)']

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metric_names))
        width = 0.8 / len(metrics)

        for i, (label, metric_dict) in enumerate(metrics.items()):
            values = [metric_dict.get(m, 0) for m in metric_names]
            offset = (i - len(metrics) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=label,
                alpha=0.7,
            )

            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

        ax.set_xlabel('Tail Risk Metric')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        return fig

    def plot_trading_signals(
        self,
        signals: Dict[str, Any],
        title: str = "Trading Signals",
    ) -> plt.Figure:
        """
        Plot trading signals.

        Parameters
        ----------
        signals : Dict[str, Any]
            Dictionary of trading signals
        title : str, optional
            Plot title

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Volatility Risk Premium
        vrp_data = [
            signals.get('implied_vol', 0),
            signals.get('realized_vol', 0),
        ]
        vrp_labels = ['Implied Vol', 'Historical Vol']
        vrp_colors = ['red' if signals.get('vrp_ratio', 1) > 1.5 else
                     'green' if signals.get('vrp_ratio', 1) < 0.8 else 'gray']

        axes[0].bar(vrp_labels, vrp_data, color=vrp_colors, alpha=0.7)
        axes[0].set_ylabel('Volatility')
        axes[0].set_title('Volatility Risk Premium')
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(vrp_data):
            axes[0].text(i, v, f'{v:.2%}', ha='center', va='bottom')

        # Crash Probability
        crash_data = [
            signals.get('implied_crash_prob', 0),
            signals.get('historical_crash_prob', 0),
        ]
        crash_labels = ['Implied\nCrash Prob', 'Historical\nCrash Prob']
        crash_ratio = signals.get('crash_risk_ratio', 1)
        crash_colors = ['red' if crash_ratio > 2 else
                       'green' if crash_ratio < 0.5 else 'gray']

        axes[1].bar(crash_labels, crash_data, color=crash_colors, alpha=0.7)
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Crash Probability Premium')
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(crash_data):
            axes[1].text(i, v, f'{v:.2%}', ha='center', va='bottom')

        # Skewness
        skew_data = [
            signals.get('implied_skewness', 0),
            signals.get('historical_skewness', 0),
        ]
        skew_labels = ['Implied\nSkewness', 'Historical\nSkewness']
        skew_diff = signals.get('skewness_diff', 0)
        skew_colors = ['red' if skew_diff > 0.3 else
                      'green' if skew_diff < -0.3 else 'gray']

        axes[2].bar(skew_labels, skew_data, color=skew_colors, alpha=0.7)
        axes[2].set_ylabel('Skewness')
        axes[2].set_title('Skewness Premium')
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        for i, v in enumerate(skew_data):
            axes[2].text(i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.suptitle(title)
        plt.tight_layout()

        return fig

    def plot_parameter_evolution(
        self,
        parameters: Dict[str, np.ndarray],
        title: str = "Parameter Evolution",
    ) -> plt.Figure:
        """
        Plot evolution of model parameters.

        Parameters
        ----------
        parameters : Dict[str, np.ndarray]
            Dictionary of parameter arrays with labels as keys
        title : str, optional
            Plot title

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        n_params = len(parameters)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params))

        if n_params == 1:
            axes = [axes]

        for ax, (param_name, values) in zip(axes, parameters.items()):
            ax.plot(values, linewidth=2, marker='o')
            ax.set_ylabel(param_name)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time')
        plt.suptitle(title)
        plt.tight_layout()

        return fig
