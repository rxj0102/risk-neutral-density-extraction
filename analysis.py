#!/usr/bin/env python
"""Main script to run complete RND analysis."""

import argparse
import logging
from pathlib import Path
import json
import numpy as np

from src.data.data_fetcher import OptionDataFetcher
from src.models.breeden_litzenberger import BreedenLitzenbergerRND
from src.models.svi import SVI
from src.models.ssvi import SSVI
from src.utils.metrics import (
    calculate_moments,
    calculate_tail_metrics,
    calculate_advanced_tail_metrics,
    calculate_variance_risk_premium,
)
from src.utils.visualization import RNDVisualizer
from src.trading.signals import TradingSignalGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run RND analysis')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker symbol')
    parser.add_argument('--expiry', type=int, default=30,
                       help='Days to expiration')
    parser.add_argument('--risk-free-rate', type=float, default=0.02,
                       help='Risk-free interest rate')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    return parser.parse_args()


def main():
    """Run complete RND analysis."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Starting analysis for {args.ticker}")

    # Fetch data
    fetcher = OptionDataFetcher(risk_free_rate=args.risk_free_rate)
    options, spot, expiry = fetcher.fetch_option_chain(args.ticker)
    historical = fetcher.fetch_historical_data(args.ticker)

    logger.info(f"Fetched {len(options)} options, spot price: {spot:.2f}")

    # Extract RND using different methods
    results = {}

    # Breeden-Litzenberger
    logger.info("Extracting RND using Breeden-Litzenberger...")
    bl = BreedenLitzenbergerRND(risk_free_rate=args.risk_free_rate)
    calls = options[options['optionType'] == 'call']
    k_bl, rnd_bl = bl.extract_rnd(
        strikes=calls['strike'].values,
        call_prices=calls['midPrice'].values,
        spot=spot,
        time_to_expiry=args.expiry / 365.25,
    )
    results['bl'] = {'strikes': k_bl, 'density': rnd_bl}

    # SVI
    logger.info("Calibrating SVI model...")
    svi = SVI()
    ivs = calls['iv'].values if 'iv' in calls.columns else None
    if ivs is None:
        logger.warning("No IV data available, skipping SVI")
    else:
        forward = spot * np.exp(args.risk_free_rate * args.expiry / 365.25)
        params, sse = svi.calibrate(
            strikes=calls['strike'].values,
            ivs=ivs,
            forward=forward,
        )
        logger.info(f"SVI calibration SSE: {sse:.6f}")
        k_svi, rnd_svi, *_ = svi.extract_rnd(
            strikes=calls['strike'].values,
            forward=forward,
            time_to_expiry=args.expiry / 365.25,
        )
        results['svi'] = {'strikes': k_svi, 'density': rnd_svi}

    # SSVI
    logger.info("Calibrating SSVI model...")
    ssvi = SSVI(power_law=True)
    if ivs is not None:
        forward = spot * np.exp(args.risk_free_rate * args.expiry / 365.25)
        params, sse = ssvi.calibrate(
            strikes=calls['strike'].values,
            ivs=ivs,
            forward=forward,
            time_to_expiry=args.expiry / 365.25,
        )
        logger.info(f"SSVI calibration SSE: {sse:.6f}")
        k_ssvi, rnd_ssvi, *_ = ssvi.extract_rnd(
            strikes=calls['strike'].values,
            forward=forward,
            time_to_expiry=args.expiry / 365.25,
        )
        results['ssvi'] = {'strikes': k_ssvi, 'density': rnd_ssvi}

    # Calculate metrics
    logger.info("Calculating risk metrics...")
    metrics = {}

    for name, result in results.items():
        mean, std, skew, kurt = calculate_moments(
            result['strikes'],
            result['density'],
        )
        tail = calculate_advanced_tail_metrics(
            result['strikes'],
            result['density'],
            spot,
        )
        metrics[name] = {
            'mean': mean,
            'std': std,
            'skewness': skew,
            'kurtosis': kurt,
            **tail,
        }

    # Generate trading signals
    logger.info("Generating trading signals...")
    signal_generator = TradingSignalGenerator()
    signals = signal_generator.generate_signals(metrics)

    # Save results
    logger.info(f"Saving results to {output_dir}")
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(output_dir / 'signals.json', 'w') as f:
        json.dump(signals, f, indent=2)

    # Generate visualizations
    logger.info("Generating visualizations...")
    viz = RNDVisualizer()

    # Density comparison
    densities = {k: v['density'] for k, v in results.items()}
    strikes_dict = {k: v['strikes'] for k, v in results.items()}
    fig = viz.plot_density_comparison(
        densities,
        strikes_dict,
        spot,
        forward=forward if 'forward' in locals() else None,
    )
    fig.savefig(output_dir / 'density_comparison.png', dpi=150, bbox_inches='tight')

    # Trading signals
    fig = viz.plot_trading_signals(metrics)
    fig.savefig(output_dir / 'trading_signals.png', dpi=150, bbox_inches='tight')

    logger.info("Analysis complete!")
    print("\n" + "="*60)
    print("TRADING SIGNALS SUMMARY")
    print("="*60)
    print(f"Overall Signal: {signals['overall']}")
    print(f"Volatility: {signals['volatility']['signal']} - {signals['volatility']['explanation']}")
    print(f"Tail Risk: {signals['tail_risk']['signal']} - {signals['tail_risk']['explanation']}")
    print(f"Skew: {signals['skew']['signal']} - {signals['skew']['explanation']}")
    print("="*60)


if __name__ == '__main__':
    main()
