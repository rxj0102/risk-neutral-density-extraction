# Risk-Neutral Density Extraction & Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive implementation of risk-neutral density extraction from options data, featuring multiple methodologies, advanced risk metrics, and trading signal generation. This project bridges theoretical finance with practical trading applications.

## 🌟 Features

### 📊 Density Extraction Methods
- **Breeden-Litzenberger**: Non-parametric extraction using second derivative of call prices
- **SVI (Stochastic Volatility Inspired)**: 5-parameter volatility surface model
- **SSVI (Surface SVI)**: 3-parameter surface SVI with no-arbitrage constraints

### 📈 Risk Metrics
- Higher-order moments (skewness, kurtosis)
- Value at Risk (VaR) at 95% and 99% confidence levels
- Expected Shortfall (CVaR)
- Drawdown probability analysis (5%, 10%, 20%, 30% levels)
- Tail risk ratios and scenario probabilities

### 💹 Trading Signals
- Volatility Risk Premium (VRP) signals
- Crash probability premium analysis
- Skewness premium trading signals
- Automated trade recommendations

### 📉 Visualizations
- Density comparisons (linear and log scale)
- Cumulative distribution functions
- Tail risk heatmaps
- Trading signal indicators
- Parameter evolution plots

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/risk-neutral-density.git
cd risk-neutral-density

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
