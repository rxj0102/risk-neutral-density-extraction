"""Module for fetching and preprocessing options data."""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import warnings

warnings.filterwarnings("ignore")


class OptionDataFetcher:
    """Fetch and preprocess options data from Yahoo Finance."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the data fetcher.

        Parameters
        ----------
        risk_free_rate : float, optional
            Risk-free interest rate (default: 0.02)
        """
        self.risk_free_rate = risk_free_rate

    def fetch_option_chain(
        self,
        ticker: str,
        expiration_date: Optional[str] = None,
        min_days_to_expiry: int = 7,
        max_days_to_expiry: int = 60,
    ) -> Tuple[pd.DataFrame, float, str]:
        """
        Fetch option chain data for a given ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        expiration_date : str, optional
            Specific expiration date (YYYY-MM-DD). If None, selects nearest
            expiration with sufficient liquidity.
        min_days_to_expiry : int, optional
            Minimum days to expiration for option selection
        max_days_to_expiry : int, optional
            Maximum days to expiration for option selection

        Returns
        -------
        Tuple[pd.DataFrame, float, str]
            Processed options data, current stock price, expiration date
        """
        stock = yf.Ticker(ticker)

        # Get current price
        current_price = stock.history(period="1d")["Close"].iloc[-1]

        # Get expiration dates
        expirations = stock.options
        if not expirations:
            raise ValueError(f"No options found for ticker {ticker}")

        # Select expiration date
        if expiration_date is None:
            expiration_date = self._select_expiration(
                expirations, min_days_to_expiry, max_days_to_expiry
            )

        # Fetch option chain
        opt_chain = stock.option_chain(expiration_date)

        # Combine and process options
        options = self._combine_options(opt_chain)
        options = self._preprocess_options(options, current_price, expiration_date)

        return options, current_price, expiration_date

    def _select_expiration(
        self,
        expirations: list,
        min_days: int,
        max_days: int,
    ) -> str:
        """
        Select the most appropriate expiration date.

        Parameters
        ----------
        expirations : list
            List of expiration dates
        min_days : int
            Minimum days to expiry
        max_days : int
            Maximum days to expiry

        Returns
        -------
        str
            Selected expiration date
        """
        today = datetime.now().date()
        valid_expirations = []

        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            days_to_expiry = (exp_date - today).days

            if min_days <= days_to_expiry <= max_days:
                valid_expirations.append((exp, days_to_expiry))

        if not valid_expirations:
            # Fallback to nearest expiration
            return expirations[0]

        # Select expiration with most liquidity (usually nearest)
        valid_expirations.sort(key=lambda x: x[1])
        return valid_expirations[0][0]

    def _combine_options(self, opt_chain: Any) -> pd.DataFrame:
        """
        Combine calls and puts into a single DataFrame.

        Parameters
        ----------
        opt_chain : yfinance.Options
            Option chain object

        Returns
        -------
        pd.DataFrame
            Combined options data
        """
        calls = opt_chain.calls.copy()
        puts = opt_chain.puts.copy()

        calls["optionType"] = "call"
        puts["optionType"] = "put"

        options = pd.concat([calls, puts], ignore_index=True)
        return options

    def _preprocess_options(
        self,
        options: pd.DataFrame,
        current_price: float,
        expiration_date: str,
    ) -> pd.DataFrame:
        """
        Clean and preprocess option data.

        Parameters
        ----------
        options : pd.DataFrame
            Raw options data
        current_price : float
            Current stock price
        expiration_date : str
            Expiration date

        Returns
        -------
        pd.DataFrame
            Processed options data
        """
        df = options.copy()

        # Calculate mid price
        df["midPrice"] = (df["bid"] + df["ask"]) / 2

        # Calculate moneyness
        df["moneyness"] = df["strike"] / current_price

        # Calculate time to expiration (in years)
        exp_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        days_to_expiry = (exp_date - today).days
        df["tau"] = days_to_expiry / 365.25

        # Filter out invalid options
        df = df[df["midPrice"] > 0]
        df = df[df["bid"] > 0]
        df = df[df["ask"] > 0]

        # Filter out extreme moneyness
        df = df[(df["moneyness"] > 0.5) & (df["moneyness"] < 2.0)]

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=["strike", "optionType"])
        df = df.sort_values(["strike", "optionType"])

        return df

    def fetch_historical_data(
        self,
        ticker: str,
        years: int = 5,
    ) -> pd.DataFrame:
        """
        Fetch historical price data for the ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        years : int, optional
            Number of years of historical data (default: 5)

        Returns
        -------
        pd.DataFrame
            Historical price data with returns
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        stock = yf.Ticker(ticker)
        hist_data = stock.history(start=start_date, end=end_date)

        # Calculate returns
        hist_data["Returns"] = hist_data["Close"].pct_change()
        hist_data["Log_Returns"] = np.log(hist_data["Close"] / hist_data["Close"].shift(1))

        return hist_data
