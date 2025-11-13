"""
Factor calculation module for quantitative factors.
"""

import pandas as pd
import numpy as np


class FactorCalculator:
    """Calculate various factors for quantitative analysis."""
    
    def __init__(self, price_data):
        """
        Initialize with price data.
        
        Args:
            price_data (dict): Dictionary with ticker as key and price DataFrame as value
        """
        self.price_data = price_data
        self.factors = {}
    
    def calculate_momentum(self, lookback=20):
        """
        Calculate momentum factor (price momentum).
        
        Args:
            lookback (int): Number of periods for momentum calculation
        
        Returns:
            pd.DataFrame: Momentum factor for each asset
        """
        momentum = {}
        for ticker, df in self.price_data.items():
            # Calculate returns
            returns = df['Adj Close'].pct_change(lookback)
            momentum[ticker] = returns
        
        return pd.DataFrame(momentum)
    
    def calculate_mean_reversion(self, lookback=20):
        """
        Calculate mean reversion factor.
        
        Args:
            lookback (int): Rolling window for mean reversion
        
        Returns:
            pd.DataFrame: Mean reversion factor (how far from moving average)
        """
        mean_reversion = {}
        for ticker, df in self.price_data.items():
            close = df['Adj Close']
            ma = close.rolling(window=lookback).mean()
            mean_reversion[ticker] = (close - ma) / ma
        
        return pd.DataFrame(mean_reversion)
    
    def calculate_volatility(self, lookback=20):
        """
        Calculate volatility factor.
        
        Args:
            lookback (int): Rolling window for volatility calculation
        
        Returns:
            pd.DataFrame: Annualized volatility for each asset
        """
        volatility = {}
        for ticker, df in self.price_data.items():
            returns = df['Adj Close'].pct_change()
            vol = returns.rolling(window=lookback).std() * np.sqrt(252)
            volatility[ticker] = vol
        
        return pd.DataFrame(volatility)
    
    def calculate_rsi(self, lookback=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            lookback (int): Period for RSI calculation
        
        Returns:
            pd.DataFrame: RSI values for each asset
        """
        rsi_values = {}
        for ticker, df in self.price_data.items():
            close = df['Adj Close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=lookback).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=lookback).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values[ticker] = rsi
        
        return pd.DataFrame(rsi_values)
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line period
        
        Returns:
            tuple: (MACD DataFrame, Signal DataFrame, Histogram DataFrame)
        """
        macd_values = {}
        signal_values = {}
        histogram_values = {}
        
        for ticker, df in self.price_data.items():
            close = df['Adj Close']
            ema_fast = close.ewm(span=fast).mean()
            ema_slow = close.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            macd_values[ticker] = macd
            signal_values[ticker] = signal_line
            histogram_values[ticker] = histogram
        
        return (pd.DataFrame(macd_values), 
                pd.DataFrame(signal_values), 
                pd.DataFrame(histogram_values))
    
    def calculate_beta(self, market_data, lookback=252):
        """
        Calculate beta relative to market index.
        
        Args:
            market_data (pd.Series): Market index prices
            lookback (int): Period for beta calculation
        
        Returns:
            pd.DataFrame: Beta values for each asset
        """
        beta_values = {}
        market_returns = market_data.pct_change()
        
        for ticker, df in self.price_data.items():
            asset_returns = df['Adj Close'].pct_change()
            
            # Calculate covariance and variance
            cov = asset_returns.cov(market_returns)
            market_var = market_returns.var()
            beta = cov / market_var
            beta_values[ticker] = beta
        
        return pd.DataFrame({'Beta': beta_values})


def normalize_factors(factor_df):
    """
    Normalize factors to zero mean and unit variance.
    
    Args:
        factor_df (pd.DataFrame): Factor values
    
    Returns:
        pd.DataFrame: Normalized factors
    """
    return (factor_df - factor_df.mean()) / factor_df.std()


def combine_factors(factors_dict, weights=None):
    """
    Combine multiple factors into a composite score.
    
    Args:
        factors_dict (dict): Dictionary with factor names as keys and DataFrames as values
        weights (dict): Weights for each factor (default: equal weight)
    
    Returns:
        pd.DataFrame: Combined factor scores
    """
    if weights is None:
        weights = {name: 1/len(factors_dict) for name in factors_dict}
    
    composite = None
    for name, factor_df in factors_dict.items():
        if composite is None:
            composite = factor_df * weights[name]
        else:
            composite += factor_df * weights[name]
    
    return composite


def momentum_factor(prices, lookback=252, skip=21):
    """
    12-1 style momentum: return over past `lookback` days excluding the most recent `skip` days.

    Args:
        prices (pd.DataFrame): Price DataFrame (columns = tickers)
        lookback (int): Number of trading days to look back (default ~252)
        skip (int): Number of days to exclude (default ~21)

    Returns:
        pd.DataFrame: Momentum scores aligned to `prices` index
    """
    # Use percentage change over lookback, then shift forward by skip to exclude most recent month
    mom = prices.pct_change(periods=lookback).shift(skip)
    return mom


def value_factor(pe_ratios):
    """
    Simple value factor: lower P/E is better. We invert P/E so higher is better.

    Args:
        pe_ratios (pd.DataFrame or pd.Series): P/E ratios indexed by ticker or by date x ticker

    Returns:
        pd.DataFrame or pd.Series: Value scores (higher = more attractive)
    """
    return -pe_ratios
