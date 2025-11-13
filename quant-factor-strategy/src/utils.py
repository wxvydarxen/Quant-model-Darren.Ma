"""
Utility functions for the quantitative strategy.
"""

import pandas as pd
import numpy as np


def calculate_returns(prices):
    """
    Calculate simple returns from prices.
    
    Args:
        prices (pd.Series): Price series
    
    Returns:
        pd.Series: Returns
    """
    return prices.pct_change()


def calculate_log_returns(prices):
    """
    Calculate log returns from prices.
    
    Args:
        prices (pd.Series): Price series
    
    Returns:
        pd.Series: Log returns
    """
    return np.log(prices / prices.shift(1))


def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns.
    
    Args:
        returns (pd.Series): Returns series
    
    Returns:
        pd.Series: Cumulative returns
    """
    return (1 + returns).cumprod() - 1


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio.
    
    Args:
        returns (pd.Series): Daily returns
        risk_free_rate (float): Annual risk-free rate (default 2%)
    
    Returns:
        float: Sharpe ratio
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    return excess_returns.mean() / returns.std() * np.sqrt(252)


def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sortino ratio (penalizes downside volatility).
    
    Args:
        returns (pd.Series): Daily returns
        risk_free_rate (float): Annual risk-free rate (default 2%)
    
    Returns:
        float: Sortino ratio
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    return excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0


def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown.
    
    Args:
        prices (pd.Series): Price series
    
    Returns:
        float: Maximum drawdown as percentage
    """
    cumulative_returns = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def calculate_calmar_ratio(returns, prices):
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Args:
        returns (pd.Series): Returns series
        prices (pd.Series): Price series
    
    Returns:
        float: Calmar ratio
    """
    annual_return = returns.mean() * 252
    max_dd = abs(calculate_max_drawdown(prices))
    return annual_return / max_dd if max_dd > 0 else 0


def correlation_matrix(price_data):
    """
    Calculate correlation matrix of assets.
    
    Args:
        price_data (dict): Dictionary of price DataFrames by ticker
    
    Returns:
        pd.DataFrame: Correlation matrix
    """
    returns_dict = {}
    for ticker, df in price_data.items():
        returns_dict[ticker] = calculate_returns(df['Adj Close'])
    
    returns_df = pd.DataFrame(returns_dict)
    return returns_df.corr()


def analyze_factor_performance(factor_scores, returns, quantile=5):
    """
    Analyze factor performance by quantiles.
    
    Args:
        factor_scores (pd.Series): Factor scores
        returns (pd.Series): Forward returns
        quantile (int): Number of quantiles (default 5)
    
    Returns:
        pd.DataFrame: Returns by quantile
    """
    factor_returns = pd.DataFrame({
        'factor': factor_scores,
        'returns': returns
    }).dropna()
    
    # Assign quantiles
    factor_returns['quantile'] = pd.qcut(
        factor_returns['factor'], 
        q=quantile, 
        labels=False, 
        duplicates='drop'
    )
    
    # Calculate returns by quantile
    returns_by_quantile = factor_returns.groupby('quantile')['returns'].agg([
        'mean', 'std', 'count'
    ])
    returns_by_quantile.columns = ['Mean Return', 'Std Dev', 'Count']
    
    return returns_by_quantile


def print_performance_summary(stats):
    """
    Print performance statistics summary.
    
    Args:
        stats (dict): Dictionary of performance metrics
    """
    print("\n" + "="*50)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("="*50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:.<30} {value:.4f}")
        else:
            print(f"{key:.<30} {value}")
    print("="*50 + "\n")


def resample_data(df, frequency='W'):
    """
    Resample data to different frequency.
    
    Args:
        df (pd.DataFrame): Original data
        frequency (str): Frequency ('D', 'W', 'M', 'Q', 'Y')
    
    Returns:
        pd.DataFrame: Resampled data
    """
    return df.resample(frequency).last()
