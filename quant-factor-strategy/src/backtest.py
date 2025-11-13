"""
Backtesting module for portfolio construction and strategy evaluation.
"""

import pandas as pd
import numpy as np


class Portfolio:
    """Portfolio management and backtesting."""
    
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        """
        Initialize portfolio.
        
        Args:
            initial_capital (float): Starting capital
            transaction_cost (float): Transaction cost as percentage (default 0.1%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.positions = {}
        self.cash = initial_capital
        self.portfolio_values = []
        self.dates = []
    
    def rebalance(self, signals, prices, top_n=10):
        """
        Rebalance portfolio based on factor signals.
        
        Args:
            signals (pd.DataFrame): Factor signals (scores)
            prices (pd.DataFrame): Current prices
            top_n (int): Number of assets to hold
        
        Returns:
            dict: New positions
        """
        # Get top N assets by signal
        top_assets = signals.nlargest(top_n, signals.columns[0]).index
        
        # Equal weight
        weight_per_asset = (self.cash * 0.95) / top_n  # Keep 5% cash buffer
        
        new_positions = {}
        for asset in top_assets:
            if asset in prices.index:
                price = prices.loc[asset]
                shares = weight_per_asset / price
                new_positions[asset] = shares
        
        self.positions = new_positions
        return new_positions
    
    def update_portfolio_value(self, prices, date):
        """
        Update portfolio value based on current prices.
        
        Args:
            prices (pd.Series): Current prices for all assets
            date: Current date
        """
        portfolio_value = self.cash
        
        for asset, shares in self.positions.items():
            if asset in prices.index:
                portfolio_value += shares * prices[asset]
        
        self.portfolio_values.append(portfolio_value)
        self.dates.append(date)
    
    def get_portfolio_value_series(self):
        """Return portfolio values as Series."""
        return pd.Series(self.portfolio_values, index=self.dates)


class Backtest:
    """Backtest a quantitative strategy."""
    
    def __init__(self, price_data, factor_signals, initial_capital=100000, 
                 rebalance_freq='M', top_n=10):
        """
        Initialize backtest.
        
        Args:
            price_data (dict): Dictionary of price DataFrames by ticker
            factor_signals (pd.DataFrame): Factor scores over time
            initial_capital (float): Starting capital
            rebalance_freq (str): Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            top_n (int): Number of assets to hold
        """
        self.price_data = price_data
        self.factor_signals = factor_signals
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self.portfolio = Portfolio(initial_capital)
        self.returns = None
    
    def run(self):
        """Run the backtest."""
        # Combine all price data into single DataFrame
        all_prices = pd.DataFrame({
            ticker: df['Adj Close'] 
            for ticker, df in self.price_data.items()
        })
        
        # Align dates
        common_dates = all_prices.index.intersection(self.factor_signals.index)
        all_prices = all_prices.loc[common_dates]
        factor_signals = self.factor_signals.loc[common_dates]
        
        # Rebalancing dates
        rebalance_dates = all_prices.resample(self.rebalance_freq).first().index
        
        for date in all_prices.index:
            # Rebalance if scheduled
            if date in rebalance_dates:
                current_signal = factor_signals.loc[date]
                self.portfolio.rebalance(current_signal, all_prices.loc[date], self.top_n)
            
            # Update portfolio value
            self.portfolio.update_portfolio_value(all_prices.loc[date], date)
        
        self.returns = self._calculate_returns()
        return self.portfolio.get_portfolio_value_series()
    
    def _calculate_returns(self):
        """Calculate portfolio returns."""
        portfolio_values = self.portfolio.get_portfolio_value_series()
        return portfolio_values.pct_change().dropna()
    
    def get_statistics(self):
        """
        Calculate performance statistics.
        
        Returns:
            dict: Performance metrics
        """
        portfolio_values = self.portfolio.get_portfolio_value_series()
        returns = self.returns
        
        total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Maximum drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Number of Trades': len(returns),
            'Final Portfolio Value': portfolio_values.iloc[-1]
        }


# --- Simple top-N monthly backtest utilities (alternative interface) ---
def rank_and_select(factor_df, n_stocks=20, freq="M"):
    """
    Rank each rebalance date and pick top N assets.

    Args:
        factor_df (pd.DataFrame): Factor scores with datetime index and tickers as columns
        n_stocks (int): Number to select
        freq (str): Resampling frequency (e.g. 'M')

    Returns:
        dict: mapping from rebalance date -> list of tickers
    """
    factor_monthly = factor_df.resample(freq).last().dropna(how="all")
    picks = {}
    for date, row in factor_monthly.iterrows():
        top = row.dropna().sort_values(ascending=False).head(n_stocks).index.tolist()
        picks[date] = top
    return picks


def simple_backtest(prices, signal, n_stocks=20, freq="M"):
    """
    Simple long-only backtest: at each rebalance date buy equal-weighted top-N,
    compute next-period returns and produce a returns series aligned to rebalance periods.

    Args:
        prices (pd.DataFrame): Price DataFrame (datetime index)
        signal (pd.DataFrame): Factor scores (datetime index)
        n_stocks (int): Number of stocks to hold
        freq (str): Rebalance frequency

    Returns:
        pd.Series: Periodic portfolio returns (index = rebal_dates[1:])
    """
    # Monthly (or freq) prices at close
    monthly_prices = prices.resample(freq).last()
    # Align signals to monthly dates
    signal = signal.reindex(monthly_prices.index).dropna(how="all")

    picks = rank_and_select(signal, n_stocks=n_stocks, freq=freq)

    # Compute returns between rebalance dates
    pct = prices.pct_change()
    pct_monthly = pct.reindex(monthly_prices.index)

    port_rets = []
    idx = []
    dates = monthly_prices.index
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        current_date = dates[i]
        chosen = picks.get(prev_date, [])
        if not chosen:
            port_rets.append(0.0)
            idx.append(current_date)
            continue
        # average return of chosen assets on current_date
        r = pct_monthly.loc[current_date, chosen].mean()
        port_rets.append(r)
        idx.append(current_date)

    port_series = pd.Series(port_rets, index=idx)
    return port_series
