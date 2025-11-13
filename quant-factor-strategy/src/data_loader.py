"""
Data loader module for downloading and cleaning financial data.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_stock_data(tickers, start_date, end_date, interval='1d'):
    """
    Download stock price data using yfinance.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval ('1d', '1wk', '1mo', etc.)
    
    Returns:
        dict: Dictionary with ticker as key and DataFrame as value
    """
    data = {}
    for ticker in tickers:
        try:
            logger.info(f"Downloading data for {ticker}...")
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            data[ticker] = df
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
    
    return data


def download_price_data(tickers, start="2015-01-01", end=None):
    """
    Download adjusted close prices for multiple tickers and return a DataFrame.

    Args:
        tickers (list): List of ticker symbols or space-separated string
        start (str): Start date in YYYY-MM-DD
        end (str): End date in YYYY-MM-DD or None (uses today)

    Returns:
        pd.DataFrame: DataFrame of adjusted close prices with tickers as columns
    """
    if isinstance(tickers, (list, tuple)):
        tickers_arg = " ".join(tickers)
    else:
        tickers_arg = tickers

    df = yf.download(tickers_arg, start=start, end=end, progress=False)
    # If single ticker, yfinance returns a single-column Series inside DataFrame structure
    if 'Adj Close' in df:
        prices = df['Adj Close']
    else:
        prices = df

    # Drop columns that are all NaN (delisted / missing)
    prices = prices.dropna(how='all', axis=1)
    return prices


def get_fundamental_data(ticker):
    """
    Retrieve fundamental data (earnings, book value, etc.) for a ticker.
    
    Args:
        ticker (str): Ticker symbol
    
    Returns:
        dict: Dictionary containing fundamental metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        fundamentals = {
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'pb_ratio': info.get('priceToBook'),
            'dividend_yield': info.get('dividendYield'),
            'earnings_growth': info.get('earningsGrowth'),
        }
        return fundamentals
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        return {}


def clean_data(df):
    """
    Clean and preprocess price data.
    
    Args:
        df (pd.DataFrame): Raw price DataFrame
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Remove rows with missing values
    df = df.dropna()
    
    # Sort by date
    df = df.sort_index()
    
    # Ensure proper data types
    df = df.astype('float64')
    
    return df


def save_data(data_dict, output_dir):
    """
    Save data dictionary to CSV files.
    
    Args:
        data_dict (dict): Dictionary with ticker as key and DataFrame as value
        output_dir (str): Output directory path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for ticker, df in data_dict.items():
        filepath = os.path.join(output_dir, f"{ticker}.csv")
        df.to_csv(filepath)
        logger.info(f"Saved {ticker} data to {filepath}")


def load_data(filepath):
    """
    Load data from CSV file.
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df
