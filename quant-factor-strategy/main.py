"""
Focused runner: Factor-Based Equity Strategy (Momentum + Value)

This `main.py` provides a compact, resume-ready pipeline:
- download adjusted close prices for a small universe
- compute 12-1 momentum (exclude last month) and a simple value factor (P/E)
- combine factors, rank each month, select top-N, run a simple monthly backtest

Usage:
    python main.py
    python main.py --tickers AAPL MSFT GOOGL --start 2015-01-01 --end 2023-12-31 --top-n 5
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import download_price_data
from src.factors import momentum_factor, value_factor
from src.backtest import simple_backtest


def main(tickers, start, end, top_n=5, freq='M'):
    # 1) Download prices (Adj Close)
    prices = download_price_data(tickers, start=start, end=end)

    # 2) Compute factors
    mom = momentum_factor(prices, lookback=252, skip=21)

    # Value factor: placeholder - try to read `data/pe_ratios.csv` if present
    try:
        pe = pd.read_csv('data/pe_ratios.csv', index_col=0, parse_dates=True)
        # Align to prices' columns; if pe provided per date x ticker use latest available
        if list(pe.columns) == list(prices.columns):
            val = value_factor(pe)
        else:
            # If provided as a single snapshot (series), broadcast
            val = value_factor(pe.iloc[-1])
            # convert to DataFrame aligned to index
            val = pd.DataFrame([val.values], index=[prices.index[0]], columns=pe.columns)
    except Exception:
        # If no P/E available, use inverse of 1-year trailing earnings proxy: use low volatility as proxy
        val = (-prices.pct_change(252)).fillna(0)

    # 3) Combine factors: z-score each and sum (equal weight)
    mom_z = (mom - mom.mean()) / mom.std()
    val_z = (val - val.mean()) / val.std()

    # Align indices and columns
    # Use mom_z as base index
    combined = mom_z.copy()
    for col in combined.columns:
        if col in val_z.columns:
            combined[col] = mom_z[col] + val_z[col].reindex(combined.index).fillna(0)

    # 4) Backtest - top-N monthly
    returns = simple_backtest(prices, combined, n_stocks=top_n, freq=freq)

    # 5) Plot cumulative returns
    (1 + returns).cumprod().plot(label='Strategy')
    # Benchmark: equal-weighted universe
    bench = prices.pct_change().resample(freq).last().mean(axis=1).loc[returns.index]
    (1 + bench).cumprod().plot(label='Universe Avg')
    plt.title('Momentum+Value Top-N Strategy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--top-n', type=int, default=5)
    args = parser.parse_args()
    main(args.tickers, args.start, args.end, top_n=args.top_n)
