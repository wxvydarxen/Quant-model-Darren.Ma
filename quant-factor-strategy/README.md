# Factor-Based Equity Strategy: Momentum + Value on a Small Universe

This repository implements a focused, resume-ready v1 of a factor-based equity strategy.

Project highlights:
- Downloads adjusted close prices using `yfinance`
- Implements a 12-1 momentum factor (exclude the most recent month) and a simple value factor (P/E)
- Ranks stocks monthly and selects the top N for an equal-weighted long-only portfolio
- Runs a monthly backtest and visualizes cumulative returns vs a simple benchmark

This structure is intentionally compact and polished for a portfolio project you can show on LinkedIn.

## Quick Start

1. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the strategy (defaults use a small sample universe):

```bash
python main.py
```

4. (Optional) Provide a `data/pe_ratios.csv` file if you want to use real P/E snapshots for the value factor.

## Files

- `data/` — optional cached CSVs (e.g., `pe_ratios.csv`)
- `notebooks/01_exploratory.ipynb` — EDA and visual exploration
- `src/data_loader.py` — price downloader helpers (`download_price_data`)
- `src/factors.py` — `momentum_factor` and `value_factor` helpers
- `src/backtest.py` — `simple_backtest` and `rank_and_select` utilities
- `src/utils.py` — performance helpers (Sharpe, drawdown, etc.)
- `main.py` — compact runner for the Momentum+Value top-N strategy

## What to highlight on your resume / LinkedIn

Title: Factor-Based Equity Strategy (Momentum + Value)

Short description:
Built a Python-based quantitative equity strategy that ranks stocks using a 12-1 momentum factor and a simple value factor, constructs a monthly top-N portfolio, and backtests performance vs a simple benchmark. Used `pandas`, `NumPy`, `yfinance`, and `matplotlib`.

Skills/tech: Python, pandas, NumPy, yfinance, matplotlib, quantitative finance, backtesting

## Next steps / Roadmap

- Add full S&P 500 universe ingestion
- Add transaction costs, slippage, and turnover constraints
- Add risk-managed weighting (risk parity / volatility scaling)
- Add performance attribution and walk-forward validation

---

Happy building! If you want, I can now:

- run a quick smoke-check of the pipeline (no external network calls are made from me, but I can run local tests), or
- push a minimal CI/test to validate imports and linting.

Which would you like next?
