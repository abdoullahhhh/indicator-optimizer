# Technical Indicator Optimizer (Optuna)

A small, modular project that **optimizes indicator parameters** for different goals (profit-seeking, conservative, or balanced).  
For the demo, we use a simple indicator — **`rsi_threshold_strategy`** — and BTC daily prices from Yahoo Finance, but **both are swappable**.

---

## Why this project?
Most strategies rely on fixed parameters. Markets change.  
This repo shows how to **tune parameters automatically** and **evaluate trade-offs by risk profile**, with a lightweight backtest to compare **before vs after**.

---

## What it does
- **Optimize any indicator’s parameters** with [Optuna](https://optuna.org/)
- Choose a **risk profile**:
  1) **Aggressive**: maximize profits (accepts larger drawdowns)  
  2) **Conservative**: prioritize risk and risk-adjusted metrics  
  3) **Balanced**: trade off between returns and risk
- **Backtest** before/after to show improvement
- **Plug-and-play** indicator and data source (the demo uses `rsi_threshold_strategy` + BTC from Yahoo; replace with your own)

> Tip: **Give the optimizer realistic parameter ranges**. Tight, sensible bounds help it search effectively and avoid wasting trials on nonsense settings.

---

## How it works (high level)
**Indicator → Signals → Backtest → Optimization**
1. **Indicator** (`rsi_threshold_strategy`) produces a long/flat regime from price.
2. **Signals** are converted into entries/exits.
3. **Backtest** runs a simple event-driven long-only simulation.
4. **Optimization** uses Optuna to search parameter space, scoring by your selected risk profile.

---

## Quickstart

### Colab (one cell)
```python
!apt-get -qq install -y libta-lib0 libta-lib-dev
!pip install yfinance vectorbt optuna plotly TA-Lib
```

```python
from core import SingleIndicatorStrategy
from indicators import fetch_btc_yf  # or your own data loader

df = fetch_btc_yf('2018-01-01')      # swap with your dataset if you like
strategy = SingleIndicatorStrategy()  # default demo params

# Before optimization
signals = strategy.get_signals(df)
pf_before = strategy.backtest(df, signals, plot=False)
print("Before:", pf_before.stats())

# Optimize (choose preset: 'aggressive' | 'balanced' | 'conservative')
best = strategy.optimize_params(df, n_trials=20, preset='balanced')
print("Best params:", best)

# After optimization
signals_opt = strategy.get_signals(df)
pf_after = strategy.backtest(df, signals_opt, plot=True)
print("After:", pf_after.stats())
```

### Local (Jupyter)
```bash
pip install -r requirements.txt
jupyter notebook notebooks/demo.ipynb
```

---

## Repository structure (compact)
```

src/
 ├─ core.py           # Strategy, backtest, optimizer
 ├─ indicators.py     # Example indicator + data loader
 └─ __init__.py
notebooks/
 └─ demo.ipynb        # Walkthrough
tests/
 └─ test_smoke.py     # Sanity check
requirements.txt
.gitignore
LICENSE
README.md

```

---

## Risk profiles (objective ideas)
- **Aggressive**: emphasize total/annualized return and profit factor; lighter penalty on drawdown/volatility.  
- **Balanced**: mix of return + Sharpe/Sortino + moderate drawdown penalty.  
- **Conservative**: weight Sharpe/Sortino and max drawdown heavily; add volatility penalty.

> Under the hood, the score combines these metrics and can include penalties (e.g., **trade-cluster penalty** or **long-hold penalty**) to discourage clustered entries or overly long continuous positions.

---

## Customize it
- Replace `rsi_threshold_strategy` with your favorite indicator(s).
- Swap `fetch_btc_yf` with your own data loader (CSV, API, etc.).
- Adjust **parameter ranges** in the Optuna search space to fit your indicator and timeframe.
- Change **risk scoring weights** to match your preferences.

---

## Roadmap / Ideas
- **Train/validation/test windows** for more robust evaluation
- **Multiple indicators** and automated **combination search** (e.g., best pair/triple)
- Add **saved studies** (SQLite) and export `best_params.json`

---

## Notes
- Demo uses: `yfinance`, `vectorbt`, `optuna`, `plotly`, and `TA-Lib` (installed system libs first on Colab).
- Keep the repo lean: `.gitignore` excludes caches, checkpoints, and study databases.

---

## License
MIT — see `LICENSE`.
