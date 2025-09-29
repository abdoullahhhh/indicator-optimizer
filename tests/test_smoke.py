# -*- coding: utf-8 -*-
"""test_smoke.ipynb
"""

import pandas as pd
import numpy as np
from core import SingleIndicatorStrategy
from indicators import fetch_btc_yf, rsi_threshold_strategy

def test_strategy_runs():
    df = fetch_btc_yf('2021-01-01', '2021-03-01')
    strategy = SingleIndicatorStrategy()
    signals = strategy.get_signals(df)
    pf = strategy.backtest(df, signals, plot=False)
    stats = pf.stats()
    assert "Total Return [%]" in stats, "Backtest did not produce stats"