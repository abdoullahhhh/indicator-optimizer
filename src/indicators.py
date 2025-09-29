import numpy as np
import pandas as pd
import talib as ta
import yfinance as yf

def rsi_threshold_strategy(df, length=14, pmom=65, nmom=32, ema_len=5):
    """
    Inputs:
      - df: DataFrame with a 'close' column
    Returns:
      - pd.Series of regime values (+1/-1 carried forward)
    """
    close = pd.Series(df['close'].astype(float), index=df.index)
    rsi = pd.Series(ta.RSI(close.values, timeperiod=length), index=df.index)
    ema_close = pd.Series(ta.EMA(close.values, timeperiod=ema_len), index=df.index)
    ema_chg = ema_close.diff()

    p_mom = (rsi.shift(1) < pmom) & (rsi > pmom) & (rsi > nmom) & (ema_chg > 0)
    n_mom = (rsi < nmom) & (ema_chg < 0)

    switch = pd.Series(np.nan, index=df.index)
    switch[p_mom.fillna(False)] = 1.0
    switch[n_mom.fillna(False)] = -1.0

    regime = switch.ffill()
    return regime

def fetch_btc_yf(start='2015-01-01', end=None):
    df = yf.download('BTC-USD', start=start, end=end, interval='1d', auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    return df[['open','high','low','close','volume']].dropna()