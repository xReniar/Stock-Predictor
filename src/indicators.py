import pandas as pd

def sma(data: pd.DataFrame, period: int):
    data[f"SMA-{period}"] = data["Close"].rolling(window=period).mean()

def std(data: pd.DataFrame, period:int):
    data[f"STD-{period}"] = data["Close"].rolling(window=period).std()

def bb(data: pd.DataFrame, period: int):
    data[f'Lower_BB-{period}'] = data[f"SMA-{period}"] - 2 * data[f"STD-{period}"]
    data[f'Upper_BB-{period}'] = data[f"SMA-{period}"] + 2 * data[f"STD-{period}"]

def ema(data: pd.DataFrame, period: int):
    data[f"EMA-{period}"] = data["Close"].ewm(span=period,adjust=False).mean()

def rsi(data: pd.DataFrame, period: int):
    delta = data["Close"].diff()
    delta = delta.dropna()

    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0], loss[loss > 0] = 0, 0

    avg_gain = gain.ewm(com=14, min_periods=period).mean()
    avg_loss = abs(loss.ewm(com = 14, min_periods=period).mean())

    data[f"RSI-{period}"] = 100 - (100/(1 + (avg_gain/avg_loss)))