import numpy as np
import yfinance as yf


def get_current_price(symbol:str):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'].iloc[0]