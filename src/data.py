import numpy as np
import yfinance as yf
from datetime import datetime,timedelta
import time

def get_current_price(symbol:str):
    todays_data = yf.Ticker(symbol).history(period='1d')
    return todays_data['Close'].iloc[0]

def get_stock_price_state(symbol:str):
    data = yf.Ticker(symbol).history("5d")
    return data["Close"].iloc[-1] - data["Close"].iloc[-2]

def get_chart(symbol:str):
    pass