import numpy as np
import yfinance as yf
from datetime import datetime,timedelta


def get_current_price(symbol:str):
    todays_data = yf.Ticker(symbol).history(period='1d')
    return todays_data['Close'].iloc[0]

def get_stock_price_state(symbol:str):
    data = yf.Ticker(symbol).history("5d")
    return data["Close"].iloc[-1] - data["Close"].iloc[-2]

def get_chart_values(symbol:str):
    current_date = datetime.now()
    data = yf.download(symbol,
                   start=current_date - timedelta(days=60),
                   end=current_date,
                   interval="1d")
    data = data.reset_index()
    return dict(
        xValues = [str(x.date()) for x in data["Date"].tolist()],
        yValues = data["Close"].tolist()
    )