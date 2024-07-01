from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import json


def get_current_price(symbol:str):
    todays_data = yf.Ticker(symbol).history(period='1d')
    return todays_data['Close'].iloc[0]

def get_stock_price_state(symbol:str):
    data = yf.Ticker(symbol).history("5d")
    return data["Close"].iloc[-1] - data["Close"].iloc[-2]

def get_chart_values(symbol:str):
    current_date = datetime.now()
    data: pd.DataFrame = yf.download(symbol,
                   start=current_date - timedelta(days=480),
                   end=current_date,
                   interval="1d",
                   progress=False)
    data = data.reset_index()
    return dict(
        xValues = [str(x.date()) for x in data["Date"].tolist()],
        yValues = data["Close"].tolist()
    )

def download_stock_data(ticker:str) -> pd.Series:
    current_date = datetime.now()
    stock_data = yf.download(ticker,
                             start=current_date - timedelta(days=3652),
                             end=current_date,
                             progress=False)
    return stock_data['Close']

def update_db_stock(stock,model,value,mse):
    db = json.load(open("../db.json"))
    db[stock][model]["value"] = value
    db[stock][model]["mse"] = mse
    with open("../db.json","w") as f:
        json.dump(db,f,indent=4)