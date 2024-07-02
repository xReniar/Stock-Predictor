from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import json
import os


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

def download_stock_data(ticker:str,timeframe:str) -> pd.Series:
    current_date = datetime.now()
    delta = 3652 if timeframe == "1d" else 730
    stock_data = yf.download(ticker,
                             start=current_date - timedelta(days=delta),
                             end=current_date,
                             interval=timeframe,
                             progress=False)
    return stock_data

def get_models():
    folder_content = [x.split(".py")[0] for x in os.listdir("src")]
    for file_name in ["data","indicators","setup","__pycache__"]:
        folder_content.remove(file_name)
    return sorted(folder_content)

def update_result_stock(stock,model,value,mse):
    db = json.load(open(f"result/{model}.json"))
    db[stock]["predicted"] = value
    db[stock]["mse"] = mse
    with open(f"result/{model}.json","w") as f:
        json.dump(db,f,indent=4)