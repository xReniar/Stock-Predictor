import linear_regression,lstm
import json

db = json.load(open("../db.json"))
stocks = db.keys()

for stock in stocks:
    print(f"creating models for {stock}")
    linear_regression.main(stock)
    lstm.main(stock)