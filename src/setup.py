import linear_regression,rnn
import json

db = json.load(open("../db.json"))
stocks = db.keys()

for stock in stocks:
    print(f"creating models for {stock}")
    linear_regression.main(stock)
    rnn.main(stock)