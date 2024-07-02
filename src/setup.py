from data import get_models
import linear_regression,lstm
import json

db = json.load(open("../db.json"))
stocks = db["stocks"]
for model in get_models():
    model_dict = {}
    for stock in stocks:
        model_dict[stock] = dict(
            predictions = 0,
            mse = 0
        )
    with open(f"../result/{model}.json","w") as f:
        json.dump(model_dict,f,indent=4)

for stock in stocks:
    print(f"creating models for {stock}")
    linear_regression.main(stock)
    lstm.main(stock)