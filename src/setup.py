import linear_regression,lstm
import json

db = json.load(open("../db.json"))
stocks = db["stocks"]
for stock in stocks:
    db[stock] = dict(
        pytorch = dict(
            value = 0,
            mse = 0
        ),
        sklearn = dict(
            value = 0,
            mse = 0
        )
    )

with open("../db.json","w") as f:
    json.dump(db,f,indent=4)

for stock in stocks:
    print(f"creating models for {stock}")
    linear_regression.main(stock)
    lstm.main(stock)