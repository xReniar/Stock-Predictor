# Stock Predictor
Machine learning course project that tries to predict if the price of the selected stocks are going to rise or fall. We used 3 models for predicting the price and they are `Linear Regression`, `LSTM` and `CNN`.

## Installing the requirements
To install all the libraries:
```bash
pip3 install -r requirements.txt
```
## How it works
The `db.json` stores all the information about the stocks used, the predicted values and the mean squared error of the training. If you wanna add other stocks check [this](https://github.com/ahnazary/Finance/blob/master/finance/src/database/valid_tickers.csv). When ready to add just add it like this:
```json
{
    "BTC-USD": {
        "pytorch": {
            "value": 60523.98296764683,
            "mse": 0.0024650844279676676
        },
        "sklearn": {
            "value": 62813.428385479114,
            "mse": 1094015.7530843548
        }
    },
    ...,
    "selected stock": {          // selected stock
        "pytorch": {
            "value": 0,
            "mse": 0
        },
        "sklearn": {
            "value": 0,
            "mse": 0
        }
    },
```
After adding all the stocks needed run the `setup.py` to create all the models, and update the `db.json`:
```bash
cd src
python3 setup.py
```
## Run server
```bash
export FLASK_APP=app # one time only
flask run
```