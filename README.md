# Stock Predictor
Machine learning course project that tries to predict if the price of the selected stocks are going to rise or fall. We used 3 models for predicting the price and they are `Linear Regression`, `LSTM` and `RNN`.
## Installing the requirements
[![techused](https://skillicons.dev/icons?i=js,html,css,bootstrap,py,flask,pytorch,sklearn)](https://skillicons.dev)
<br>
To install all the libraries:
```bash
pip3 install -r requirements.txt
```
## How it works
The `db.json` stores all the information about the stocks used, the predicted values and the mean squared error of the training. If you wanna add other stocks check [this](https://github.com/ahnazary/Finance/blob/master/finance/src/database/valid_tickers.csv). When ready to add just add it in the `stocks`:
```json
{
    "stocks": ["add","here","the","selected","stocks"]
}
```
After adding all the stocks needed, run the `setup.py` to create all the models and update the files inside `result` folder :
```bash
python3 src/setup.py
```
## Run server
```bash
export FLASK_APP=app # one time only
flask run
```