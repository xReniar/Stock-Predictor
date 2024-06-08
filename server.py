from flask import Flask, render_template, request
from src.data import get_current_price
import json

app = Flask(__name__)

@app.route("/")
def default():
    db = json.load(open("db.json"))
    stocks = [element["name"] for element in db["stocks"]]
    return render_template("stocks.html",
                           stocks=stocks)

@app.route("/value/<stock>")
def value(stock):
    return str(get_current_price(stock))

@app.route("/stock/<stock>")
def single_stock(stock):
    return render_template("stock.html")

@app.route("/train/<model>/<stock>")
def train(model,stock):
    if model == "scikit-learn":
        pass
    if model == "pytorch":
        pass

@app.route("/predict/<model>/<stock>", methods=["POST"])
def predict(model,stock):
    if model == "scikit-learn":
        pass
    if model == "pytorch":
        pass
    data = request.to_json(force = True)
    return str(data)