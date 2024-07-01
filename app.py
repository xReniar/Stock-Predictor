from flask import Flask, render_template,jsonify
from src.data import *
import json

app = Flask(__name__)

@app.route("/")
def default():
    return render_template("stocks.html")

@app.route("/get_stocks")
def get_stocks():
    db = json.load(open("db.json"))
    #stocks = sorted([element["name"] for element in db["stocks"]])
    stocks = sorted(db.keys())
    return jsonify(stocks)

@app.route("/chart/<string:stock>")
def chart_page(stock):
    return render_template("chart.html",
                           stock=stock)

# single value requests
@app.route("/value/<string:stock>")
def current_value(stock):
    return str(get_current_price(stock))

@app.route("/price_state/<string:stock>")
def price_state(stock):
    return str(get_stock_price_state(stock))

@app.route("/chart_values/<string:stock>")
def chart_values(stock):
    return get_chart_values(stock)

@app.route("/prediction/<string:model>/<string:stock>")
def stock_prediction(model,stock):
    db = json.load(open("db.json"))
    stock_obj = db[stock]
    return str(stock_obj[f"{model}"]["value"])

@app.route("/mse/<string:model>/<string:stock>")
def stock_error(model,stock):
    db = json.load(open("db.json"))
    stock_obj = db[stock]
    return str(stock_obj[f"{model}"]["mse"])
