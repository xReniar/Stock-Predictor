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
    stocks = sorted([element["name"] for element in db["stocks"]])
    return jsonify(stocks)

@app.route("/chart/<string:stock>")
def chart_page(stock):
    return render_template("chart.html",
                           stock=stock)

@app.route("/value/<string:stock>")
def current_value(stock):
    return str(get_current_price(stock))

@app.route("/price_state/<string:stock>")
def price_state(stock):
    return str(get_stock_price_state(stock))

@app.route("/chart_values/<string:stock>")
def chart_values(stock):
    return get_chart_values(stock)