from flask import Flask, render_template, request
from src.data import get_current_price
import json

app = Flask(__name__)

@app.route("/")
def default():
    db = json.load(open("db.json"))
    stocks = [element["name"] for element in db["stocks"]]
    stocks.sort()
    return render_template("stocks.html",
                           stocks=stocks)

@app.route("/chart/<string:stock>")
def chart_page(stock):
    return render_template("chart.html",
                           stock=stock)

@app.route("/value/<string:stock>")
def value(stock):
    return str(get_current_price(stock))