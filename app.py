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
    return jsonify(sorted(db["stocks"]))

@app.route("/chart/<string:stock>")
def chart_page(stock):
    return render_template("chart.html",
                           response = dict(
                               stock_name = stock,
                               models = sorted(get_models())
                           ))

# single value requests
@app.route("/models")
def models():
    return sorted(get_models())

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
    db = json.load(open(f"result/{model}.json"))
    stock_obj = db[stock]
    return str(stock_obj["prediction"])

@app.route("/mse/<string:model>/<string:stock>")
def stock_error(model,stock):
    db = json.load(open(f"result/{model}.json"))
    stock_obj = db[stock]
    return str(stock_obj["mse"])
