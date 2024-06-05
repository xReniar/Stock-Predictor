from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def default():
    return render_template("stocks.html")

@app.route("/stock/<stock>")
def single_stock(stock):
    return render_template("stock.html")