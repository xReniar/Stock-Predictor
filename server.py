from flask import Flask, render_template

templates_dir = "./templates"

app = Flask(__name__)

@app.route("/")
def default():
    return render_template("default.html")