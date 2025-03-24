from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/load_tables")
def load_tables():
    return "<p>Hello, World!</p>"