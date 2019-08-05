from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    # return "Hello"

@app.route('/',methods=["POST"])
def add():
    return render_template("login.html")


app.run(host='127.0.0.1', port = 8000, debug=True)
