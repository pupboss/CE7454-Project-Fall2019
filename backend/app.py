#!/usr/bin/python

from flask import Flask
from flask_cors import CORS
from flask import request

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, CE7454'

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.form
    return '', 204

if __name__ == '__main__':
    app.run(host="0.0.0.0")
