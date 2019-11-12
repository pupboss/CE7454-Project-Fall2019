#!/usr/bin/python

from flask import Flask
from flask_cors import CORS
from flask import request
from flask import jsonify

import torch

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):

        x = self.hidden(x)
        x = self.predict(x)
        return x

model = Net(n_feature=26, n_hidden=10, n_output=1)
checkpoint = torch.load("../model/VanillaMLP.pth")
model.load_state_dict(checkpoint)

def rating_pred(genre, budget, duration, year):
    mean = torch.load('../data/mean.pt')
    std = torch.load('../data/std.pt')

    budget = torch.Tensor([budget])
    duration = torch.Tensor([duration])
    year = torch.Tensor([year])
    budget = (budget - mean[0]) / std[0]
    year = (year - mean[1]) / std[1]
    duration = (duration - mean[2]) / std[2]

    genre_len = 23
    genre_onehot = torch.FloatTensor(genre_len).zero_()
    for i in genre:
        genre_onehot[i] += 1

    x = torch.cat([budget, year, duration, genre_onehot])

    pre_rating = float(model(x).detach().numpy())

    return pre_rating

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, CE7454'

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.form
    budget = int(data['budget'])
    year = int(data['year'])
    duration = int(data['duration'])
    genres = data['genres']
    genre = [int(p) for p in genres.split(',')]

    r = rating_pred(genre, budget, duration, year)
    result = {'rating': r}

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
