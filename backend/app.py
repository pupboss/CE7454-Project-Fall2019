#!/usr/bin/python

from flask import Flask
from flask_cors import CORS
from flask import request
from flask import jsonify

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):

        x = self.hidden(x)
        x = self.predict(x)
        return x

class BoxOfficeModel(nn.Module):
    def __init__(self, len_genre):
        super(BoxOfficeModel, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(len_genre + 3, 64),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(64,2)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.MLP(x)
        x = self.softmax(x)
        return x

model = Net(n_feature=26, n_hidden=10, n_output=1)
checkpoint = torch.load("../model/VanillaMLP.pth")
model.load_state_dict(checkpoint)

boxoffice_model = BoxOfficeModel(len_genre=23)
checkpoint = torch.load("../model/ckpt_boxoffice7.pth")
boxoffice_model.load_state_dict(checkpoint['Model_state_dict'])

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

    x = torch.cat([genre_onehot, year, duration, budget]).unsqueeze(0)
    pre_boxoffice = boxoffice_model(x).squeeze(0).detach().numpy()

    return pre_rating, float(pre_boxoffice[0]), float(pre_boxoffice[1])

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

    rt, loss, profit = rating_pred(genre, budget, duration, year)
    result = {'rating': rt, 'boxOfficeData': [{'status': 'Loss', 'probability': loss}, {'status': 'Profit', 'probability': profit}]}

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7454)
