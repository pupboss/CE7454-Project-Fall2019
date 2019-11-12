import torch
import pickle
import torch.nn as nn
from crew import Crew
from hyperparams import Hyperparams as hps

with open("/Users/admin/Downloads/crew.pkl",'rb') as f:
    Attributes = pickle.load(f)

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

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
checkpoint = torch.load("/Users/admin/github/CE7454-Project-Fall2019/model/VanillaMLP.pth")
model.load_state_dict(checkpoint)

boxoffice_model = BoxOfficeModel(len_genre=23)
checkpoint = torch.load("/Users/admin/github/CE7454-Project-Fall2019/model/ckpt_boxoffice7.pth")
boxoffice_model.load_state_dict(checkpoint['Model_state_dict'])

def pred(genre, budget, duration, year):
    mean = torch.load('/Users/admin/github/CE7454-Project-Fall2019/data/mean.pt')
    std = torch.load('/Users/admin/github/CE7454-Project-Fall2019/data/std.pt')

    budget = torch.Tensor([budget])
    duration = torch.Tensor([duration])
    year = torch.Tensor([year])
    budget = (budget - mean[0]) / std[0]
    year = (year - mean[1]) / std[1]
    duration = (duration - mean[2]) / std[2]

    genre_len = Attributes.cal_len('genre')
    genre_onehot = torch.FloatTensor(genre_len).zero_()
    for i in genre:
        genre_onehot[i] += 1

    x = torch.cat([budget, year, duration, genre_onehot])

    pre_rating = float(model(x).detach().numpy())

    x = torch.cat([genre_onehot, year, duration, budget]).unsqueeze(0)
    pre_boxoffice = boxoffice_model(x).squeeze(0).detach().numpy()

    return pre_rating, pre_boxoffice

genre_string = "20,10,9"
genre = [int(p) for p in genre_string.split(',')]
budget = 20000
duration = 100
year = 2005
print(pred(genre, budget, duration, year))

