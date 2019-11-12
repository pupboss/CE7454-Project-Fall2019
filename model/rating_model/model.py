import torch
import torch.nn as nn
from hyperparams import Hyperparams as hps
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as imagemodels

class Flatten(nn.Module):
	def forward(self, x):
		x = x.view(x.size()[0], -1)
		return x

class RatingModel(nn.Module):
	def __init__(self, len_actor, len_writer, len_director, len_genre):
		super(RatingModel, self).__init__()
		# LSTM
		self.lstm = nn.LSTM(50, 50, num_layers=1, batch_first=True)
		self.LSTM_MLP = nn.Linear(hps.seq_max_len*50, 256)

		# Res Net 18
		seed_model = imagemodels.__dict__['resnet18'](pretrained=True)
		seed_model = nn.Sequential(*list(seed_model.children())[:-1])
		for param in seed_model.parameters():
			param.requires_grad = False
		last_layer_index = len(list(seed_model.children()))
		seed_model.add_module(str(last_layer_index), Flatten())
		seed_model.add_module(str(last_layer_index+1), nn.Linear(512,128))
		self.CNN = seed_model

		# Attributes
		embed_size = 16
		self.actor_embed = nn.Embedding(len_actor, embed_size)
		self.writer_embed = nn.Embedding(len_writer, embed_size)
		self.director_embed = nn.Embedding(len_director, embed_size)
		self.MLP_attri = nn.Sequential(
			nn.Linear(len_genre + 3 + embed_size * 5, 128),
			nn.Dropout(p=0.2),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.Dropout(p=0.2),
			nn.ReLU(),
			nn.Linear(128, 128)
		)

		self.predictor = nn.Sequential(
			nn.Linear(128+256+128, 512),
			nn.Dropout(p=0.2),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.Dropout(p=0.2),
			nn.ReLU(),
			nn.Linear(512, 1)
		)


	def forward(self, attributes, actor, writer, director, genre, seq, seq_len, img):

		# LSTM Net
		seq_packed = pack_padded_sequence(seq, seq_len, batch_first=True, enforce_sorted=False)
		seq_packed_output, hidden = self.lstm(seq_packed)
		seq_unpacked = pad_packed_sequence(seq_packed_output, batch_first=True, padding_value=0, total_length=hps.seq_max_len)[0]
		seq_unpacked = seq_unpacked.flatten(1)
		seq_unpacked = self.LSTM_MLP(seq_unpacked)

		img = self.CNN(img)

		actor = self.actor_embed(actor).flatten(1)
		writer = self.writer_embed(writer).flatten(1)
		director = self.director_embed(director).flatten(1)

		attributes = torch.cat([actor, writer, director, genre, attributes], dim=1)
		attributes = self.MLP_attri(attributes)

		pre_rating = torch.cat([attributes, seq_unpacked, img], dim=1)
		pre_rating = self.predictor(pre_rating)

		return pre_rating


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


	def forward(self, genre, attributes):

		x = torch.cat([genre, attributes], dim=1)
		x = self.MLP(x)

		x = self.softmax(x)

		return x
