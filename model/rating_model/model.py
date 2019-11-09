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
		# embed_size = 16
		# embed_size_genre = 50
		# self.actor_embed = nn.Embedding(len_actor, embed_size)
		# self.writer_embed = nn.Embedding(len_writer, embed_size)
		# self.director_embed = nn.Embedding(len_director, embed_size)
		# self.MLP = nn.Sequential(
		# 	nn.Linear(embed_size*5+embed_size_genre, 64),
		# 	nn.Dropout(p=0.5),
		# 	nn.Linear(64, 64),
		# 	nn.Dropout(p=0.5),
		# 	nn.Linear(64, 1))

		self.lstm = nn.LSTM(50, 50, num_layers=2, batch_first=True)
		self.LSTM_MLP = nn.Sequential(
			nn.Linear(hps.seq_max_len*50, 1024),
			nn.Dropout(p=0.2),
			nn.ReLU6(),
			nn.Linear(1024, 512),
			nn.ReLU6(),
			nn.Dropout(p=0.2),
			nn.Linear(512, 256)
		)

		# Res Net 18
		seed_model = imagemodels.__dict__['resnet18'](pretrained=False)
		seed_model = nn.Sequential(*list(seed_model.children())[:-1])
		last_layer_index = len(list(seed_model.children()))
		seed_model.add_module(str(last_layer_index), Flatten())
		seed_model.add_module(str(last_layer_index+1), nn.Linear(512,256))
		self.CNN = seed_model

		self.MLP = nn.Sequential(
			nn.Linear(512,512),
			nn.ReLU6(),
			nn.Linear(512,512),
			nn.ReLU6(),
			nn.Linear(512,1)
		)


	def forward(self, actor, writer, director, genre, seq, seq_len, ratio, img):
		# actor = self.actor_embed(actor).flatten(1)
		# writer = self.writer_embed(writer).flatten(1)
		# director = self.director_embed(director).flatten(1)
		# genre = self.genre_embed(genre).flatten(1)

		# LSTM Net
		seq_packed = pack_padded_sequence(seq, seq_len, batch_first=True, enforce_sorted=False)
		seq_packed_output, hidden = self.lstm(seq_packed)
		seq_unpacked = pad_packed_sequence(seq_packed_output, batch_first=True, padding_value=0, total_length=hps.seq_max_len)[0]
		seq_unpacked = seq_unpacked.flatten(1)
		seq_unpacked = self.LSTM_MLP(seq_unpacked)

		img = self.CNN(img)

		x = torch.cat([seq_unpacked, img], dim=1)
		x = self.MLP(x)

		# x = torch.cat([genre], dim=1)
		# x = self.MLP(ratio)
		# x = self.MLP(genre)

		# Res Net 18
		# x = self.CNN(img)


		return x
