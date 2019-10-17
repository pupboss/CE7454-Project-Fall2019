import torch
import torch.nn as nn

class RatingModel(nn.Module):
	def __init__(self, len_actor, len_writer, len_director):
		super(RatingModel, self).__init__()
		embed_size = 128

		self.actor_embed = nn.Embedding(len_actor, embed_size)
		self.writer_embed = nn.Embedding(len_writer, embed_size)
		self.director_embed = nn.Embedding(len_director, embed_size)
		self.MLP = nn.Sequential(
			nn.Linear(embed_size*5, 512),
			nn.Dropout(p=0.5),
			nn.Linear(512,256),
			nn.Dropout(p=0.5),
			nn.Linear(256,1))

	def forward(self, actor, writer, director):
		actor = self.actor_embed(actor)
		writer = self.writer_embed(writer)
		director = self.director_embed(director)
		x = torch.cat([actor, writer, director], dim = 1).flatten(1)
		x = self.MLP(x)
		
		return x