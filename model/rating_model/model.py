from modules import *


class RatingModel(nn.Module):
	def __init__(self, len_actor, len_writer, len_director, len_genre, len_vocabulary):
		super(RatingModel, self).__init__()

		self.lstm = My_LSTM(len_vocabulary, hidden_size=128, num_layers=1)
		self.cnn = ResNet18(hidden_size=128)
		self.mlp = MLP(len_actor, len_writer, len_director, len_genre, lstm_h=128, cnn_h=128, output_size=1)

	def forward(self, attributes, actor, writer, director, genre, seq, seq_len, img):

		seq_unpacked = self.lstm(seq, seq_len)
		img = self.cnn(img)
		pre_rating = self.mlp(actor, writer, director, genre, attributes, seq_unpacked, img)

		return pre_rating


class BoxOfficeModel(nn.Module):
	def __init__(self, len_actor, len_writer, len_director, len_genre, len_vocabulary):
		super(BoxOfficeModel, self).__init__()

		self.lstm = My_LSTM(len_vocabulary, hidden_size=128, num_layers=1)
		self.cnn = ResNet18(hidden_size=128)
		self.mlp = MLP(len_actor, len_writer, len_director, len_genre,lstm_h=128, cnn_h=128, output_size=2)
		self.softmax = nn.Softmax(dim=1)


	def forward(self, attributes, actor, writer, director, genre, seq, seq_len, img):
		seq_unpacked = self.lstm(seq, seq_len)
		img = self.cnn(img)
		pre_rating = self.mlp(actor, writer, director, genre, attributes, seq_unpacked, img)
		pre_rating = self.softmax(pre_rating)

		return pre_rating
