from modules import *


class RatingModel(nn.Module):
	def __init__(self, len_actor, len_writer, len_director, len_genre, len_vocabulary):
		super(RatingModel, self).__init__()

		self.lstm = My_LSTM(len_vocabulary, hidden_size=128, output_size=64, num_layers=1)
		self.cnn = ResNet18(hidden_size=64)
		self.mlp = MLP(len_actor, len_writer, len_director, len_genre, lstm_h=64, cnn_h=64, output_size=1)

	def forward(self, attributes, actor, writer, director, genre, seq, seq_len, img):

		seq_unpacked, lstm_gate = self.lstm(seq, seq_len)
		img, img_gate = self.cnn(img)
		pre_rating = self.mlp(actor, writer, director, genre, attributes, seq_unpacked, img)

		return pre_rating, lstm_gate, img_gate


class BoxOfficeModel(nn.Module):
	def __init__(self, len_actor, len_writer, len_director, len_genre, len_vocabulary):
		super(BoxOfficeModel, self).__init__()

		self.lstm = My_LSTM(len_vocabulary, hidden_size=128, output_size=64, num_layers=1)
		self.cnn = ResNet18(hidden_size=64)
		self.mlp = MLP(len_actor, len_writer, len_director, len_genre,lstm_h=64, cnn_h=64, output_size=2)
		self.softmax = nn.Softmax(dim=1)


	def forward(self, attributes, actor, writer, director, genre, seq, seq_len, img):
		seq_unpacked, lstm_gate  = self.lstm(seq, seq_len)
		img, img_gate = self.cnn(img)
		pre_rating = self.mlp(actor, writer, director, genre, attributes, seq_unpacked, img)
		pre_rating = self.softmax(pre_rating)

		return pre_rating, lstm_gate, img_gate
