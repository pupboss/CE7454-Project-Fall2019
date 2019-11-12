import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
from hyperparams import Hyperparams as hps
import bcolz
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IMDB(Dataset):
	def __init__(self, file_path, crew):
		self.data = pd.read_csv(file_path, sep='\t')
		self.crew = crew

		vectors = bcolz.open(f'{hps.glove_path}6B.50.dat')[:]
		words = pickle.load(open(f'{hps.glove_path}6B.50_words.pkl', 'rb'))
		word2idx = pickle.load(open(f'{hps.glove_path}6B.50_idx.pkl', 'rb'))

		glove = {w: vectors[word2idx[w]] for w in words}

		self.glove = glove
		self.img_dirpath = hps.poster_path
		self.transform = transforms.Compose([
									transforms.Resize([224,224]),
									transforms.ToTensor(),
									transforms.Normalize(mean=(.5,.5,.5),std=(.5,.5,.5))])

	def load_image(self,img_path):
		img = Image.open(self.img_dirpath + img_path + ".jpg")
		if img.mode != "RGB":
			img = img.convert("RGB")
		img = self.transform(img)

		return img

	def __getitem__(self, index):

		rating = float(self.data['averageRating'][index])
		ratio = float(self.data['Ratio'][index])
		story_line = (self.data['Story_Line'][index]).split(' ')
		Year = (float(self.data['startYear'][index]) - 2009.41) / 5.01
		Run_time = (float(self.data['runtimeMinutes'][index]) - 105.48) / 17.93
		Budget = (float(self.data['Budget'][index]) - 2.33e7) / 3.76e7

		actor1 = self.crew('actor', self.data['Star_num1'][index])
		actor2 = self.crew('actor', self.data['Star_num2'][index])
		actor3 = self.crew('actor', self.data['Star_num3'][index])
		director = self.crew('director', self.data['directors'][index].split(',')[0])
		writer = self.crew('writer', self.data['writers'][index].split(',')[0])
		genre = [self.crew('genre', x) for x in self.data['genres'][index].split(',')]
		genre_len = self.crew.cal_len('genre')

		profitable_label = self.data['Profitable'][index]

		genre_onehot = torch.FloatTensor(genre_len).zero_()
		for i in genre:
			genre_onehot[i]+=1

		rating = torch.Tensor([rating])
		attributes = torch.Tensor([Year, Run_time, Budget])
		writer = torch.LongTensor([writer])
		director = torch.LongTensor([director])
		genre = torch.LongTensor([genre])
		actor = torch.LongTensor([actor1, actor2, actor3])
		profitable_label = torch.LongTensor([profitable_label])

		seq_max_len = hps.seq_max_len
		seq_len = len(story_line[:seq_max_len])
		seq = np.zeros((seq_max_len, 50))

		for i, word in enumerate(story_line[:seq_max_len]):
			try:
				seq[i] = self.glove[word]
			except KeyError:
				seq[i] = 0

		seq = torch.Tensor(seq)

		img = self.load_image(img_path=self.data['tconst'][index])

		return rating, attributes, actor, writer, director, genre_onehot, seq, seq_len, img, profitable_label

	def __len__(self):
		return len(self.data)