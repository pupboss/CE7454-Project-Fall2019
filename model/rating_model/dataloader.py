import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IMDB(Dataset):
	def __init__(self, file_path, crew):
		self.data = pd.read_csv(file_path, sep='\t')
		self.crew = crew

	def __getitem__(self, index):

		rating = float(self.data['Average_Rating'][index])
		ratio = float(self.data['Ratio'][index])
		if ratio >=0:
			ratio = 1
		else:
			ratio = -1
		actor1 = self.crew('actor', self.data['Star_num1'][index])
		actor2 = self.crew('actor', self.data['Star_num1'][index])
		actor3 = self.crew('actor', self.data['Star_num1'][index])
		director = self.crew('director', self.data['Director_num'][index].split(',')[0])
		writer = self.crew('writer', self.data['Writer_num'][index].split(',')[0])

		rating = torch.Tensor([rating])
		ratio = torch.Tensor([ratio])
		writer = torch.LongTensor([writer])
		director = torch.LongTensor([director])
		actor = torch.LongTensor([actor1, actor2, actor3])

		return rating, ratio, actor, writer, director

	def __len__(self):
		return len(self.data)