import pandas as pd
import pickle
from hyperparams import Hyperparams as hps


class Crew(object):
	"""docstring for Crew"""
	def __init__(self):
		self.actor2idx = {}
		self.idx2actor = {}
		self.actor_idx = 0

		self.writer2idx = {}
		self.idx2writer = {}
		self.writer_idx = 0

		self.director2idx = {}
		self.idx2director = {}
		self.director_idx = 0

	def add_actor(self, actor):
		if not (actor in self.actor2idx):
			self.actor2idx[actor] = self.actor_idx
			self.idx2actor[self.actor_idx] = actor
			self.actor_idx += 1

	def add_writer(self, writer):
		if not (writer in self.writer2idx):
			self.writer2idx[writer] = self.writer_idx
			self.idx2writer[self.writer_idx] = writer
			self.writer_idx += 1

	def add_director(self, director):
		if not (director in self.director2idx):
			self.director2idx[director] = self.director_idx
			self.idx2director[self.director_idx] = director
			self.director_idx += 1

	def __call__(self, people, imdb_num):
		if people == 'actor':
			try:
				return self.actor2idx[imdb_num]
			except KeyError:
				return self.actor2idx['\\N']
		if people == 'director':
			try:
				return self.director2idx[imdb_num]
			except KeyError:
				return self.director2idx['\\N']
		if people == 'writer':
			try:
				return self.writer2idx[imdb_num]
			except KeyError:
				return self.writer2idx['\\N']

	def cal_len(self, people):
		if people == 'actor':
			return len(self.actor2idx)
		if people == 'director':
			return len(self.director2idx)
		if people == 'writer':
			return len(self.writer2idx)


def add_crew(file):
	data = pd.read_csv(file, sep='\t')
	crew = Crew()

	crew.add_actor('\\N')
	crew.add_writer('\\N')
	crew.add_director('\\N')

	for row in data.iterrows():
		crew.add_actor(row[1]['Star_num1'])
		crew.add_actor(row[1]['Star_num2'])
		crew.add_actor(row[1]['Star_num3'])

		if row[1]['Director_num'] == '\\N':
			pass
		else:
			for each in row[1]['Director_num'].split(','):
				crew.add_director(each)

		if row[1]['Writer_num'] == '\\N':
			pass
		else:
			for each in row[1]['Writer_num'].split(','):
				crew.add_writer(each)
	return crew


def main():
	crew = add_crew(hps.all_data)
	with open(hps.crew_path, 'wb') as f:
		pickle.dump(crew, f)


if __name__ == '__main__':
	main()
