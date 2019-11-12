import pandas as pd
import pickle
from hyperparams import Hyperparams as hps
import bcolz
import numpy as np
import os
from tqdm import tqdm

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

		self.genre2idx = {}
		self.idx2genre = {}
		self.genre_idx = 0

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

	def add_genre(self, genre):
		if not (genre in self.genre2idx):
			self.genre2idx[genre] = self.genre_idx
			self.idx2genre[self.genre_idx] = genre
			self.genre_idx += 1


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

		if people == 'genre':
			try:
				return self.genre2idx[imdb_num]
			except KeyError:
				return self.genre2idx['\\N']


	def cal_len(self, people):
		if people == 'actor':
			return len(self.actor2idx)
		if people == 'director':
			return len(self.director2idx)
		if people == 'writer':
			return len(self.writer2idx)
		if people == 'genre':
			return len(self.genre2idx)


def add_crew(file):
	data = pd.read_csv(file, sep='\t')
	crew = Crew()

	crew.add_actor('\\N')
	crew.add_writer('\\N')
	crew.add_director('\\N')
	crew.add_genre('\\N')

	for row in data.iterrows():
		crew.add_actor(row[1]['Star_num1'])
		crew.add_actor(row[1]['Star_num2'])
		crew.add_actor(row[1]['Star_num3'])


		if row[1]['directors'] == '\\N':
			pass
		else:
			for each in row[1]['directors'].split(','):
				crew.add_director(each)

		if row[1]['writers'] == '\\N':
			pass
		else:
			for each in row[1]['writers'].split(','):
				crew.add_writer(each)

		if row[1]['genres'] == '\\N':
			pass
		else:
			for each in row[1]['genres'].split(','):
				crew.add_genre(each)

	return crew

def vocabulary():
	words = []
	idx = 0
	word2idx = {}
	vectors = bcolz.carray(np.zeros(1), rootdir=f'{hps.glove_path}6B.50.dat', mode='w')

	with open(f'{hps.glove_path}glove.6B.50d.txt', 'rb') as f:
		for l in f:
			line = l.decode().split()
			word = line[0]
			words.append(word)
			word2idx[word] = idx
			idx += 1
			vect = np.array(line[1:]).astype(np.float)
			vectors.append(vect)

	vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{hps.glove_path}6B.50.dat', mode='w')
	vectors.flush()
	pickle.dump(words, open(f'{hps.glove_path}6B.50_words.pkl', 'wb'))
	pickle.dump(word2idx, open(f'{hps.glove_path}6B.50_idx.pkl', 'wb'))

def main():

	if not os.path.isfile(hps.glove_path +'6B.50_words.pkl') and not os.path.isfile(hps.glove_path +'6B.50_idx.pkl') :
		vocabulary()

	crew = add_crew(hps.all_data)
	with open(hps.crew_path, 'wb') as f:
		pickle.dump(crew, f)


if __name__ == '__main__':
	main()
