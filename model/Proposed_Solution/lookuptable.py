import pandas as pd
import pickle
from hyperparams import Hyperparams as hps
from collections import Counter


class LookupTable(object):

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

		self.word2idx = {}
		self.idx2word = {}
		self.word_idx = 0

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

	def add_word(self, word):
		if not (word in self.word2idx):
			self.word2idx[word] = self.word_idx
			self.idx2word[self.word_idx] = word
			self.word_idx +=1


	def __call__(self, something, index):
		if something == 'actor':
			try:
				return self.actor2idx[index]
			except KeyError:
				return self.actor2idx['\\N']

		if something == 'director':
			try:
				return self.director2idx[index]
			except KeyError:
				return self.director2idx['\\N']

		if something == 'writer':
			try:
				return self.writer2idx[index]
			except KeyError:
				return self.writer2idx['\\N']

		if something == 'genre':
			try:
				return self.genre2idx[index]
			except KeyError:
				return self.genre2idx['\\N']

		if something == 'word':
			try:
				return self.word2idx[index]
			except KeyError:
				return self.word2idx['<unknown>']


	def cal_len(self, something):
		if something == 'actor':
			return len(self.actor2idx)
		if something == 'director':
			return len(self.director2idx)
		if something == 'writer':
			return len(self.writer2idx)
		if something == 'genre':
			return len(self.genre2idx)
		if something == 'word':
			return len(self.word2idx)

	def update_vocabulary(self, vocabulary):

		for word in vocabulary:
			if vocabulary[word] <= hps.min_frequency:
				continue
			else:
				self.add_word(word)


def add_info(file):
	data = pd.read_csv(file, sep='\t')
	lookuptable = LookupTable()

	lookuptable.add_actor('\\N')
	lookuptable.add_writer('\\N')
	lookuptable.add_director('\\N')
	lookuptable.add_genre('\\N')
	lookuptable.add_word('<padding>')
	lookuptable.add_word('<unknown>')

	vocabulary = Counter()

	for row in data.iterrows():
		lookuptable.add_actor(row[1]['Star_num1'])
		lookuptable.add_actor(row[1]['Star_num2'])
		lookuptable.add_actor(row[1]['Star_num3'])

		if row[1]['directors'] == '\\N':
			pass
		else:
			for each in row[1]['directors'].split(','):
				lookuptable.add_director(each)

		if row[1]['writers'] == '\\N':
			pass
		else:
			for each in row[1]['writers'].split(','):
				lookuptable.add_writer(each)

		if row[1]['genres'] == '\\N':
			pass
		else:
			for each in row[1]['genres'].split(','):
				lookuptable.add_genre(each)

		for word in row[1]['Story_Line'].lower().split(' '):
			vocabulary[word] += 1

	lookuptable.update_vocabulary(vocabulary)

	return lookuptable

def main():

	lookuptable = add_info(hps.all_data)
	with open(hps.lookuptable_path, 'wb') as f:
		pickle.dump(lookuptable, f)


if __name__ == '__main__':
	main()
