class Hyperparams(object):
	all_data = "../data/imdb.tsv"
	train_path = '../data/train.tsv'
	val_path = '../data/val.tsv'
	lookuptable_path = '../data/lookuptable.pkl'
	poster_path = '../data/posters/'

	batch_size = 16
	min_frequency = 5
	seq_max_len = 100

