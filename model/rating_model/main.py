import os
import pickle
import torch
import torch.nn as nn
import time
from crew import Crew
from torch.utils.data import DataLoader
from hyperparams import Hyperparams as hps
from dataloader import IMDB
from model import RatingModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(train_loader, val_loader, model, loss_function, optimizer):

	epoch = 1

	while True:

		model.train()

		log_loss = 0

		for step, (rating, ratio, actor, writer, director) in enumerate(train_loader):

			rating = rating.to(device)
			ratio = ratio.to(device)
			actor = actor.to(device)
			writer = writer.to(device)
			director = director.to(device)
			predicted_rating = model(actor, writer, director)
			# predicted_ratio = model(actor, writer, director)
			optimizer.zero_grad()

			loss = loss_function(predicted_rating, rating)
			# loss = loss_function(predicted_ratio, ratio)
			loss.backward()

			optimizer.step()

			log_loss += loss.item()

			if step % 100 == 99:
				log = "epoch:{}, step:{}, loss:{:.3f}".format(
				epoch, step*hps.batch_size, log_loss/100)
				print(log)
				log_loss = 0

		val(val_loader, model)

		epoch += 1

def val(val_loader, model):

	model.eval()

	loss = 0

	for step, (rating, ratio, actor, writer, director) in enumerate(val_loader):
		rating = rating.to(device)
		ratio = ratio.to(device)
		actor = actor.to(device)
		writer = writer.to(device)
		director = director.to(device)
		predicted_rating = model(actor, writer, director)
		# predicted_ratio = model(actor, writer, director)
		loss += abs(predicted_rating - rating)

	print("Validation Result:", loss/(step+1))


def main():

	with open(hps.crew_path ,'rb') as f:
			crew = pickle.load(f)
	train_loader = DataLoader(IMDB(hps.train_path, crew),
							batch_size = hps.batch_size,
							shuffle = True,
							num_workers = 16)
	val_loader = DataLoader(IMDB(hps.val_path, crew),
							batch_size = 1,
							shuffle = False,
							num_workers = 16)
	loss_function = nn.L1Loss().to(device)
	model = RatingModel(crew.cal_len('actor'),
						crew.cal_len('writer'),
						crew.cal_len('director')
						)
	model = model.to(device)
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.001)
	run(train_loader, val_loader, model, loss_function, optimizer)


main()