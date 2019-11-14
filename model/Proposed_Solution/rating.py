import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
import torch
import torch.nn as nn
from lookuptable import LookupTable
from torch.utils.data import DataLoader
from hyperparams import Hyperparams as hps
from dataloader import IMDB
from model import RatingModel
import torch.onnx
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(train_loader, val_loader, model, loss_function, optimizer, epoch):

	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(5,200,5)), gamma=1/1.1)

	plt_train_loss = []
	plt_val_loss = []

	while True:
		if epoch>200:
			break

		model.train()

		log_loss = 0
		step = 0
		for (rating, attributes, actor, writer, director, genre, seq, seq_len, img, profitable_label) in tqdm(train_loader,desc="Training"):

			rating = rating.to(device)
			actor = actor.to(device)
			writer = writer.to(device)
			director = director.to(device)
			genre = genre.to(device)
			seq = seq.to(device)
			img = img.to(device)
			attributes = attributes.to(device)

			pre_rating = model(attributes, actor, writer, director, genre, seq, seq_len, img)

			optimizer.zero_grad()

			loss = loss_function(pre_rating, rating)

			loss.backward()

			clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

			optimizer.step()

			log_loss += loss.item()

			scheduler.step(epoch)
			if step % 50 == 0:
				plt_train_loss.append(float(loss.item()))
			step += 1

		with open("/home/shenmeng/tmp/imdb/results/train_loss_no_gate_200_epoch.txt","w") as f:
			for record in plt_train_loss:
				f.write(str(record)+'\n')

		log = "epoch:{}, step:{}, loss:{:.3f}".format(
		epoch, len(train_loader), log_loss/len(train_loader))
		print(log)

		torch.save({
			'Model_state_dict': model.state_dict(),
			'epoch': epoch
		}, '/home/shenmeng/tmp/imdb/ckpt/ckpt_Adam_no_gate_200_epoch{}.pth'.format(epoch))

		val_loss = val(val_loader, model)
		plt_val_loss.append(val_loss)

		with open("/home/shenmeng/tmp/imdb/results/val_loss_no_gate_200_epoch.txt", "w") as f:
			for record in plt_val_loss:
				f.write(str(record)+'\n')

		epoch += 1



def val(val_loader, model):

	model.eval()

	loss = 0
	with torch.no_grad():
		for (rating, attributes, actor, writer, director, genre, seq, seq_len, img, profitable_label) in val_loader:
			rating = rating.to(device)
			actor = actor.to(device)
			writer = writer.to(device)
			director = director.to(device)
			genre = genre.to(device)
			seq = seq.to(device)
			img = img.to(device)
			attributes = attributes.to(device)

			predicted_rating = model(attributes, actor, writer, director, genre, seq, seq_len, img)

			predicted_rating = predicted_rating.squeeze(1).detach().cpu().numpy()
			rating = rating.squeeze(1).detach().cpu().numpy()
			loss += abs(predicted_rating - rating).sum()


		val_loss = float(loss/(len(val_loader))/hps.batch_size)
		print("Validation Result:{:.3f}".format(val_loss))

	return val_loss


def main():
	load_ckpt = False
	which_ckpt = "/home/shenmeng/tmp/imdb/ckpt/ckpt_1.pth"
	with open(hps.lookuptable_path,'rb') as f:
		lookuptable = pickle.load(f)
	train_loader = DataLoader(IMDB(hps.train_path, lookuptable),
							  batch_size=hps.batch_size,
							  shuffle=True,num_workers=32)
	val_loader = DataLoader(IMDB(hps.val_path, lookuptable),
							batch_size=hps.batch_size,
							shuffle=False,
							num_workers=32)
	loss_function = nn.L1Loss().to(device)
	model = RatingModel(lookuptable.cal_len('actor'),
						lookuptable.cal_len('writer'),
						lookuptable.cal_len('director'),
						lookuptable.cal_len('genre'),
						lookuptable.cal_len('word')
						)
	model = model.to(device)

	params = [p for p in model.parameters() if p.requires_grad]

	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name)
	optimizer = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-3)
	epoch = 1

	if load_ckpt:
		checkpoint = torch.load(which_ckpt)
		epoch = checkpoint['epoch'] + 1
		model.load_state_dict(checkpoint['Model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

	run(train_loader, val_loader, model, loss_function, optimizer, epoch)

main()