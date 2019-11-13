import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pickle
import torch

import torch.nn as nn
from lookuptable import LookupTable
from torch.utils.data import DataLoader
from hyperparams import Hyperparams as hps
from dataloader import IMDB
from model import BoxOfficeModel
import torch.onnx
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def run(train_loader, val_loader, model, loss_function, optimizer, epoch):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(5,100,5)), gamma=(1/1.1))

    while True:
        if epoch > 100:
            break

        model.train()

        log_loss = 0

        for (rating, attributes, actor, writer, director, genre, seq, seq_len, img, profitable_label) in tqdm(train_loader, desc="Training"):
            actor = actor.to(device)
            writer = writer.to(device)
            director = director.to(device)
            genre = genre.to(device)
            seq = seq.to(device)
            img = img.to(device)
            profitable_label = profitable_label.to(device)
            profitable_label = profitable_label.squeeze(1)
            attributes = attributes.to(device)

            predicted_box_office = model(attributes, actor, writer, director, genre, seq, seq_len, img)

            optimizer.zero_grad()

            loss = loss_function(predicted_box_office, profitable_label)

            loss.backward()

            optimizer.step()

            log_loss += loss.item()

            scheduler.step(epoch)

        log = "epoch:{}, step:{}, loss:{:.3f}".format(
            epoch, len(train_loader), log_loss / len(train_loader))
        print(log)

        torch.save({
            'Model_state_dict': model.state_dict(),
            'epoch': epoch
        }, '/home/shenmeng/tmp/imdb/ckpt/ckpt_boxoffice{}.pth'.format(epoch))

        val(val_loader, model)

        epoch += 1


def val(val_loader, model):
    model.eval()

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for (rating, attributes, actor, writer, director, genre, seq, seq_len, img, profitable_label) in val_loader:

            actor = actor.to(device)
            writer = writer.to(device)
            director = director.to(device)
            genre = genre.to(device)
            seq = seq.to(device)
            img = img.to(device)
            attributes = attributes.to(device)

            profitable_label = profitable_label.to(device)
            profitable_label = profitable_label.squeeze(1)

            predicted_box_office = model(attributes, actor, writer, director, genre, seq, seq_len, img)
            _, index = torch.max(predicted_box_office, 1)

            index = index.detach().cpu().numpy()
            profitable_label = profitable_label.detach().cpu().numpy()

            for i in range(len(index)):
                if index[i] == 1 and profitable_label[i] == 1:
                    TP += 1
                elif index[i] == 1 and profitable_label[i] == 0:
                    FP +=1
                elif index[i] == 0 and profitable_label[i] == 1:
                    FN +=1
                elif index[i] == 0 and profitable_label[i] == 0:
                    TN +=1

        pre = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_Score = (2 * pre * recall) / (pre + recall)


    print("Pre: {:.3f} , Recall: {:.3f} , F1: {:.3f}".format(pre, recall, F1_Score))


def main():
    load_ckpt = False
    which_ckpt = "/home/shenmeng/tmp/imdb/ckpt/ckpt_1.pth"
    with open(hps.lookuptable_path, 'rb') as f:
        lookuptable = pickle.load(f)
    train_loader = DataLoader(IMDB(hps.train_path, lookuptable),
                              batch_size=hps.batch_size,
                              shuffle=True, num_workers=32)
    val_loader = DataLoader(IMDB(hps.val_path, lookuptable),
                            batch_size=hps.batch_size,
                            shuffle=False,
                            num_workers=32)
    loss_function = nn.CrossEntropyLoss().to(device)
    model = BoxOfficeModel(
        lookuptable.cal_len('actor'),
        lookuptable.cal_len('writer'),
        lookuptable.cal_len('director'),
        lookuptable.cal_len('genre'),
        lookuptable.cal_len('word')
    )

    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-3)
    epoch = 1

    if load_ckpt:
        checkpoint = torch.load(which_ckpt)
        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['Model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    run(train_loader, val_loader, model, loss_function, optimizer, epoch)


main()