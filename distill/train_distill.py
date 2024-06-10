import os
import random
import monai
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import cv2
import torch.nn.functional as F
from matplotlib import pyplot as plt
import argparse
from med_dataset import Distill_Dataset
from models.swin import SwinTransformer

parser = argparse.ArgumentParser()
parser.add_argument(
    "-data_root", type=str, default="?",
    help="Path to the npy data root."
)
parser.add_argument(
    "-work_dir", type=str, default="./work_dir",
    help="Path to the working directory where checkpoints and logs will be saved."
)
parser.add_argument(
    "-num_epochs", type=int, default=10,
    help="Number of epochs to train."
)
parser.add_argument(
    "-batch_size", type=int, default=32,
    help="Batch size."
)
parser.add_argument(
    "-num_workers", type=int, default=8,
    help="Number of workers for dataloader."
)
parser.add_argument(
    "-lr", type=float, default=1e-4,
    help="Learning rate."
)
parser.add_argument(
    "-weight_decay", type=float, default=0.01,
    help="Weight decay."
)

args = parser.parse_args()
work_dir = args.work_dir
data_root = args.data_root
num_epochs = args.num_epochs
batch_size = args.batch_size
num_workers = args.num_workers
lr = args.lr
weight_decay = args.weight_decay

torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

train_dataset = Distill_Dataset(data_root=data_root)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
# criterion = nn.MSELoss(reduction='mean')
criterion = nn.L1Loss(reduction='mean')
model = SwinTransformer()
encoder_weight_file = "?"
model.load_state_dict(torch.load(encoder_weight_file)["model"])
model = model.cuda()

optimizer = optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.9,
    patience=5,
    cooldown=0
)

start_epoch = 0
best_loss = 1
train_losses = []
print("############start train################")
for epoch in range(start_epoch + 1, num_epochs):
    epoch_loss = 0
    epoch_start_time = time()
    pbar = tqdm(train_loader)
    print(f"-----------Epoch {epoch}----------------")
    for step, (image, embedding) in enumerate(pbar):
        image = image.cuda()
        embedding = embedding.cuda()
        optimizer.zero_grad()
        pred, _ = model(image)
        loss = criterion(pred, embedding)
        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
    epoch_end_time = time()
    epoch_loss_reduced = epoch_loss / len(train_loader)
    train_losses.append(epoch_loss_reduced)
    lr_scheduler.step(epoch_loss_reduced)
    model_weights = model.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": epoch_loss_reduced,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, join(work_dir, "swin_encoder2.pth"))
    if epoch_loss_reduced < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
        best_loss = epoch_loss_reduced
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, join(work_dir, "Swin_encoder.pth"))
    epoch_loss_reduced = 1e10
    # %% plot loss
    plt.plot(train_losses)
    plt.title("MAE Loss for Distillation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(join(work_dir, "train_loss_new.png"))
    plt.close()
