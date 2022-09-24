#!/usr/bin/env python

import syncbn

import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100
from time import time

torch.set_num_threads(1)

def string_time(elapsed):
    return "%im %is" %(int(elapsed / 60), int(elapsed % 60))

def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        self.bn1 = syncbn.SyncBatchNorm(128)  # to be replaced with SyncBatchNorm
        self.bn2 = syncbn.SyncBatchNorm(100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.dropout2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        # i changed the position of BN layers so that (dim=0) would work!
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        # same :)
        output = self.bn2(x)

        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run_training(rank, size):
    torch.manual_seed(0)
    NUM_EPOCHS = 10
    AGGREGATION = 1 # if int > 1, only one of AGGREGATION steps will call optimizer
    assert AGGREGATION > 0

    dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
    )
    # where's the validation dataset?
    train_sz = int(len(dataset) * 0.8)
    val_sz = len(dataset) - train_sz

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_sz, val_sz])
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset, size, rank), batch_size=64)
    val_loader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset, size, rank), batch_size=64)

    model = Net()
    device = torch.device("cuda")  # replace with "cuda" afterwards
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    start_time = time()

    for epoch in range(NUM_EPOCHS):
        # ================== Training ==================
        model.train()
        epoch_loss = torch.zeros((1,), device=device)
        epoch_corrects = 0
        num_elems = 0

        for iter, batch in enumerate(train_loader):
            data, target = batch[0].to(device), batch[1].to(device)
            num_elems += data.shape[0]

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.detach()
            loss.backward()

            if (iter + 1) % AGGREGATION == 0:
                average_gradients(model)
                optimizer.step()

            epoch_corrects += (output.argmax(dim=1) == target).float().sum()

        train_loss.append(epoch_loss.item() / len(train_loader))
        train_acc.append(epoch_corrects.cpu() / num_elems)

        print(f"Train epoch {epoch + 1}, Rank {dist.get_rank()}, loss: {train_loss[-1]}, acc: {train_acc[-1]}, time: " 
              + string_time(time() - start_time))

        # ================== Validation ==================
        model.eval()
        epoch_loss = torch.zeros((1,), device=device)
        epoch_corrects = 0
        num_elems = 0

        # likely calculating val acc this way is not exactly correct, 
        # but we will fix it in the next task!
        for iter, batch in enumerate(val_loader):
            data, target = batch[0].to(device), batch[1].to(device)
            num_elems += data.shape[0]

            with torch.no_grad():
                output = model(data)
                epoch_corrects += (output.argmax(dim=1) == target).float().sum()
                loss = torch.nn.functional.cross_entropy(output, target)
                epoch_loss += loss.detach()

        val_loss.append(epoch_loss.item() / len(val_loader))
        val_acc.append(epoch_corrects.cpu() / num_elems)

        print(f"Eval epoch {epoch + 1}, Rank {dist.get_rank()}, loss: {val_loss[-1]}, acc: {val_acc[-1]}, time: " 
              + string_time(time() - start_time))
    
    if rank == 0:
        np.save("train_loss.npy", np.array(train_loss))
        np.save("train_acc.npy", np.array(train_acc))
        np.save("val_loss.npy", np.array(val_loss))
        np.save("val_acc.npy", np.array(val_acc))



if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend="gloo")