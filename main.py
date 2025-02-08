import torch
import random
import argparse
import numpy as np
import torchvision
import datasets.transforms as T
from datasets import build_dataset
import torch.nn as nn
from pathlib import Path
import utils.misc as utils
from models import build_model
from typing import Optional, List
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)

    parser.add_argument('--seed', default=42, type=int)

# device  select GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0：cls  1: detect    2: seg
class1 = 0

# dataset load
    # classify
def main(args):

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



    model, criterion, postprocessors = build_model(args)

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if class1 == 0:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=True,transform=transform,download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False,transform=transform,download=True)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    elif class1 == 1:
        
        # coco dataset
        
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)

        # 1、batch_size
        batch_size = True
        if batch_size:
            train_dataset = DataLoader(dataset_train, batch_size=2,
                                        shuffle=True, collate_fn=utils.collate_fn)
            test_dataset = DataLoader(dataset_val, batch_size=2,
                                        shuffle=True, collate_fn=utils.collate_fn)

        # 2、batch_sampler
            # batch_sample   What?
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        # collate_fn  What?
            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)
            data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        











    






# the length of dataset
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

# building the model
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),            
            nn.ReLU(),            
            nn.MaxPool2d(2),            
            nn.Conv2d(32,32,5,1,2),            
            nn.ReLU(),            
            nn.MaxPool2d(2),            
            nn.Conv2d(32,64,5,1,2),            
            nn.ReLU(),            
            nn.MaxPool2d(2),            
            nn.Flatten(),            
            nn.Linear(64*4*4,64),            
            nn.Linear(64,10) 
        )
    def forward(self, x):
        x = self.model(x)
        return x

# create a model
model = Module()

model.to(device)

# loss and optim
criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.SGD(params=model.parameters(), lr=0.01)

# epoch
epochs = 10

for epoch in range(epochs):
    idx = 0
    for image, labels in train_loader:
        model.train()

        image = image.to(device)
        labels = labels.to(device)

        outputs = model(image)
        loss = criterion(outputs, labels)
        # optimization the weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    idx += 1
    # start test
    total_loss = 0
    total_accuary = 0
    accuary = 0
    with torch.no_grad():
        model.eval()
        for image, labels in test_loader:
            image = image.to(device)            
            labels = labels.to(device) 
            outputs = model(image)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            accuary = (outputs.argmax(1) == labels).sum()
            total_accuary += accuary
            print(f'total_accuary: {total_accuary / test_data_size * 100:.2f} %')

    if epoch == 9:
        torch.save(model, f'model_{epoch}.pth')
        print('the model is saved!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)