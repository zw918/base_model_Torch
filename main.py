import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# dataset load

    # classify
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
])

train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=True,transform=transform,download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False,transform=transform,download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# the length of dataset
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

# building the model
class Module(nn.Module):
    def __init__(self):
        super().__init__()


# create a model
model = Module()

# loss and optim
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(params=model.parameters(), lr=0.01)

# epoch
epochs = 10

for epoch in range(epochs):
    idx = 0
    for image, labels in train_loader:
        model.train()
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
            outputs = model(image)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            accuary = (outputs.argmax(1) == labels).sum()
            total_accuary += accuary
            print(f'total_accuary: {total_accuary / test_data_size * 100:.2f} %')

    if epoch == 9:
        torch.save(model, f'model_{epoch}.pth')
        print('the model is saved!')