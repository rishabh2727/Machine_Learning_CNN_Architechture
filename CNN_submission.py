import timeit
from collections import OrderedDict

import torch
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split


    
num_epochs = 5
batch_size = 200
log_interval = 100


class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        if in_channels == 1:
            in_features = 64*14*14
        elif in_channels == 3:
            in_features = 64*16*16
        self.fc1 = nn.Linear(in_features, 128)  
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.softmax(self.fc2(x),dim=1)
        return x
    


def load_dataset(
        dataset_name: str,
):
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))

        train_dataset, valid_dataset = random_split(full_dataset, [48000, 12000])

    elif dataset_name == "CIFAR10":
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    else:
        raise Exception("Unsupported dataset.")

    return train_dataset, valid_dataset





def eval(model, valid_loader, dataset, cross_loss,device):
    
    model.eval()  
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss += cross_loss(output, target).item()
    loss /= len(valid_loader.dataset)
    print(dataset + 'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(valid_loader.dataset),
                                                                               100. * correct / len(valid_loader.dataset)))
    
    return loss


def train(
        model,
        train_dataset,
        valid_dataset,
        device

):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-6) 
    cross_loss = nn.CrossEntropyLoss()
    
    
    
    train_loader = torch.utils.data.DataLoader(
       train_dataset, batch_size= batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size= batch_size, shuffle=False)
    
    
    eval(model,valid_loader,"Validation",cross_loss,device)

    for epoch in range(1, num_epochs +1):
        model.train()  
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = cross_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


        eval(model,valid_loader,"Validation",cross_loss,device)
    
        results = dict(
        model=model
    )

    return results

 

def CNN(dataset_name, device):


    if dataset_name == "CIFAR10":
        in_channels= 3
    elif dataset_name == "MNIST":
        in_channels = 1
    else:
        raise AssertionError(f'invalid dataset: {dataset_name}')

    model = Net(in_channels).to(device)

    train_dataset, valid_dataset = load_dataset(dataset_name)

    results = train(model, train_dataset, valid_dataset, device)

    return results

