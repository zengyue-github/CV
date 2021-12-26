# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 10:06:02 2021

@author: 10673
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size = 128

transform = transforms.Compose([transforms.Resize(252), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#Data set
train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                           train=True, 
                                           transform=transform,  
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                          train=False, 
                                          transform=transforms)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

net = torchvision.models.alexnet(pretrained=False, num_classes = 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)


num_epochs = 64
for epoch in range(num_epochs):
    count =1
    for i, (images, labels) in enumerate(train_loader):  
        count += 1
        print("epoch, count:",epoch,count)
        
        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch: ", epoch, " loss: ", loss.item())

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        #images = images.to(device)
        #labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))