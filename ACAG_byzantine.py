# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:40:57 2022

@author: ChandlerZhu
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import copy
from torch import Tensor
transform=torchvision.transforms.Compose(
[transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
n_epochs = 3
num_straggler = 4
batch_size_train = 20
batch_size_test = 400
learning_rate = 0.3
momentum = 0
log_interval = 10
M = 10
typeMNIST = 'balanced'
length_out =10
miu = 1e4
scale = 2
miu_e = 1e-4
# transform=torchvision.transforms.Compose(
# [transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
dataset_workers = []
sampler_workers = []
loader_workers = []
dataset_workers_attack = []
sampler_workers_attack = []
loader_workers_attack = []
'''dataset_global = torchvision.datasets.MNIST('./dataset/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))'''
# dataset_global = torchvision.datasets.EMNIST(
#                             root='./data',
#                             train=True,
#                             transform=torchvision.transforms.ToTensor(),
#                             download = False,
#                             split = typeMNIST
#                             )
dataset_global = torchvision.datasets.CIFAR10(
                            root='./data',
                            train=True,
                            transform=transform,
                            download = False,
                            )
length = len(dataset_global)
# index = dataset_global.targets.argsort()
index = np.array(dataset_global.targets).argsort()
# index = torch.randperm(length)
# index = np.array(torch.randperm(length))

for i in range(M):
    # dataset_workers.append(torchvision.datasets.EMNIST(
    #                         root='./data',
    #                         train=True,
    #                         transform=torchvision.transforms.ToTensor(),
    #                         download = False,
    #                         split = typeMNIST
    #                         ))
    dataset_workers.append(torchvision.datasets.CIFAR10(
                            root='./data',
                            train=True,
                            transform=transform,
                            download = False,
                            ))
    dataset_workers_attack.append(torchvision.datasets.CIFAR10(
                            root='./data',
                            train=True,
                            transform=transform,
                            download = False,
                            ))


    dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[index], np.array(dataset_workers[i].targets)[index].tolist()
    # dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[index], dataset_workers[i].targets[index]
    dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[int(length / M * i) : int(length / M * (i + 1))]\
    , dataset_workers[i].targets[int(length / M * i) : int(length / M * (i + 1))]

    sampler_workers.append(sampler.BatchSampler(sampler.RandomSampler(data_source=dataset_workers[i], replacement=True), batch_size=batch_size_train, drop_last=False))
    loader_workers.append(torch.utils.data.DataLoader(dataset_workers[i],batch_sampler=sampler_workers[i], shuffle=False))
    
    dataset_workers_attack[i].data, dataset_workers_attack[i].targets = dataset_workers_attack[i].data[index], (9-np.array(dataset_workers_attack[i].targets)[index]).tolist()
    # dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[index], dataset_workers[i].targets[index]
    dataset_workers_attack[i].data, dataset_workers_attack[i].targets = dataset_workers_attack[i].data[int(length / M * i) : int(length / M * (i + 1))]\
    , dataset_workers_attack[i].targets[int(length / M * i) : int(length / M * (i + 1))]

    sampler_workers_attack.append(sampler.BatchSampler(sampler.RandomSampler(data_source=dataset_workers_attack[i], replacement=True), batch_size=batch_size_train, drop_last=False))
    loader_workers_attack.append(torch.utils.data.DataLoader(dataset_workers_attack[i],batch_sampler=sampler_workers_attack[i], shuffle=False))


# test_dataset = torchvision.datasets.EMNIST(
#                             root='./data',
#                             train=False,
#                             transform=torchvision.transforms.ToTensor(),
#                             download = False,
#                             split = typeMNIST
#                             )
test_dataset = torchvision.datasets.CIFAR10(
                            root='./data',
                            train=False,
                            transform=transform,
                            download = False,
                            )
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_sampler = sampler.BatchSampler(sampler.RandomSampler(data_source=test_dataset, replacement=False), batch_size=batch_size_test, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset
  ,batch_sampler=test_sampler)

test_sampler_clear = sampler.BatchSampler(sampler.RandomSampler(data_source=test_dataset, replacement=False), batch_size=100, drop_last=False)
test_loader_clear = torch.utils.data.DataLoader(test_dataset
  ,batch_sampler=test_sampler_clear)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 384) 
        self.fc2 = nn.Linear(384, 192) 
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def relu(x):
    if x < 0:
        return 0
    else:
        return x
criterion = nn.CrossEntropyLoss()
network1 = Net()
S = M
U = 5
epochs = 10
iters = 600
acc = 70
param_num = 10
A = 1

#%% ACSA
root_dataset = copy.deepcopy(test_dataset)
index = np.array(torch.randperm(len(root_dataset)))
root_dataset.data, root_dataset.targets = root_dataset.data[index], np.array(root_dataset.targets)[index].tolist()
root_dataset.data, root_dataset.targets = root_dataset.data[0:3000], root_dataset.targets[0:3000]
test_sampler_clear = sampler.BatchSampler(sampler.RandomSampler(data_source=root_dataset, replacement=False), batch_size=800, drop_last=False)
test_loader_clear = torch.utils.data.DataLoader(root_dataset, batch_sampler=test_sampler_clear)
train_losses_ACSA = []
test_losses_ACSA = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
# grad_avg = {}
global_direction = {}
for l in range(param_num):
    # grad_avg[l] = 0
    global_direction[l] = 0
for m in range(M):
    optimizer_worker.append(0)
for i in range(iters):
    if i % epochs == 0:
        network.eval()
        test_loss_ACSA = 0
        correct_ACSA = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data)
            # output = network(data)
            test_loss_ACSA += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ACSA += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ACSA /= len(test_loader.dataset)
        test_losses_ACSA.append(100. * correct_ACSA / len(test_loader.dataset))
    # optimizer.param_groups[0]['lr'] *= 0.99
    network_temp = copy.deepcopy(network)
    optimizer_temp = optim.SGD(network_temp.parameters(), lr=0.1, momentum=momentum)
    network_temp.train()
    for batch_idx, (data, target) in enumerate(test_loader_clear):
        # output = network_temp(data)
        # loss = criterion(output, target)
        # loss.backward()
        # optimizer_temp.step()
        data_temp = data
        target_temp = target
        break
    for iter in range(U):
        output = network_temp(data_temp)
        loss = criterion(output, target_temp)
        loss.backward()
        optimizer_temp.step()
    for l in range(param_num):
        global_direction[l] = optimizer_temp.param_groups[0]['params'][l].detach() - optimizer.param_groups[0]['params'][l].detach()
    # factor = 1
    # for l in range(param_num):
    #     grad_avg[l] = global_direction[l] * factor
    selection = np.random.choice(range(M), S, replace=False)
    attacks = list(np.random.choice(selection, A, replace=False))
    # selection = list(range(M))
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum)        
        if m in attacks:
            loaders = loader_workers[m]
        else:
            loaders = loader_workers[m]
        for batch_idx, (data, target) in enumerate(loaders):
            optimizer_worker[m].zero_grad()
            # output = network_worker(data)
            output = network_worker(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_worker[m].step()
            if batch_idx == U - 1:
                grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                break
        
    for m in selection:
        for l in range(param_num):
            grad_workers[m][l] = grad_workers[m][l].detach_() - optimizer.param_groups[0]['params'][l].detach()
        if m in attacks:
            for l in range(param_num):
                grad_workers[m][l] = np.random.normal(0,3,size = (1,1))[0][0] * grad_workers[m][l]
    # temp = [0]*4
    # for l in range(10):
    #     for m in selection:
    #         temp[l] += grad_workers[m][l]
    #     temp[l] /= S

    for l in range(param_num):
        for m in selection:
            ll = global_direction[l]
            lamb = 1 - torch.dot(grad_workers[m][l].view(1,-1)[0], ll.view(1,-1)[0])/ll.norm()/grad_workers[m][l].norm()
            # lamb = lamb * 0.01
            # print(lamb)
            lamb = lamb * 0.1
            
            # lamb = 1
            grad_workers[m][l] = grad_workers[m][l]/grad_workers[m][l].norm()*ll.norm()
                # ll = temp[l]
            # ll = ll/ll.norm() * grad_workers[m][l].norm()
            lamb = relu(lamb)
            lamm = 1-lamb
            grad_workers[m][l] = lamm*grad_workers[m][l] + (1-lamm)*ll
            # grad_workers[m][l] = ll
            # grad_workers[m][l] = (1-lamb)*grad_workers[m][l] + (lamb)*ll
            
    temp = [0]*param_num
    for l in range(param_num):
        for m in selection:
            temp[l] += grad_workers[m][l]
        temp[l] /= S
        # grad_avg[l] = (grad_avg[l] * (i) + temp[l]) / (i+1)
    optimizer.zero_grad()
    for l in range(param_num):
        optimizer.param_groups[0]['params'][l].grad = -temp[l]
        
    for batch_idx, (data, target) in enumerate(test_loader):
        output = network(data)
        optimizer.step()
        # output = network(data)
        loss = criterion(output, target)
        break
    print(loss.item(), "ACAG", i)
    train_losses_ACSA.append(loss.item())
    
    
    if test_losses_ACSA[-1] >= acc:
        break
#%% FLTrust


train_losses_FLTrust = []
test_losses_FLTrust = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
# grad_avg = {}
global_direction = {}
for l in range(param_num):
    global_direction[l] = 0
for m in range(M):
    optimizer_worker.append(0)
for i in range(iters):
    if i % epochs == 0:
        network.eval()
        test_loss_FLTrust = 0
        correct_FLTrust = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data)
            # output = network(data)
            test_loss_FLTrust += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_FLTrust += pred.eq(target.data.view_as(pred)).sum()
        test_loss_FLTrust /= len(test_loader.dataset)
        test_losses_FLTrust.append(100. * correct_FLTrust / len(test_loader.dataset))
    # optimizer.param_groups[0]['lr'] *= 0.99
    network_temp = copy.deepcopy(network)
    optimizer_temp = optim.SGD(network_temp.parameters(), lr=0.1, momentum=momentum)
    network_temp.train()
    for batch_idx, (data, target) in enumerate(test_loader_clear):
        # output = network_temp(data)
        # loss = criterion(output, target)
        # loss.backward()
        # optimizer_temp.step()
        data_temp = data
        target_temp = target
        break
    for iter in range(U):
        output = network_temp(data_temp)
        loss = criterion(output, target_temp)
        loss.backward()
        optimizer_temp.step()
    for l in range(param_num):
        global_direction[l] = optimizer_temp.param_groups[0]['params'][l].detach() - optimizer.param_groups[0]['params'][l].detach()
    # factor = 1
    # for l in range(param_num):
    #     grad_avg[l] = global_direction[l] * factor
    selection = np.random.choice(range(M), S, replace=False)
    attacks = list(np.random.choice(selection, A, replace=False))
    # selection = list(range(M))
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum)  
        if m in attacks:
            loaders = loader_workers[m]
        else:
            loaders = loader_workers[m]
        for batch_idx, (data, target) in enumerate(loaders):
            optimizer_worker[m].zero_grad()
            # output = network_worker(data)
            output = network_worker(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_worker[m].step()
            if batch_idx == U - 1:
                grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                break
        
    for m in selection:
        for l in range(param_num):
            grad_workers[m][l] = grad_workers[m][l].detach_() - optimizer.param_groups[0]['params'][l].detach()
        if m in attacks:
            for l in range(param_num):
                grad_workers[m][l] = np.random.normal(0, 3, size = (1,1))[0][0] * grad_workers[m][l]
    # temp = [0]*4
    # for l in range(10):
    #     for m in selection:
    #         temp[l] += grad_workers[m][l]
    #     temp[l] /= S
    ll = [0]*param_num
    lamb = [0]*M
    for l in range(param_num):
        ll[l] = (global_direction[l].norm())**2
    ll_norm = sqrt(sum(ll))
    good = 0
    for m in selection:
        grad_concat = [0]*param_num
        for l in range(param_num):
            grad_concat[l] = (grad_workers[m][l].norm())**2
        grad_norm = sqrt(sum(grad_concat))
        ip = 0
        for l in range(param_num):
            ip += torch.dot(grad_workers[m][l].view(1,-1)[0], global_direction[l].view(1,-1)[0])
        lamb[m] = relu(ip)/ll_norm/grad_norm
        if ip >= 0:
            good += 1
        
        # lamb = lamb * 0.01
        # print(lamb)
        # lamb = lamb * 0.3
        # lamb = 1
        for l in range(param_num):
            grad_workers[m][l] = grad_workers[m][l]/grad_norm*ll_norm
        # grad_workers[m][l] = ll
        # grad_workers[m][l] = (1-lamb)*grad_workers[m][l] + (lamb)*ll
    print(good)
    temp = [0]*param_num 
    for l in range(param_num):
        for m in selection:
            temp[l] += lamb[m]*grad_workers[m][l]
            # temp[l] += grad_workers[m][l]
        temp[l] /= sum(lamb)
        # temp[l] /= len(selection)
        # grad_avg[l] = (grad_avg[l] * (i) + temp[l]) / (i+1)
    optimizer.zero_grad()
    for l in range(param_num):
        optimizer.param_groups[0]['params'][l].grad = -temp[l]
        
    for batch_idx, (data, target) in enumerate(test_loader):
        output = network(data)
        optimizer.step()
        # output = network(data)
        loss = criterion(output, target)
        break
    print(loss.item(), "FLTrust", i)
    train_losses_FLTrust.append(loss.item())
    
    
    if test_losses_FLTrust[-1] >= acc:
        break


#%% DSGD
train_losses_DSGD = []
test_losses_DSGD = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
for m in range(M):
    optimizer_worker.append(0)
for i in range(iters):
    if i % 10 == 0:
        network.eval()
        test_loss_DSGD = 0
        correct_DSGD = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data)
            # output = network(data)
            test_loss_DSGD += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_DSGD += pred.eq(target.data.view_as(pred)).sum()
        test_loss_DSGD /= len(test_loader.dataset)
        test_losses_DSGD.append(100. * correct_DSGD / len(test_loader.dataset))
    selection = np.random.choice(range(M), S, replace=False)
    attacks = list(np.random.choice(selection, A, replace=False))
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum)        
        if m in attacks:
            loaders = loader_workers[m]
        else:
            loaders = loader_workers[m]
        for batch_idx, (data, target) in enumerate(loaders):
            optimizer_worker[m].zero_grad()
            # output = network_worker(data)
            output = network_worker(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_worker[m].step()
            if batch_idx == U - 1:
                grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                break
    for m in selection:
        for l in range(param_num):
            grad_workers[m][l] = grad_workers[m][l].detach_() - optimizer.param_groups[0]['params'][l].detach()
        if m in attacks:
            for l in range(param_num):
                grad_workers[m][l] = np.random.normal(0,3,size = (1,1))[0][0] * grad_workers[m][l]
    temp = [0]*param_num
    for l in range(param_num):
        for m in selection:
            temp[l] += grad_workers[m][l]
        temp[l] /= S
        # grad_avg[l] = (grad_avg[l] * (i) + temp[l]) / (i+1)
    optimizer.zero_grad()
    for l in range(param_num):
        optimizer.param_groups[0]['params'][l].grad = -temp[l]
    
    for batch_idx, (data, target) in enumerate(test_loader):
        output = network(data)
        optimizer.step()
        # output = network(data)
        loss = criterion(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "DSGD", i)
    train_losses_DSGD.append(loss.item())  
    if test_losses_DSGD[-1] >= acc:
        break

#%%
figure()
plot(np.arange(0, len(test_losses_ACSA)*epochs, epochs), test_losses_ACSA)
plot(np.arange(0, len(test_losses_FLTrust)*epochs, epochs), test_losses_FLTrust)
plot(np.arange(0, len(test_losses_DSGD)*epochs, epochs), test_losses_DSGD)
xlabel('training rounds')
ylabel('test accuracy')
legend(['ACAG','FLTrust','FedAvg'])
#plot(train_losses_AMS)
savefig('ACAG_byzantine_noniid.pdf')
# xlim([0, 30000])
grid('on')
