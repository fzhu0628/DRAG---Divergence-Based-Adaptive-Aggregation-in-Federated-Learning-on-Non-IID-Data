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
M = 40
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


    dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[index], np.array(dataset_workers[i].targets)[index].tolist()
    # dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[index], dataset_workers[i].targets[index]
    dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[int(length / M * i) : int(length / M * (i + 1))]\
    , dataset_workers[i].targets[int(length / M * i) : int(length / M * (i + 1))]

    sampler_workers.append(sampler.BatchSampler(sampler.RandomSampler(data_source=dataset_workers[i], replacement=True), batch_size=batch_size_train, drop_last=False))
    loader_workers.append(torch.utils.data.DataLoader(dataset_workers[i],batch_sampler=sampler_workers[i], shuffle=False))


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
        x = F.softmax(self.fc3(x))
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.n_cls = 10
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64*5*5, 384) 
#         self.fc2 = nn.Linear(384, self.n_cls) 

#     def forward(self,x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64*5*5)
#         x = F.relu(self.fc1(x))
#         x = (self.fc2(x))
#         return x
criterion = nn.CrossEntropyLoss()
network1 = Net()
S = 10
U = 10
epochs = 10
iters = 12000
acc = 70
param_num = 10
param_num_ACAG = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network1 = Net().to(device)

#%% FedLin
train_losses_FedLin = []
test_losses_FedLin = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
g_global = {}
for m in range(M):
    g_global[m] = [0]*param_num
for m in range(M):
    optimizer_worker.append(0)
for i in range(iters):
    if i % 10 == 0:
        network.eval()
        test_loss_FedLin = 0
        correct_FedLin = 0
        with torch.no_grad():
          for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            # output = network(data)
            test_loss_FedLin += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_FedLin += pred.eq(target.data.view_as(pred)).sum()
        test_loss_FedLin /= len(test_loader.dataset)
        test_losses_FedLin.append(100. * correct_FedLin / len(test_loader.dataset))
    selection = np.random.choice(range(M), S, replace=False)
    grad_workers= {}
    
    if i == 0:
        for m in selection:
            network_worker1 = copy.deepcopy(network)
            network_worker1.train()
            optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
            # local_rounds = int(np.max(time_workers) / time_workers[flag])
            for aa in range(10000):
                for batch_idx, (data, target) in enumerate(loader_workers[m]):
                    data, target = data.to(device), target.to(device)
                    optimizer_worker[m].zero_grad()
                    output = network_worker(data)
                    # output = network_worker(data)
                    loss = criterion(output, target)
                    loss.backward()
                    if batch_idx == 0:
                        for l in range(param_num):
                            g_global[m][l] = optimizer_worker[m].param_groups[0]['params'][l].grad
                    if batch_idx == 0:
                        break
                if batch_idx == 0:
                    break
    
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                data, target = data.to(device), target.to(device)
                optimizer_worker[m].zero_grad()
                output = network_worker(data)
                # output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                if batch_idx == 0:
                    g_local = [0]*param_num
                    for l in range(param_num):
                        g_local[l] = optimizer_worker[m].param_groups[0]['params'][l].grad
                for l in range(param_num):
                    optimizer_worker[m].param_groups[0]['params'][l].grad = optimizer_worker.param_groups[0]['params'][l].grad - g_local[l] + g_global[l] 
                optimizer_worker[m].step()
                if batch_idx == U - 1:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params'] 
                    for l in range(param_num):
                        g_global[m][l] = optimizer_worker[m].param_groups[0]['params'][l].grad
                    break
            if batch_idx == U - 1:
                break
    g_global_final = [0]*param_num
    for l in range(param_num):
        for m in selection:
            g_global_final[l] += g_global[m][l]
        g_global_final[l] /= len(selection)
        
    temp = [0]*param_num
    for l in range(param_num):
        count = 0
        for m in selection:
            # temp[l] += (grad_workers[m][l]- optimizer.param_groups[0]['params'][l])*p[m]/tt
            temp[l] += (grad_workers[m][l]- optimizer.param_groups[0]['params'][l])
            count += 1
        temp[l] /= count
    
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        for l in range(param_num):
            optimizer.param_groups[0]['params'][l].grad = -temp[l] 
        optimizer.step()
        output = network(data)
        # output = network(data)
        loss = criterion(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "FedLin", i)
    train_losses_FedLin.append(loss.item())  
    if test_losses_FedLin[-1] >= acc:
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
            data, target = data.to(device), target.to(device)
            output = network(data)
            # output = network(data)
            test_loss_DSGD += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_DSGD += pred.eq(target.data.view_as(pred)).sum()
        test_loss_DSGD /= len(test_loader.dataset)
        test_losses_DSGD.append(100. * correct_DSGD / len(test_loader.dataset))
    selection = np.random.choice(range(M), S, replace=False)
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                data, target = data.to(device), target.to(device)
                optimizer_worker[m].zero_grad()
                output = network_worker(data)
                # output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if batch_idx == U - 1:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params'] 
                    break
            if batch_idx == U - 1:
                break
    temp = [0]*param_num
    for l in range(param_num):
        count = 0
        for m in selection:
            # temp[l] += (grad_workers[m][l]- optimizer.param_groups[0]['params'][l])*p[m]/tt
            temp[l] += (grad_workers[m][l]- optimizer.param_groups[0]['params'][l])
            count += 1
        temp[l] /= count
    
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        for l in range(param_num):
            optimizer.param_groups[0]['params'][l].grad = -temp[l] 
        optimizer.step()
        output = network(data)
        # output = network(data)
        loss = criterion(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "DSGD", i)
    train_losses_DSGD.append(loss.item())  
    if test_losses_DSGD[-1] >= acc:
        break

#%% SCAFFOLD
c, c_plus, c_delta, workers_delta = {}, {}, {}, {}
for i in range(M):
    c[i] = [0]*param_num
    c_plus[i] = [0]*param_num
    c_delta[i] = [0]*param_num
    workers_delta[i] = [0]*param_num
c_global = [0]*param_num
train_losses_SCAFFOLD = []
test_losses_SCAFFOLD = []
network = copy.deepcopy(network1)
#optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.1, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
eta_l = 0.1
for i in range(iters):
    if i % epochs == 0:
        network.eval()
        test_loss_SCAFFOLD = 0
        correct_SCAFFOLD = 0
        with torch.no_grad():
          for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # data, target = data.cuda(), target.cuda()
            output = network(data)
            test_loss_SCAFFOLD += F.nll_loss(output, target, size_average=True).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_SCAFFOLD += pred.eq(target.data.view_as(pred)).sum()
        test_loss_SCAFFOLD /= len(test_loader.dataset)
        test_losses_SCAFFOLD.append(100. * correct_SCAFFOLD / len(test_loader.dataset))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # data, target = data.cuda(), target.cuda()
            output = network(data)
            loss = criterion(output, target)
            print(loss.item(), 'SCAFFOLD',i)
            train_losses_SCAFFOLD.append(loss.item())
            break
    selection = list(np.random.choice(range(M), S, replace=False))
    # attacks = list(np.random.choice(selection, A, replace=False))
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        # network_worker = network
        optimizer_worker = optim.SGD(network_worker.parameters(), lr=eta_l, momentum=momentum)
        for batch_idx, (data, target) in enumerate(loader_workers[m]):
            data, target = data.to(device), target.to(device)
            optimizer_worker.zero_grad()
            output = network_worker(data)
            loss = criterion(output, target)
            loss.backward()
            for l in range(param_num):
                optimizer_worker.param_groups[0]['params'][l].grad = optimizer_worker.param_groups[0]['params'][l].grad - c[m][l] + c_global[l]
            optimizer_worker.step()
            if batch_idx == U-1:
                grad_workers[m] = optimizer_worker.param_groups[0]['params']
                break
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            network_temp = copy.deepcopy(network)
            optim_temp = optim.SGD(network_temp.parameters(), lr=eta_l, momentum=momentum)
            output_temp = network_temp(data)
            loss_temp = criterion(output_temp, target)
            loss_temp.backward()
            break
        # for l in range(4):
        #     # c_plus[m][l] = copy.deepcopy(optim_temp.param_groups[0]['params'][l].grad)
        #     c_plus[m][l] = c[m][l] - c_global[l] + (1/((U+1)*eta_l))*(optimizer.param_groups[0]['params'][l] - grad_workers[m][l])
        # for l in range(4):
            # c_delta[m][l] = c[m][l] - c_global[l] + (1/((U+1)*eta_l))*(optimizer.param_groups[0]['params'][l] - grad_workers[m][l]) - c[m][l]
        # for l in range(4):
        #     workers_delta[m][l] = grad_workers[m][l] - optimizer.param_groups[0]['params'][l]
        for l in range(param_num):
            c_plus[m][l] = copy.deepcopy(optim_temp.param_groups[0]['params'][l].grad).detach()
            # c_plus[m][l] = c[m][l] - c_global[l] + (1/((U)*eta_l))*(optimizer.param_groups[0]['params'][l].detach() - grad_workers[m][l].detach())
        for l in range(param_num):
            c_delta[m][l] = c_plus[m][l] - c[m][l]
        for l in range(param_num):
            workers_delta[m][l] = grad_workers[m][l].detach() - optimizer.param_groups[0]['params'][l].detach()
            workers_delta[m][l].detach_()
        for l in range(param_num):
            c[m][l] = c_plus[m][l]
    c_global_delta = [0]*param_num
    for l in range(param_num):
        for m in selection:
            c_global_delta[l] += c_delta[m][l]
        c_global_delta[l] /= S
    for l in range(param_num):
        c_global[l] += c_global_delta[l] * S / M
        
    temp = [0]*param_num
    for l in range(param_num):
        for m in selection:
            temp[l] += workers_delta[m][l]
            # temp[l] += grad_workers[m][l].grad
        temp[l] /= S
    optimizer.zero_grad()
    for l in range(param_num):
        optimizer.param_groups[0]['params'][l].grad = -temp[l]
        # optimizer.param_groups[0]['params'][l].grad = temp[l]
    optimizer.step()
    del grad_workers
    
    
    if test_losses_SCAFFOLD[-1] >= acc:
        break





#%% AdaBest
c, c_plus, c_delta, workers_delta = {}, {}, {}, {}
mu = 0.02
beta = 0.2
age = [1] * M
for i in range(M):
    c[i] = [0]*param_num
    workers_delta[i] = [0]*param_num
c_global = [0]*param_num
train_losses_AdaBest = []
test_losses_AdaBest = []
network = copy.deepcopy(network1)
#optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.1, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
eta_l = 0.1
for i in range(iters):
    if i % epochs == 0:
        network.eval()
        test_loss_AdaBest = 0
        correct_AdaBest = 0
        with torch.no_grad():
          for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # data, target = data.cuda(), target.cuda()
            output = network(data)
            test_loss_AdaBest += F.nll_loss(output, target, size_average=True).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_AdaBest += pred.eq(target.data.view_as(pred)).sum()
        test_loss_AdaBest /= len(test_loader.dataset)
        test_losses_AdaBest.append(100. * correct_AdaBest / len(test_loader.dataset))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # data, target = data.cuda(), target.cuda()
            output = network(data)
            loss = criterion(output, target)
            print(loss.item(), 'AdaBest',i)
            train_losses_AdaBest.append(loss.item())
            break
    selection = list(np.random.choice(range(M), S, replace=False))
    # attacks = list(np.random.choice(selection, A, replace=False))
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        # network_worker = network
        optimizer_worker = optim.SGD(network_worker.parameters(), lr=eta_l, momentum=momentum)
        for batch_idx, (data, target) in enumerate(loader_workers[m]):
            data, target = data.to(device), target.to(device)
            optimizer_worker.zero_grad()
            output = network_worker(data)
            loss = criterion(output, target)
            loss.backward()
            for l in range(param_num):
                optimizer_worker.param_groups[0]['params'][l].grad = optimizer_worker.param_groups[0]['params'][l].grad - c[m][l]
            optimizer_worker.step()
            if batch_idx == U-1:
                grad_workers[m] = optimizer_worker.param_groups[0]['params']
                break
        # if m in attacks:
        #     for l in range(param_num):
        #         grad_workers[m][l] = -0.1*grad_workers[m][l]

        for l in range(param_num):
            # c_plus[m][l] = copy.deepcopy(optim_temp.param_groups[0]['params'][l].grad)
            c[m][l] = 1/age[m]*c[m][l] + mu*(optimizer.param_groups[0]['params'][l].detach() - grad_workers[m][l].detach())
        
    theta_bar = [0]*param_num
    for l in range(param_num):
        for m in selection:
            theta_bar[l] += grad_workers[m][l].detach()
        theta_bar[l] /= S
    
    for l in range(param_num):
        if i == 0:
            c_global[l] = -beta*theta_bar[l].detach()
        else:
            c_global[l] = beta * (theta_old[l].detach() - theta_bar[l].detach())

        
    for m in range(M):
        if m in selection:
            age[m] = 1
        else:
            age[m] += 1
    temp = [0]*param_num
    for l in range(param_num):
        temp[l] = theta_bar[l] - c_global[l]
            # temp[l] += grad_workers[m][l].grad
    theta_old = copy.deepcopy(theta_bar)
    optimizer.zero_grad()
    
    for l in range(param_num):
        optimizer.param_groups[0]['params'][l].grad = -(temp[l]-optimizer.param_groups[0]['params'][l].detach())
        # optimizer.param_groups[0]['params'][l].grad = temp[l]
    optimizer.step()
    del grad_workers
    # gc.collect()
    
    if test_losses_AdaBest[-1] >= acc:
        break
#%% FedProx
# train_losses_FedProx = []
# test_losses_FedProx = []
# network = copy.deepcopy(network1)
# # optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
# optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
# network.train()
# mu = 0.2
# gamma = 0.2
# optimizer_worker = []
# for m in range(M):
#     optimizer_worker.append(0)
# for i in range(iters):
#     if i % epochs == 0:
#         network.eval()
#         test_loss_FedProx = 0
#         correct_FedProx = 0
#         with torch.no_grad():
#           for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = network(data)
#             # output = network(data)
#             test_loss_FedProx += F.nll_loss(output, target, size_average=False).item()
#             pred = output.data.max(1, keepdim=True)[1]
#             correct_FedProx += pred.eq(target.data.view_as(pred)).sum()
#         test_loss_FedProx /= len(test_loader.dataset)
#         test_losses_FedProx.append(100. * correct_FedProx / len(test_loader.dataset))
#     for batch_idx, (data, target) in enumerate(test_loader):
#         data, target = data.to(device), target.to(device)
#         output = network(data)
#         # output = network(data)
#         loss = criterion(output, target)
#         break
#     # optimizer.param_groups[0]['lr'] *= 0.99
#     print(loss.item(), "FedProx", i)
#     train_losses_FedProx.append(loss.item())
#     selection = np.random.choice(range(M), S, replace=False)
#     # attacks = list(np.random.choice(selection, A, replace=False))
#     # selection = list(range(M))
#     for batch_idx, (data, target) in enumerate(test_loader):
#         data, target = data.to(device), target.to(device)
#         network_temp = copy.deepcopy(network)
#         optim_temp = optim.SGD(network_temp.parameters(), lr=0.1, momentum=momentum)
#         output_temp = network_temp(data)
#         loss_temp = criterion(output_temp, target)
#         loss_temp.backward()
#         break
#     norm_temp = torch.tensor([])
#     for l in range(param_num):
#         norm_temp = torch.cat((norm_temp, optim_temp.param_groups[0]['params'][l].grad.detach().reshape(-1).cpu()))
#     norm_temp = torch.norm(norm_temp)
#     # diff1 = optim_temp.param_groups[0]['params']
#     # norm_temp = np.linalg.norm(np.concatenate((diff1[0].grad.detach().numpy().reshape(-1),diff1[1].grad.detach().numpy().reshape(-1),diff1[2].grad.detach().numpy().reshape(-1),diff1[3].grad.detach().numpy().reshape(-1),diff1[4].grad.detach().numpy().reshape(-1),diff1[5].grad.detach().numpy().reshape(-1),diff1[6].grad.detach().numpy().reshape(-1),diff1[7].grad.detach().numpy().reshape(-1),diff1[8].grad.detach().numpy().reshape(-1),diff1[9].grad.detach().numpy().reshape(-1))))
#     # print("norm_temp:", norm_temp)
#     grad_workers= {}
#     for m in selection:
#         network_worker = copy.deepcopy(network)
#         network_worker.train()
#         optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum)
#         for aa in range(10000):
#             for batch_idx, (data, target) in enumerate(loader_workers[m]):
#                 data, target = data.to(device), target.to(device)
#                 optimizer_worker[m].zero_grad()
#                 # output = network_worker(data)
#                 output = network_worker(data)
#                 loss = criterion(output, target)
#                 loss.backward()
#                 for l in range(param_num):
#                     optimizer_worker[m].param_groups[0]['params'][l].grad = optimizer_worker[m].param_groups[0]['params'][l].grad + mu*(optimizer_worker[m].param_groups[0]['params'][l].detach()-optimizer.param_groups[0]['params'][l].detach())
#                 optimizer_worker[m].step()
#                 norm_star = torch.tensor([])
#                 for l in range(param_num):
#                     norm_star = np.concatenate((norm_star, optimizer_worker[m].param_groups[0]['params'][l].grad.detach().cpu().reshape(-1)))
#                 norm_star = np.linalg.norm(norm_star)
#                 # diff = optimizer_worker[m].param_groups[0]['params']
#                 # norm_star = np.linalg.norm(np.concatenate((diff[0].grad.detach().numpy().reshape(-1),diff[1].grad.detach().numpy().reshape(-1),diff[2].grad.detach().numpy().reshape(-1),diff[3].grad.detach().numpy().reshape(-1),diff[4].grad.detach().numpy().reshape(-1),diff[5].grad.detach().numpy().reshape(-1),diff[6].grad.detach().numpy().reshape(-1),diff[7].grad.detach().numpy().reshape(-1),diff[8].grad.detach().numpy().reshape(-1),diff[9].grad.detach().numpy().reshape(-1))))
#                 # print("norm_star:", norm_star)
#                 if norm_star <= gamma*norm_temp:
#                 # if batch_idx==U-1:
#                     grad_workers[m]=optimizer_worker[m].param_groups[0]['params'] 
#                     break
#             if norm_star <= gamma*norm_temp:
#             # if batch_idx==U-1:
#                 break
#     # del optim_temp, norm_star, norm_temp
#     # gc.collect()
#         # if m in attacks:
#         #     for l in range(10):
#         #         grad_workers[m][l] = -grad_workers[m][l]
#     temp = [0]*param_num
#     for l in range(param_num):
#         for m in selection:
#             temp[l] += grad_workers[m][l] - optimizer.param_groups[0]['params'][l]
#         temp[l] /= S
        
#     optimizer.zero_grad()
#     for l in range(param_num):
#         optimizer.param_groups[0]['params'][l].grad = -temp[l] 
#     optimizer.step()   

#     if test_losses_FedProx[-1] >= acc:
#         break
#%% test
# sampler=(sampler.BatchSampler(sampler.RandomSampler(data_source=dataset_global, replacement=True), batch_size=batch_size_train, drop_last=False))
# loader=(torch.utils.data.DataLoader(dataset_global,batch_sampler=sampler, shuffle=False))
# train_losses_test = []
# test_losses_test = []
# network = copy.deepcopy(network1)
# # optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
# optimizer = optim.SGD(network.parameters(), lr=0.05, momentum=momentum)
# network.train()
# optimizer_worker = []
# for m in range(M):
#     optimizer_worker.append(0)
# for i in range(iters):
#     if i % epochs == 0:
#         network.eval()
#         test_loss_test = 0
#         correct_test = 0
#         with torch.no_grad():
#           for data, target in test_loader:
#             output = network(data)
#             # output = network(data)
#             test_loss_test += criterion(output, target, size_average=False).item()
#             pred = output.data.max(1, keepdim=True)[1]
#             correct_test += pred.eq(target.data.view_as(pred)).sum()
#         test_loss_test /= len(test_loader.dataset)
#         test_losses_test.append(100. * correct_test / len(test_loader.dataset))
#     for batch_idx, (data, target) in enumerate(test_loader):
#         output = network(data)
#         # output = network(data)
#         loss = criterion(output, target)
#         break
#     # optimizer.param_groups[0]['lr'] *= 0.99
#     print(loss.item(), "test", i)
#     train_losses_test.append(loss.item())
       
#     for batch_idx, (data, target) in enumerate(loader):
#         optimizer.zero_grad()
#         # output = network_worker(data)
#         output = network(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
        
    

    
#     if test_losses_test[-1] >= acc:
#         break
#%% ACSA
alpha = 0.5
train_losses_ACSA = []
test_losses_ACSA = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
grad_avg = {}
for l in range(param_num_ACAG):
    grad_avg[l] = 0
for m in range(M):
    optimizer_worker.append(0)
for i in range(iters):
    if i % epochs == 0:
        network.eval()
        test_loss_ACSA = 0
        correct_ACSA = 0
        with torch.no_grad():
          for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            # output = network(data)
            test_loss_ACSA += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ACSA += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ACSA /= len(test_loader.dataset)
        test_losses_ACSA.append(100. * correct_ACSA / len(test_loader.dataset))
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = network(data)
        optimizer.step()
        # output = network(data)
        loss = criterion(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "ACAG", i)
    train_losses_ACSA.append(loss.item())
    selection = np.random.choice(range(M), S, replace=False)
    # attacks = list(np.random.choice(selection, A, replace=False))
    # selection = list(range(M))
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum)        
        for batch_idx, (data, target) in enumerate(loader_workers[m]):
            data, target = data.to(device), target.to(device)
            optimizer_worker[m].zero_grad()
            # output = network_worker(data)
            output = network_worker(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_worker[m].step()
            if batch_idx == U - 1:
                grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                break
        # if m in attacks:
        #     for l in range(10):
        #         grad_workers[m][l] = 0 * grad_workers[m][l]
    for m in selection:
        for l in range(param_num_ACAG):
            grad_workers[m][l] = grad_workers[m][l].detach_() - optimizer.param_groups[0]['params'][l].detach()
    # temp = [0]*4
    # for l in range(10):
    #     for m in selection:
    #         temp[l] += grad_workers[m][l]
    #     temp[l] /= S

    



    temp = [0]*param_num_ACAG
    for l in range(param_num_ACAG):
        if i == 0:
            for m in selection:
                temp[l] += grad_workers[m][l]
            temp[l] /= S
        else:
            temp[l] = grad_avg[l]
    ll = torch.cat((temp[0].view(1,-1)[0],temp[1].view(1,-1)[0]))
    for l in range(param_num_ACAG-2):
        ll = torch.cat((ll.view(1,-1)[0], temp[l+2].view(1,-1)[0]))
    for m in selection:
        ww = torch.cat((grad_workers[m][0].view(1,-1)[0], grad_workers[m][1].view(1,-1)[0]))
        for l in range(param_num_ACAG-2):
            ww = torch.cat((ww.view(1,-1)[0], grad_workers[m][l+2].view(1,-1)[0]))
        lamb = 1 - torch.dot(ww.view(1,-1)[0], ll.view(1,-1)[0])/ll.norm()/ww.norm()
        lamb = lamb * 0.1
        for l in range(param_num_ACAG):
            temp[l] = temp[l]/temp[l].norm() * grad_workers[m][l].norm()
            grad_workers[m][l] = (1-lamb)*grad_workers[m][l] + (lamb)*temp[l]
    
    
                
                
                
                   
                
            # lamb = 1 - torch.dot(grad_workers[m][l].view(1,-1)[0], ll.view(1,-1)[0])/ll.norm()/grad_workers[m][l].norm()
            # lamb = lamb * 0.1
            # # lamb = lamb * 0.3
            
            #     # ll = temp[l]
            # ll = ll/ll.norm() * grad_workers[m][l].norm()
            # grad_workers[m][l] = (1-lamb)*grad_workers[m][l] + (lamb)*ll
            
    temp = [0]*param_num_ACAG
    for l in range(param_num_ACAG):
        for m in selection:
            temp[l] += grad_workers[m][l]
        temp[l] /= S
    for l in range(param_num_ACAG):
        grad_avg[l] = grad_avg[l] * (1-alpha) + temp[l] * alpha
        # grad_avg[l] = (grad_avg[l] * (i) + temp[l]) / (i+1)
    optimizer.zero_grad()
    for l in range(param_num_ACAG):
        optimizer.param_groups[0]['params'][l].grad = -temp[l]
        
    

    
    
    if test_losses_ACSA[-1] >= acc:
        break


#%%
for a in range(len(test_losses_ACSA)):
    test_losses_ACSA[a] = np.array(test_losses_ACSA[a].cpu())

for a in range(len(test_losses_AdaBest)):
    test_losses_AdaBest[a] = np.array(test_losses_AdaBest[a].cpu())

for a in range(len(test_losses_SCAFFOLD)):
    test_losses_SCAFFOLD[a] = np.array(test_losses_SCAFFOLD[a].cpu())

for a in range(len(test_losses_DSGD)):
    test_losses_DSGD[a] = np.array(test_losses_DSGD[a].cpu())

for a in range(len(test_losses_FedLin)):
    test_losses_FedLin[a] = np.array(test_losses_FedLin[a].cpu())
#%%
figure()
plot(np.arange(0, len(test_losses_AdaBest)*epochs, epochs), test_losses_AdaBest,linestyle='-',color='green')
plot(np.arange(0, len(test_losses_SCAFFOLD)*epochs, epochs), test_losses_SCAFFOLD,linestyle='-.',color='blue')
plot(np.arange(0, len(test_losses_DSGD)*epochs, epochs), test_losses_DSGD,linestyle=':',color='orange')
# plot(np.arange(0, len(test_losses_FedProx)*epochs, epochs), test_losses_FedProx,linestyle='--',color='black')
plot(np.arange(0, len(test_losses_ACSA)*epochs, epochs), test_losses_ACSA,linestyle='-',color='red')
xlabel('training rounds')
ylabel('test accuracy')
legend(['AdaBest','SCAFFOLD','FedAvg','ACAG'])
#plot(train_losses_AMS)
grid('on')
# xlim([0, 3000])

savefig('ACAG_PART.pdf')
# 



