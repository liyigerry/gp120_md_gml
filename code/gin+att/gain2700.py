import sys
sys.path.append('/home/dldx/UniRep/Newtry')
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from my_dataloader import GraphDataLoader, collate
from gain_quick import GIN
from tqdm import tqdm
import random
import os
import time
import scipy.sparse as sp
import pickle as pkl
import dgl
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels,names = map(list, zip(*samples))
     
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels),names

def train( net, trainloader, optimizer, criterion, epoch):
    net.train()
    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor

    for graphs, labels,names in (trainloader):
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(device)
        #feat = graphs.ndata['attr'].to(device)
        feat = graphs.ndata['feat'].float().to(device)
        output = net(graphs, feat)
        loss = criterion(output[0], labels)
        running_loss += loss.item()
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss 

@torch.no_grad()
def eval_net( net, dataloader, criterion):
    net.eval()
    total = 0
    total_correct = 0
    targets,outputs=[],[]
    for data in dataloader:
        graphs, labels,names = data
        feat = graphs.ndata['feat'].float().to(device)
        labels = labels.to(device)
        output = net(graphs, feat)
        _, predicted = torch.max(output[0].data, 1)
        total += len(labels)
        total_correct += (predicted == labels.data).sum().item()
        targets.append(labels)
        outputs.append(output[0])
    acc = 1.0*total_correct / total

    return  acc,targets,outputs

# set up seeds, args.seed supported
seed=2021
torch.manual_seed(seed=seed)
np.random.seed(seed=seed)

#指定GPU
torch.cuda.set_device(3)
if torch.cuda.is_available():

    device = torch.device("cuda")
    torch.cuda.manual_seed_all(seed=seed)
else:
    device = torch.device("cpu")
print(device)
    
with open("/home/dldx/UniRep/data/5fyj_45_2700.p", 'rb') as f:                      
    dataset = pkl.load(f)
print(len(dataset))

def split_rand(data, split_ratio=0.7, seed=0, shuffle=True):
    data_valid,data_Train=[],[]
    num_entries = len(data)
    indices = list(range(num_entries))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = int(math.floor(split_ratio * num_entries))
    train_idx, valid_idx = indices[:split], indices[split:]
    for i in valid_idx:
        data_valid.append(dataset[i])
    for i in train_idx:
        data_Train.append(dataset[i])
    print(
            "train_set : valid_set = %d : %d",
            len(train_idx), len(valid_idx))

    return data_Train, data_valid

b=[]
for i in range(270):
    b.append(10*i)
data_test,data_train=[],[]
for i in range(2700):
    if i in b:
        data_test.append(dataset[i])
    else:
        data_train.append(dataset[i])
data_Train, data_valid=split_rand(data_train, split_ratio=0.8, seed=2021, shuffle=True)

with open("/home/dldx/UniRep/data/data_Test_of_2700.p", 'wb') as f:                      
    pkl.dump(data_test,f)
with open("/home/dldx/UniRep/data/data_Train_of_2700.p", 'wb') as f:                      
    pkl.dump(data_Train,f)
with open("/home/dldx/UniRep/data/data_Valid_of_2700.p", 'wb') as f:                      
    pkl.dump(data_valid,f)
    
#模型参数
start = time.time()
acc_scores=[]
datalist=[]
for fold_idx in range(1):
    model =GIN(5, 2, 17, 64, 3, 0.5, True, "sum", "sum").to(device)

    epochs=200
    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_loader = DataLoader(data_Train, batch_size=32, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(data_valid, batch_size=32, shuffle=True, collate_fn=collate)
    
    train_loss=[]
    valid_accs=[]
    valid_acc_tmp=0
    flag=False
    count=0
    for epoch in range(epochs):
        print('正在训练{}轮...'.format(epoch))
        loss=train( model, train_loader, optimizer, criterion, epoch)
        train_loss.append(loss)
        scheduler.step()

        #---------------------------动态输出训练结果-----------------------------     
        valid_acc,target,output = eval_net(model, valid_loader, criterion)
        valid_accs.append(valid_acc)
        targets=target
        outputs=output
        print('valid acc:',valid_acc)
        if valid_acc > 0.85 and valid_acc>valid_acc_tmp:
            count+=1
            valid_acc_tmp=valid_acc
            PATH='/home/dldx/UniRep/Model_Trained/gain_2700_train_best'
            torch.save(model.state_dict(),PATH)

datalist=[train_loss,valid_accs]    
loss_acc='gain_acc_2700.p'
with open(loss_acc, 'wb') as f:
    pkl.dump(datalist, f)
#    #保存模型
PATH='/home/dldx/UniRep/Model_Trained/gain_2700_train_200epoch'
# #Save:
torch.save(model.state_dict(),PATH)
end = time.time()
print("运行时间:%.2f秒"%(end-start))      
print("work down!")