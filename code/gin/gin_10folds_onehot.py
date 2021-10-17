import sys
sys.path.append('/home/dldx/UniRep/pipgcn')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from my_dataloader import GraphDataLoader, collate
from gin import GIN
from tqdm import tqdm
import random
import os
import time
import scipy.sparse as sp
import pickle
import dgl
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

dataset_name = "/home/dldx/UniRep/dataset/data_2700_onehot.p"
with open(dataset_name, 'rb') as f:
    data2700 = pickle.load(f)
print(len(data2700) )
def train( net, trainloader, optimizer, criterion, epoch):
    net.train()
    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)

    for pos, (graphs, labels,names) in zip(bar, trainloader):
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(device)
        #feat = graphs.ndata['attr'].to(device)
        feat = graphs.ndata['feat'].float().to(device)
        output = net(graphs, feat)
#         outputs=output[0]
        loss = criterion(output, labels)
        running_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        bar.set_description('epoch-{}'.format(epoch))
    bar.close()
    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss 

@torch.no_grad()
def eval_net( net, dataloader, criterion):
    net.eval()

    total = 0
    total_loss = 0
    total_correct = 0
    targets,outputs=[],[]
    for data in dataloader:
        graphs, labels,names = data
        feat = graphs.ndata['feat'].float().to(device)
        labels = labels.to(device)
        total += len(labels)
        output = net(graphs, feat)
#         outputs=output[0]
        _, predicted = torch.max(output.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(output, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)
        targets.append(labels)
        outputs.append(output)
    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    return acc,targets,outputs

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
    
#模型参数
start = time.time()
acc_scores=[]
datalist=[]
for fold_idx in range(10):
    model =GIN(5, 2, 20, 64, 3, 0.5, True, "sum", "sum").to(device)

    epochs=100
    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    tbar = tqdm(range(epochs), unit="epoch", position=3, ncols=0, file=sys.stdout)
    vbar = tqdm(range(epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)
    lrbar = tqdm(range(epochs), unit="epoch", position=5, ncols=0, file=sys.stdout)
    trainloader, validloader = GraphDataLoader(data2700, batch_size=64, device=device,
                                                collate_fn=collate, seed=2021, shuffle=True,
                                                split_name='fold10', fold_idx=fold_idx).train_valid_loader()


    valid_acc_tmp=0
    for epoch, _, _ in zip(tbar, vbar, lrbar):

        train( model, trainloader, optimizer, criterion, epoch)
        scheduler.step()
        valid_acc,target,output = eval_net(model, validloader, criterion)
        targets=target
        outputs=output
        if valid_acc > valid_acc_tmp:
            valid_acc_tmp=valid_acc
                #    #保存模型
#             PATH='/home/dldx/UniRep/Model_Trained/gin_onehot_best_new'+str(fold_idx)
#             torch.save(model.state_dict(),PATH)
        
    datalist.append([valid_acc_tmp,targets,outputs])


    
    tbar.close()
    vbar.close()
    lrbar.close()
    #    #保存模型
#     PATH='/home/dldx/UniRep/Model_Trained/gin_onehot_new'+str(fold_idx)
#     torch.save(model.state_dict(),PATH)

    print("第{}折完成,准确率{}".format(fold_idx,valid_acc_tmp)) 

loss_acc='gin_acc_10fold_onehot_new.p'
with open(loss_acc, 'wb') as f:
    pickle.dump(datalist, f)
end = time.time()
print("运行时间:%.2f秒"%(end-start))      
print("work down!")