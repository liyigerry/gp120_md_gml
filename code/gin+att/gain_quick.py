# -*- coding: utf-8 -*- 
# @Time:2020-06-20
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
 Paramters
        ---------
 num_layers: int
 The number of linear layers
 input_dim: int
 The dimensionality of input features
 hidden_dim: int
 The dimensionality of hidden units at ALL layers
 output_dim: int
 The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        """model parameters setting
 Paramters
        ---------
 num_layers: int
 The number of linear layers in the neural network
 num_mlp_layers: int
 The number of linear layers in mlps
 input_dim: int
 The dimensionality of input features
 hidden_dim: int
 The dimensionality of hidden units at ALL layers
 output_dim: int
 The number of classes for prediction
 final_dropout: float
 dropout ratio on the final linear layer
 learn_eps: boolean
 If True, learn epsilon to distinguish center nodes from neighbors
 If False, aggregate neighbors and center nodes altogether.
 neighbor_pooling_type: str
 how to aggregate neighbors (sum, mean, or max)
 graph_pooling_type: str
 how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        #self.kenel_1 = nn.Parameter(torch.Tensor(size = (1,input_dim)))#-----
        self.kenel_2 = nn.Parameter(torch.Tensor(size = (1,hidden_dim)))#---------------#nn.Parameter：可以训练的torch参数
        nn.init.xavier_normal_(self.kenel_2, gain=nn.init.calculate_gain('relu'))#-----#将该参数self.kenel_2初始化
        

        for layer in range(self.num_layers-1):#-----
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):#-----
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):

        seed=0#-----
        torch.manual_seed(seed=seed)#-----
        np.random.seed(seed=seed)#-----
        if torch.cuda.is_available():#-----

            device = torch.device("cuda")#-----
            torch.cuda.manual_seed_all(seed=seed)#-----
        else:
            device = torch.device("cpu")#-----
        '''CUDA========================================================='''
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]
        for i in range(self.num_layers-1):#-----
            h = self.ginlayers[i](g, h)
            # print(self.ginlayers[i].eps)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        '''============================Attetion=============================''' 
        score_over_layer = 0
        # num_nodes of each grapg
        num_nodes = g.batch_num_nodes#-----s#-----
        #num_nodes[0]'：476，节点数
        # node index for each graph
        node_index = []#-----
        index = 0#-----
        #-----
        for i in range(len(num_nodes)):
            index += num_nodes[i]
            node_index.append(index)
            
        # perform pooling over all nodes in each graph in every layer
        total_atten_score = []
        for i, h in enumerate(hidden_rep):
            # pooled_h = self.pool(g, h)
            atten_score = [] # atten_score of each graph
            if i != 0:
                #self.kenel_2.view(1,-1):可学习参数,和隐藏层表示矩阵相乘,得到atten_sc，也就是所谓的alpha的值
                atten_sc = torch.mm(self.kenel_2.view(1,-1), h.t())

            else:
                continue
            #'atten_sc.shape:',torch.Size([1, 7655])
            #len(num_nodes)=batchsize=16
            atten_h = torch.Tensor().to(device)#tensor([])
            for _ in range(len(num_nodes)): #对于每一个图循环：
        
                if _ != 0:##选取结点对应的alpha的值
                    atten_score_temp = atten_sc[:, node_index[_-1]:node_index[_]]  #从最后一个结点到当前结点
                    node_feat_solo=h[node_index[_-1]:node_index[_]]
                    atten_score_solo=F.softmax(atten_score_temp,dim=1)# 用softmax对alpha的值做归一化处理
                    atten_score.append(atten_score_solo)
                    atten_h_temp=torch.mm(atten_score_solo,node_feat_solo) #'atten_h_temp.shape:',torch.Size([64])
                    atten_h = torch.cat((atten_h, atten_h_temp), 0).to(device)
                else: #从0到当前结点
                    atten_score_temp = atten_sc[:, 0:node_index[_]]
                    node_feat_solo=h[0:node_index[_]]
                    atten_score_solo=F.softmax(atten_score_temp,dim=1)
                    atten_score.append(atten_score_solo)
                    atten_h_temp=torch.mm(atten_score_solo,node_feat_solo) #'atten_h_temp.shape:',torch.Size([64])
                    atten_h = torch.cat((atten_h, atten_h_temp), 0).to(device)
                                  
            total_atten_score.append(atten_score)

            '''============================Attetion================================''' 
            '''============================Linear-Drop=============================''' 
            score_over_layer += self.drop(self.linears_prediction[i](atten_h))
            
        return score_over_layer, total_atten_score