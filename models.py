import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from torch_geometric.nn import GATConv
import os
import sys
import math
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from aggregators import *
from torch.autograd import Variable
from torchvision.models.convnext import LayerNorm2d


class LayerNormConv2d(nn.Module):
    """
    Layer norm the just works on the channel axis for a Conv2d

    Ref:
    - code modified from https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/LayerNorm.py
    - paper: https://arxiv.org/abs/1607.06450

    Usage:
        ln = LayerNormConv(3)
        x = Variable(torch.rand((1,3,4,2)))
        ln(x).size()
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNormConv2d, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features).cuda()).unsqueeze(-1).unsqueeze(-1)
        self.beta = nn.Parameter(torch.zeros(features).cuda()).unsqueeze(-1).unsqueeze(-1)
        self.eps = eps
        self.features = features

    def _check_input_dim(self, input):
        if input.size(1) != self.gamma.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.features))

    def forward(self, x):
        self._check_input_dim(x)
        x_flat = x.transpose(1,-1).contiguous().view((-1, x.size(1)))
        mean = x_flat.mean(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        std = x_flat.std(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)


class Mlp(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(Mlp, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.act_fn = F.relu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        # x = self.batchnorm(x.permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FC(nn.Module):
    def __init__(self, input_dim, out_dim, activation=True, batchnorm=True):
        super(FC, self).__init__()
        self.fc = nn.Conv2d(input_dim, out_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0))

        self.activation = activation
        self.norm = batchnorm
        self.act_fn = F.relu
        self._init_weights()

        self.batchnorm = nn.BatchNorm2d(out_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, std=1e-6)

    def forward(self, x):
        x = x.permute(2, 0, 1).contiguous().unsqueeze(0)
        x = self.fc(x)

        if self.norm:
            x = self.batchnorm(x)
        if self.activation:
            x = self.act_fn(x)
        x = x.squeeze(0).permute(1, 2, 0).contiguous()
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(GCN, self).__init__()
        self.nconv = nconv()
        self.gcn1 = nn.Linear(input_dim, hid_dim)
        self.gcn2 = nn.Linear(hid_dim, hid_dim)
        self.act_fn = F.relu
        self.batchnorm = nn.BatchNorm2d(hid_dim, eps=1e-6)
        self.dropout = Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.gcn1.weight)
        nn.init.xavier_uniform_(self.gcn2.weight)
        nn.init.normal_(self.gcn1.bias, std=1e-6)
        nn.init.normal_(self.gcn2.bias, std=1e-6)

    def forward(self, x, adj):
        V, T, D = x.size()
        x = self.nconv(x, adj)
        x = self.gcn1(x)
        x = self.act_fn(x)
        x = self.batchnorm(x.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        x = self.dropout(x)
        x = self.nconv(x, adj)
        x = self.gcn2(x)
        return x

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, adj):
        V, T, D = x.size()
        x = torch.sparse.mm(adj, x.contiguous().view(V, -1))
        x = x.view(-1, T, D)
        return x.contiguous()

class temporalAttention(nn.Module):
    '''
        temporal attention mechanism
        X:      [batch_size, num_step, num_vertex, D]
        STE:    [batch_size, num_step, num_vertex, D]
        K:      number of attention heads
        d:      dimension of each attention outputs
        return: [batch_size, num_step, num_vertex, D]
        '''

    def __init__(self, input_dim, heads, head_dim):
        super(temporalAttention, self).__init__()
        D = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.FC_q = nn.Conv2d(input_dim, D, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.FC_k = nn.Conv2d(input_dim, D, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.FC_v = nn.Conv2d(input_dim, D, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.FC = nn.Conv2d(D, D, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.FC_q.weight)
        nn.init.xavier_uniform_(self.FC_k.weight)
        nn.init.xavier_uniform_(self.FC_v.weight)
        nn.init.xavier_uniform_(self.FC.weight)
        nn.init.normal_(self.FC_q.bias, std=1e-6)
        nn.init.normal_(self.FC_k.bias, std=1e-6)
        nn.init.normal_(self.FC_v.bias, std=1e-6)
        nn.init.normal_(self.FC.bias, std=1e-6)

    def forward(self, x):
        # x: [num_nodes, num_hours, input_dim] -> [1, heads * dim, num_nodes, num_hours]
        x = x.permute(2, 0, 1).contiguous().unsqueeze(0)
        query = self.FC_q(x)
        key = self.FC_k(x)
        value = self.FC_v(x)
        # [heads, dim, num_nodes, num_hours]
        query = torch.cat(torch.split(query, self.head_dim, dim=1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=1), dim=0)
        # query: [heads, num_nodes, num_hours, dim]
        # key:   [heads, num_nodes, dim, num_hours]
        # value: [heads, num_nodes, num_hours, dim]
        query = query.permute(0, 2, 3, 1).contiguous()
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 3, 1).contiguous()
        # [heads, num_nodes, num_hours, num_hours]
        attention = torch.matmul(query, key)
        attention /= (self.head_dim ** 0.5)
        # softmax
        attention = F.softmax(attention, dim=-1)
        # [1, heads * head_dim, num_nodes, num_hours]
        x = torch.matmul(attention, value)
        x = torch.cat(torch.split(x, 1, dim=0), dim=-1).permute(0, 3, 1, 2).contiguous()
        # [num_nodes, num_hours, heads * head_dim]
        x = self.FC(x)
        x = x.permute(0, 2, 3, 1).contiguous().squeeze()
        del query, key, value, attention
        return x

class gatedFusion(nn.Module):
    def __init__(self, dim):
        super(gatedFusion, self).__init__()
        self.fc_s = FC(dim, dim, activation=False, batchnorm=False)
        self.fc_t = FC(dim, dim, activation=False, batchnorm=False)
        self.fc_h1 = FC(dim, dim)
        self.fc_h2 = FC(dim, dim, activation=False, batchnorm=False)

    def forward(self, xs, xt):
        hs = self.fc_s(xs)
        ht = self.fc_t(xt)
        z = torch.sigmoid(torch.add(hs, ht))
        h = torch.add(torch.mul(z, xs), torch.mul(1 - z, xt))
        h = self.fc_h1(h)
        h = self.fc_h2(h)
        return h

class GTEGC(nn.Module):
    def __init__(self, nfeat, landuse_categories, nhid_features=64, nhid_landuse=32, nhid=64, dropout=0.6, layers=2,
                 gnn='GCN'):
        super(GTEGC, self).__init__()
        self.nhid = nhid
        self.landuse_categories = landuse_categories
        self.split = nfeat - landuse_categories
        self.layers = layers

        self.te_transform = Mlp(18, nhid, dropout)
        self.x_transform = nn.Linear(nfeat - landuse_categories, nhid_features)
        self.landuse_transform = nn.Linear(landuse_categories, nhid_landuse)

        self.linear = nn.Linear(nhid_features + nhid_landuse, nhid)

        self.act_fn = F.relu
        self.dropout = nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm1d(nhid_features + nhid_landuse, eps=1e-6)
        self.batchnorm2 = nn.BatchNorm2d(nhid, eps=1e-6)
        self.batchnorm3 = nn.ModuleList([nn.BatchNorm2d(nhid, eps=1e-6) for _ in range(layers)])

        self.gnn = gnn
        if gnn == 'GCN':
            self.gnns = nn.ModuleList([GCN(nhid, nhid, dropout) for _ in range(layers)])
        elif gnn == 'GMLP':
            self.gnns = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(layers)])
            # self.gnns = nn.ModuleList([GCN(nhid, nhid, dropout) for _ in range(layers)])
        else:
            raise NotImplementedError('GNN layer {} is not implemented'.format(gnn))
        self.temporal_attentions = nn.ModuleList([temporalAttention(nhid, 8, 8) for _ in range(layers)])
        self.gated_fusions = nn.ModuleList([gatedFusion(nhid) for _ in range(layers)])
        self.fc1 = FC(nhid, nhid)
        self.fc2 = FC(nhid, 1, activation=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.x_transform.weight)
        nn.init.xavier_uniform_(self.landuse_transform.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.normal_(self.x_transform.bias, std=1e-6)
        nn.init.normal_(self.landuse_transform.bias, std=1e-6)
        nn.init.normal_(self.linear.bias, std=1e-6)

    def forward(self, network_features, adj):

        if self.gnn == 'PGCN':
            adp = self.graph_construction(network_features)

        TE = F.one_hot(torch.arange(0, 18)).to(dtype=torch.float32).cuda()
        TE = self.te_transform(TE)

        x, landuses = torch.split(network_features, [self.split, self.landuse_categories], dim=-1)
        x = self.x_transform(x)
        landuses = self.landuse_transform(landuses)
        z = torch.cat((x, landuses), dim=-1)

      z = self.act_fn(z)
        z = self.batchnorm1(z)
        z = self.dropout(z)
        z = self.linear(z)

        z = TE.unsqueeze(0) + z.unsqueeze(1)
        z = z.permute(2, 0, 1).unsqueeze(0)
        z = self.batchnorm2(z).squeeze(0).permute(1, 2, 0)

        z = self.act_fn(z)

        for i in range(self.layers):
            zr = z
            if self.gnn == 'PGCN':
                zs = self.gnns[i](z, [adj, adj.T, adp])
            elif self.gnn == 'GMLP':
                zs = self.gnns[i](z)
            else:
                zs = self.gnns[i](z, adj)
            zt = self.temporal_attentions[i](z)
            z = self.gated_fusions[i](zs, zt)
            z = z + zr
            z = self.batchnorm3[i](z.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
            z = self.act_fn(z)
        z = self.act_fn(self.fc1(z))

        z = self.fc2(z)
        return z.squeeze()


class ablation_noTE(nn.Module):
    def __init__(self, nfeat, landuse_categories, nhid_features=64, nhid_landuse=32, nhid=64, dropout=0.6, layers=2):
        super(ablation_noTE, self).__init__()
        self.nhid = nhid
        self.landuse_categories = landuse_categories
        self.split = nfeat - landuse_categories
        self.layers = layers

        self.te_transform = Mlp(18, nhid, dropout)
        self.tfc1 = FC(1, nhid, activation=True, batchnorm=True)
        self.tfc2 = FC(nhid, 18, activation=False, batchnorm=False)

        self.x_transform = nn.Linear(nfeat - landuse_categories, nhid_features)
        # self.x_transform = nn.Linear(nfeat - landuse_categories, nhid)
        self.landuse_transform = nn.Linear(landuse_categories, nhid_landuse)

        self.linear = nn.Linear(nhid_features + nhid_landuse, nhid)
        # self.linear = nn.Linear(nhid, nhid)

        self.act_fn = F.relu
        self.dropout = nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm1d(nhid_features + nhid_landuse, eps=1e-6)
        # self.batchnorm1 = nn.BatchNorm1d(nhid, eps=1e-6)
        self.batchnorm2 = nn.BatchNorm2d(nhid, eps=1e-6)
        self.batchnorm3 = nn.ModuleList([nn.BatchNorm2d(nhid, eps=1e-6) for _ in range(layers)])
        # self.mlp = Mlp(nhid_features + nhid_landuse, self.nhid, dropout)
        # self.mlp = FC(nhid_features + nhid_landuse, 256, 1, dropout)
        self.gnns = nn.ModuleList([GCN(nhid, nhid, dropout) for _ in range(layers)])
        self.temporal_attentions = nn.ModuleList([temporalAttention(nhid, 8, 8) for _ in range(layers)])
        self.gated_fusions = nn.ModuleList([gatedFusion(nhid) for _ in range(layers)])
        self.fc1 = FC(nhid, nhid)
        self.fc2 = FC(nhid, 1, activation=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.x_transform.weight)
        nn.init.xavier_uniform_(self.landuse_transform.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.normal_(self.x_transform.bias, std=1e-6)
        nn.init.normal_(self.landuse_transform.bias, std=1e-6)
        nn.init.normal_(self.linear.bias, std=1e-6)

    def forward(self, network_features, adj):
        x, landuses = torch.split(network_features, [self.split, self.landuse_categories], dim=-1)

        x = self.x_transform(x)
        landuses = self.landuse_transform(landuses)
        z = torch.cat((x, landuses), dim=-1)

        z = self.act_fn(z)
        z = self.batchnorm1(z)
        z = self.dropout(z)
        z = self.linear(z) # z: [N, D]

        z = z.unsqueeze(-1)
        z = self.tfc2(self.tfc1(z))

        z = self.batchnorm2(z.permute(1, 0, 2).unsqueeze(0)).squeeze(0).permute(1, 2, 0)

        z = self.act_fn(z)

        for i in range(self.layers):
            zr = z
            zs = self.gnns[i](z, adj)
            zt = self.temporal_attentions[i](z)

            z = self.gated_fusions[i](zs, zt)
            z = z + zr
            z = self.batchnorm3[i](z.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
            z = self.act_fn(z)
        z = self.fc1(z)
        z = self.fc2(z)

        return z.squeeze()


class GNNmodel(nn.Module):
    def __init__(self, nfeat, landuse_categories, nhid_features=64, nhid_landuse=32, nhid=64, dropout=0.6):
        super(GNNmodel, self).__init__()
        self.nhid = nhid
        self.landuse_categories = landuse_categories
        self.split = nfeat - landuse_categories

        self.x_transform = nn.Linear(nfeat - landuse_categories, nhid_features)
        self.landuse_transform = nn.Linear(landuse_categories, nhid_landuse)

        self.linear = nn.Linear(nhid_features + nhid_landuse, nhid)

        self.act_fn = F.relu
        self.dropout = nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm1d(nhid_features + nhid_landuse, eps=1e-6)
        # self.batchnorm1 = nn.BatchNorm1d(nhid, eps=1e-6)
        self.batchnorm2 = nn.BatchNorm1d(nhid, eps=1e-6)
        self.batchnorm3 = nn.ModuleList([nn.BatchNorm1d(nhid, eps=1e-6) for _ in range(2)])
        self.gnns = nn.ModuleList([GCN(nhid, nhid, dropout) for _ in range(2)])
        self.fc1 = nn.Linear(nhid, nhid)
        self.fc2 = nn.Linear(nhid, 18)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.x_transform.weight)
        nn.init.xavier_uniform_(self.landuse_transform.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.x_transform.bias, std=1e-6)
        nn.init.normal_(self.landuse_transform.bias, std=1e-6)
        nn.init.normal_(self.linear.bias, std=1e-6)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, network_features, adj):
        x, landuses = torch.split(network_features, [self.split, self.landuse_categories], dim=-1)

        x = self.x_transform(x)
        landuses = self.landuse_transform(landuses)
        z = torch.cat((x, landuses), dim=-1)

        z = self.act_fn(z)
        z = self.batchnorm1(z.permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        z = self.dropout(z)
        z = self.linear(z) # z: [N, D]

        z = self.batchnorm2(z)

        z = self.act_fn(z)

        for i in range(2):
            zr = z
            z = self.gnns[i](z.unsqueeze(1), adj).squeeze()
            z = z + zr
            z = self.batchnorm3[i](z)
            z = self.act_fn(z)
        z = self.fc1(z)
        z = self.act_fn(z)
        z = self.fc2(z)

        return z, z



