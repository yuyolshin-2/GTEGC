from random import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import os
import pickle as pkl
import networkx as nx
from scipy.linalg import block_diag
from sympy.logic.inference import valid
from torch.nn.init import sparse

from normalization import fetch_normalization, row_normalize, min_max_normalize, standardize, StandardScaler, MinMaxScaler
from time import perf_counter


class NetworkDataLoader(object):
    def __init__(self, xs, ys, adj, mask, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        self.permutation = None

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding_size = list(xs.size())
            x_padding_size[0] = num_padding
            y_padding_size = list(ys.size())
            y_padding_size[0] = num_padding
            mask_padding_size = list(mask.size())
            mask_padding_size[0] = num_padding
            x_padding = torch.zeros(x_padding_size).cuda()
            y_padding = torch.zeros(y_padding_size).cuda()
            mask_padding = torch.zeros(mask_padding_size).cuda()
            xs = torch.concatenate((xs, x_padding), dim=0)
            ys = torch.concatenate((ys, y_padding), dim=0)
            mask = torch.concatenate((mask, mask_padding), dim=0)

        self.size = len(xs)
        self.num_batch = int(((self.size - 1) // self.batch_size) + 1)
        self.xs = xs
        self.ys = ys
        self.adj = adj
        self.mask = mask

    def shuffle(self):
        self.permutation = np.random.permutation(self.size)
        xs, ys = self.xs[self.permutation, ...], self.ys[self.permutation, ...]
        # adj = self.adj[self.permutation, :][:, self.permutation]
        adj = permute_sparse_matrix(self.adj, self.permutation)
        mask = self.mask[self.permutation, ...]

        self.xs = xs
        self.ys = ys
        self.adj = adj
        self.mask = mask

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                # adj_i = self.adj[start_ind: end_ind, start_ind: end_ind]
                adj_i = sparse_indexing(self.adj, start_ind=start_ind, end_ind=end_ind)
                mask_i = self.mask[start_ind: end_ind, start_ind: end_ind]

                if self.permutation is not None:
                    yield x_i, y_i, adj_i, mask_i, self.permutation[torch.arange(start_ind, end_ind)]
                else:
                    yield x_i, y_i, adj_i, mask_i, torch.arange(start_ind, end_ind)
                self.current_ind += 1

        return _wrapper()


class MultiNetworkDataLoader(object):
    def __init__(self, xlist, ylist, adjlist, masklist, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        self.permutation = None

        new_xlist = []
        new_ylist = []
        new_adjlist = []
        new_masklist = []

        for i, (xs, ys, adj, mask) in enumerate(zip(xlist, ylist, adjlist, masklist)):
            if pad_with_last_sample:
                num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
                x_padding_size = list(xs.size())
                x_padding_size[0] = num_padding
                y_padding_size = list(ys.size())
                y_padding_size[0] = num_padding
                mask_padding_size = list(mask.size())
                mask_padding_size[0] = num_padding
                x_padding = torch.zeros(x_padding_size).cuda()
                y_padding = torch.zeros(y_padding_size).cuda()
                mask_padding = torch.zeros(mask_padding_size).cuda()
                xs = torch.concatenate((xs, x_padding), dim=0)
                ys = torch.concatenate((ys, y_padding), dim=0)
                mask = torch.concatenate((mask, mask_padding), dim=0)

            new_xlist.append(xs)
            new_ylist.append(ys)
            new_adjlist.append(adj)
            new_masklist.append(mask)

        self.xlist = new_xlist
        self.ylist = new_ylist
        self.adjlist = new_adjlist
        self.masklist = new_masklist

        self.sizes = [len(xs) for xs in new_xlist]
        self.num_batchs = [int(((ss - 1) // self.batch_size) + 1) for ss in self.sizes]

        self.shuffle_samples = False

    def shuffle(self):
        self.permutations = [np.random.permutation(ss) for ss in self.sizes]

        new_xlist = [xs[self.permutations[i], ...] for i, xs in enumerate(self.xlist)]
        new_ylist = [ys[self.permutations[i], ...] for i, ys in enumerate(self.ylist)]
        new_masklist = [mask[self.permutations[i], ...] for i, mask in enumerate(self.masklist)]
        new_adjlist = [permute_sparse_matrix(adj, self.permutations[i]) for i, adj in enumerate(self.adjlist)]

        self.xlist = new_xlist
        self.ylist = new_ylist
        self.masklist = new_masklist
        self.adjlist = new_adjlist

        self.shuffle_samples = True


    def get_iterator(self):
        if self.shuffle_samples:
            self.batch_indices = list(np.random.permutation(sum(self.num_batchs)))
        else:
            self.batch_indices = list(np.arange(sum(self.num_batchs)))
        self.current_ind = 0

        def _wrapper():
            while not not self.batch_indices:
                bid = self.batch_indices.pop()
                bid += 1
                i = 0
                while bid > 0:
                    if bid - self.num_batchs[i] <= 0:
                        start_ind = self.batch_size * (bid - 1)
                        end_ind = min(self.sizes[i], self.batch_size * bid)
                        x_i = self.xlist[i][start_ind: end_ind, ...]
                        y_i = self.ylist[i][start_ind: end_ind, ...]
                        # adj_i = self.adjlist[i][start_ind: end_ind, start_ind: end_ind]
                        adj_i = sparse_indexing(self.adjlist[i], start_ind, end_ind)
                        mask_i = self.masklist[i][start_ind: end_ind, start_ind: end_ind]

                        bid -= self.num_batchs[i]
                        if self.permutation is not None:
                            yield x_i, y_i, adj_i, mask_i, self.permutations[i][torch.arange(start_ind, end_ind)], i
                        else:
                            yield x_i, y_i, adj_i, mask_i, torch.arange(start_ind, end_ind), i
                    else:
                        bid -= self.num_batchs[i]
                        i += 1
                # self.current_ind += 1


        return _wrapper()


def get_A_r(adj, r):
    if r == 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N = adj.shape[0]
        indices = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
        values = torch.ones(N, device=device)
        adj_label = torch.sparse_coo_tensor(indices, values, (N, N), device=device)
        adj_label = adj_label.coalesce()
    elif r == 1:
        adj_label = adj_label
    elif r == 2:
        # adj_label = adj_label @ adj_label
        adj_label = torch.sparse.mm(adj_label, adj_label)
    elif r == 3:
        # adj_label = adj_label @ adj_label @ adj_label
        adj_label = torch.sparse.mm(torch.sparse.mm(adj_label, adj_label), adj_label)
    elif r == 4:
        # adj_label = adj_label @ adj_label @ adj_label @ adj_label
        adj_label = torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(adj_label, adj_label), adj_label), adj_label)
    elif r == 5:
        # adj_label = adj_label @ adj_label @ adj_label @ adj_label @ adj_label
        adj_label = torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(adj_label, adj_label), adj_label), adj_label), adj_label)
    elif r == 6:
        # adj_label = adj_label @ adj_label @ adj_label @ adj_label @ adj_label
        adj_label = torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(
            adj_label, adj_label), adj_label), adj_label), adj_label), adj_label)
    else:
        a1 = adj_label
        for i in range(r - 1):
            adj_label = torch.sparse.mm(adj_label, a1)
    return adj_label


def preprocess_roadnetwork(adj, features, labels, normalization="FirstOrderGCN", minmax=None):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)

    features, _, _ = min_max_normalize(features, minmax)
    features = torch.nan_to_num(features, nan=0.0)

    labelScaler = MinMaxScaler(min_=0, max_=80)

    label_dict = {}
    label_dict['data'] = labels
    label_dict['scaler'] = labelScaler

    return adj, features, label_dict


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def extract_subgraph_from_indices(sparse_mx, node_indices):
    if torch.cuda.is_available():
        node_indices = node_indices.cuda()
    mask = torch.isin(sparse_mx._indices()[0], node_indices) & torch.isin(sparse_mx._indices()[1], node_indices)

    filtered_indices = sparse_mx._indices()[:, mask]
    filtered_values = sparse_mx._values()[mask]

    node_map = {node.item(): i for i, node in enumerate(node_indices)}
    mapped_indices = torch.tensor([[node_map[idx.item()] for idx in filtered_indices[0]],
                                   [node_map[idx.item()] for idx in filtered_indices[1]]])
    if torch.cuda.is_available():
        mapped_indices = mapped_indices.cuda()
    sub_sparse_adj = torch.sparse_coo_tensor(mapped_indices, filtered_values, (len(node_indices), len(node_indices)))
    return sub_sparse_adj


def permute_sparse_matrix(sparse_mx, perm):
    sparse_mx = sparse_mx.coalesce()
    indices = sparse_mx.indices()
    values = sparse_mx.values()
    perm = torch.LongTensor(perm).cuda()

    # Permute the row and column indices based on the given permutation
    new_indices = torch.stack([perm[indices[0]], perm[indices[1]]], dim=0)

    # Create a new sparse tensor with the same values but permuted indices
    permuted_sparse_mx = torch.sparse_coo_tensor(new_indices, values, sparse_mx.shape, dtype=sparse_mx.dtype)

    return permuted_sparse_mx


def sparse_indexing(adj, start_ind, end_ind):
    """
    Index a sparse COO adjacency matrix to extract a submatrix.

    Args:
        adj (torch.sparse_coo_tensor): Input sparse adjacency matrix in COO format.
        start_ind (int): Starting index of the submatrix.
        end_ind (int): Ending index (exclusive) of the submatrix.

    Returns:
        torch.sparse_coo_tensor: The extracted submatrix in sparse format.
    """
    # Extract COO indices and values

    indices = adj._indices()  # Shape [2, nnz]
    values = adj._values()    # Shape [nnz]

    # Filter indices within the specified range
    mask = (indices[0] >= start_ind) & (indices[0] < end_ind) & \
           (indices[1] >= start_ind) & (indices[1] < end_ind)

    # Select relevant indices and values
    sub_indices = indices[:, mask] - start_ind  # Shift indices for the submatrix
    sub_values = values[mask]

    # Create a new sparse tensor with the updated indices and values
    new_size = (end_ind - start_ind, end_ind - start_ind)
    sub_adj = torch.sparse_coo_tensor(sub_indices, sub_values, new_size, dtype=adj.dtype, device=adj.device)

    return sub_adj


def load_dataset(args, normalization="AugNormAdj", cuda=True, indiv_norm=True):
    """
    Load Road Networks Datasets.
    """

    used_cols = ['Id', 'Length', 'FRC', 'SpeedLimit',
                 'nSegment', '111', '112', '121', '122', '123', '124', '131', '132',
                 '133', '141', '142', '200', '300', '400', '500']

    if type(args.area) != list:
        areas = [args.area]
    else:
        areas = args.area

    features = []
    labels = []
    adjs = []
    adj_sizes = []
    for area in areas:
        network_data_path = os.path.join(args.data_directory, area, args.network_data)
        adj_data_path = os.path.join(args.data_directory, area, args.adj_data)
        label_data_path = os.path.join(args.data_directory, area, args.label_data)

        tmp_features = pd.read_csv(network_data_path, usecols=used_cols)
        tmp_features = torch.FloatTensor(tmp_features[used_cols].iloc[:, 1:].values)
        tmp_labels = pd.read_csv(label_data_path).values[:, 3:]
        tmp_adj = sp.load_npz(adj_data_path)
        if indiv_norm:
            tmp_adj, tmp_features, tmp_label_dict = preprocess_roadnetwork(
                tmp_adj, tmp_features, tmp_labels, normalization)
            adj_sizes.append(tmp_adj.shape[0])
            features.append(tmp_features)
            labels.append(tmp_label_dict)
            adjs.append(tmp_adj)
        else:
            adj_sizes.append(tmp_adj.shape[0])
            features.append(tmp_features)
            labels.append(tmp_labels)
            adjs.append(tmp_adj)

    if not indiv_norm:
        mins = []
        maxs = []
        for i, area in enumerate(areas):
            mins.append(torch.min(features[i], dim=0).values)
            maxs.append(torch.max(features[i], dim=0).values)
        mins = torch.stack(mins, dim=0).min(dim=0, keepdim=True).values
        maxs = torch.stack(maxs, dim=0).max(dim=0, keepdim=True).values
        for i, area in enumerate(areas):
            tmp_adj, tmp_features, tmp_label_dict = preprocess_roadnetwork(
                adjs[i], features[i], labels[i], normalization, minmax=(mins, maxs))

            features[i] = tmp_features
            labels[i] = tmp_label_dict
            adjs[i] = tmp_adj

    if type(args.transfer_area) != list:
        transfer_areas = [args.transfer_area]
    else:
        transfer_areas = args.transfer_area

    transfer_features = []
    transfer_labels = []
    transfer_adjs = []
    transfer_adj_sizes = []

    for area in transfer_areas:
        transfer_network_data_path = os.path.join(args.data_directory, area, args.network_data)
        transfer_adj_data_path = os.path.join(args.data_directory, area, args.adj_data)
        transfer_label_data_path = os.path.join(args.data_directory, area, args.label_data)

        tmp_transfer_features = pd.read_csv(transfer_network_data_path, usecols=used_cols)
        tmp_transfer_features = torch.FloatTensor(tmp_transfer_features[used_cols].iloc[:, 1:].values)
        tmp_transfer_labels = pd.read_csv(transfer_label_data_path).values[:, 3:]
        tmp_adj = sp.load_npz(transfer_adj_data_path)
        if indiv_norm:
            tmp_adj, tmp_transfer_features, tmp_transfer_label_dict = preprocess_roadnetwork(
                tmp_adj, tmp_transfer_features, tmp_transfer_labels, normalization)
        else:
            tmp_adj, tmp_transfer_features, tmp_transfer_label_dict = preprocess_roadnetwork(
                tmp_adj, tmp_transfer_features, tmp_transfer_labels, normalization, minmax=(mins, maxs)
            )
        transfer_labels.append(tmp_transfer_label_dict)
        transfer_features.append(tmp_transfer_features)
        transfer_adjs.append(tmp_adj)
        transfer_adj_sizes.append(tmp_adj.shape[0])

    # porting to pytorch
    features = [torch.FloatTensor(np.array(fs.to_dense())).float() for fs in features]
    for i in range(len(labels)):
        labels[i]['data'] = torch.FloatTensor(labels[i]['data'])
        mask = ~torch.isnan(labels[i]['data'])
        labels[i]['mask'] = mask

    transfer_features = [torch.FloatTensor(np.array(tfs.to_dense())).float() for tfs in transfer_features]
    for i in range(len(transfer_labels)):
        transfer_labels[i]['data'] = torch.FloatTensor(transfer_labels[i]['data'])
        transfer_mask = ~torch.isnan(transfer_labels[i]['data'])
        transfer_labels[i]['mask'] = transfer_mask

    adjs = [sparse_mx_to_torch_sparse_tensor(a).float() for a in adjs]

    transfer_adjs = [sparse_mx_to_torch_sparse_tensor(tadj).float() for tadj in transfer_adjs]

    if cuda:
        features = [f.cuda() for f in features]
        adjs = [a.cuda() for a in adjs]
        for i in range(len(labels)):
            labels[i]['data'] = labels[i]['data'].cuda()
            labels[i]['mask'] = labels[i]['mask'].cuda()
        
        transfer_features = [tfs.cuda() for tfs in transfer_features]
        transfer_adjs = [tadj.cuda() for tadj in transfer_adjs]
        for i in range(len(transfer_labels)):
            transfer_labels[i]['data'] = transfer_labels[i]['data'].cuda()
            transfer_labels[i]['mask'] = transfer_labels[i]['mask'].cuda()
      
    # indices for regression

    indices = {}
    train_indices = []
    valid_indices = []
    test_indices = []
    permutations = []
    for j, l in enumerate(adj_sizes):
        random_indices = torch.randperm(l)

        train_idx = random_indices[:int(l * 0.6)]
        val_idx = random_indices[int(l * 0.6):int(l * 0.8)]
        test_idx = random_indices[int(l * 0.8):]

        train_indices.append(train_idx)
        valid_indices.append(val_idx)
        test_indices.append(test_idx)
        permutations.append(random_indices)
        indices['train'] = train_indices
        indices['val'] = valid_indices
        indices['test'] = test_indices
        indices['permutation'] = permutations

    return adjs, features, labels, indices, transfer_adjs, transfer_features, transfer_labels


def masked_mse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels) & ~torch.isnan(preds)
    else:
        mask = (labels!=null_val) & (preds!=null_val)

    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels) & ~torch.isnan(preds)
    else:
        mask = (labels!=null_val) & (preds!=null_val)

    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels) & ~torch.isnan(preds)
    else:
        mask = (labels!=null_val) & (preds!=null_val)

    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_r2(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels) & ~torch.isnan(preds)
    else:
        mask = (labels!=null_val) & (preds!=null_val)

    preds = preds[mask].view(-1)
    labels = labels[mask].view(-1)

    ss_res = torch.sum((labels - preds) ** 2)
    y_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - y_mean) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    return r2


def masked_spearman(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels) & ~torch.isnan(preds)
    else:
        mask = (labels!=null_val) & (preds!=null_val)

    preds = preds[mask].view(-1)
    labels = labels[mask].view(-1)

    pred_rank = torch.argsort(torch.argsort(preds))
    label_rank = torch.argsort(torch.argsort(labels))
    d = pred_rank - label_rank
    d2 = d.float() ** 2
    n = preds.size(0)
    spearman_corr = 1 - (6 * torch.sum(d2) / (n * (n ** 2 - 1)))
    return spearman_corr


def masked_kendall(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels) & ~torch.isnan(preds)
    else:
        mask = (labels!=null_val) & (preds!=null_val)

    preds = preds[mask].view(-1)
    labels = labels[mask].view(-1)

    n = preds.size(0)

    concordant = 0
    disconcordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            x_diff = preds[i] - preds[j]
            y_diff = labels[i] - labels[j]

            if (x_diff * y_diff) > 0:
                concordant += 1
            elif (x_diff * y_diff) < 0:
                disconcordant += 1

    tau = (concordant - disconcordant) / (0.5 * n * (n - 1))
    return tau


def metric(pred, real, corrs=False, null_val=0.0, inverse=False, scaler=None):
    if inverse:
        pred = scaler.inverse_transform(pred)
        real = scaler.inverse_transform(real)
    if not np.isnan(null_val):
        pred = torch.nan_to_num(pred, nan=null_val)
        real = torch.nan_to_num(real, nan=null_val)
    if np.isnan(null_val):
        num_nulled = torch.where(torch.abs(pred) > 200)[0].size(0)
        pred = torch.where(torch.abs(pred) > 200, torch.nan, pred)
    else:
        num_nulled = torch.where(torch.abs(pred) > 200)[0].size(0)
        pred = torch.where(torch.abs(pred) > 200, null_val, pred)
    mae = masked_mae(pred, real, null_val).item()
    mape = masked_mape(pred, real, null_val).item()
    rmse = masked_rmse(pred, real, null_val).item()
    r2 = masked_r2(pred, real, null_val).item()
    if corrs:
        spearman = masked_spearman(pred, real, null_val).item()
        kendall = masked_kendall(pred, real, null_val)
        return mae, mape, rmse, r2, spearman, kendall
    else:
        return mae, mape, rmse, r2



def get_feature_dis_ncontrast(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the cosine similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


if __name__ == '__main__':
    load_dataset()



