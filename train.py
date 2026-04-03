import torch
import torch.nn.functional as F
from numba.cloudpickle import instance
from torch.nn.functional import mse_loss

from utils_ssl import metric, get_feature_dis_ncontrast, masked_mae
from normalization import standardize
import numpy as np
import sys

def Ncontrast(x, adj_label, temperature=1):
    """
    compute the Ncontrast loss
    """
    x_dis = get_feature_dis_ncontrast(x)
    x_dis = torch.exp(x_dis / temperature)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label.to_dense(), 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    #### if you want to consider only negative pairs for denominator
    return loss


def fit_eval_reg(args, reg, features, labels, indices, corrs=False, inverse_transform=False, cross_val=True):

    permutation = indices['permutation']
    scaler = labels['scaler']
    L = features.size(0)

    folds = torch.chunk(permutation, 5)
    if cross_val:
        metrics = 0
        for i in range(5):
            idx_train = torch.cat([folds[j] for j in range(5) if j != i])
            idx_val = folds[i]

            train_null_index = torch.where(~torch.isnan(labels['data'][idx_train]))[0]
            val_null_index = torch.where(~torch.isnan(labels['data'][idx_val]))[0]
            reg.fit(features[idx_train, ...][train_null_index, ...], labels['data'][idx_train][train_null_index])
            reg_output = reg.predict(features[idx_val, ...][val_null_index, ...])
            metrics += np.array(metric(reg_output, labels['data'][idx_val][val_null_index],
                                       corrs=corrs, null_val=args.null_val, inverse=inverse_transform, scaler=scaler))
        metrics /= 5
    else:
        idx_train, idx_val = indices['train'], indices['val']
        train_null_index = torch.where(~torch.isnan(labels['data'][idx_train]))[0]
        val_null_index = torch.where(~torch.isnan(labels['data'][idx_val]))[0]
        reg.fit(features[idx_train, ...][train_null_index, ...], labels['data'][idx_train][train_null_index])
        reg_output = reg.predict(features[idx_val, ...][val_null_index, ...])
        metrics = metric(reg_output, labels['data'][idx_val][val_null_index],
                         corrs=corrs, null_val=args.null_val, inverse=inverse_transform, scaler=scaler)

    return metrics, reg


def train_batch(args, model, features, adj, optimizer,
                index, labels, label_mask, scaler, cluster_results=None, contrast=False):
    model.train()
    optimizer.zero_grad()
    network_output, z_proto = model(features, adj)
    # network_output = network_output
    labels = torch.nan_to_num(labels, nan=0.0)
    network_output = scaler.inverse_transform(network_output)
    mse_loss = masked_mae(network_output, labels, null_val=0.0)

    losses = []
    loss = mse_loss

    metrics = metric(network_output, labels, null_val=0.0, corrs=False)

    contrast_loss = 0.0
    if contrast:
        instance_contrast_loss = args.alpha * Ncontrast(z_proto, adj, temperature=args.tau)
    else:
        instance_contrast_loss = 0.0

      contrast_loss = instance_contrast_loss
      loss += contrast_loss
      losses.append(loss.item())

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
    optimizer.step()

    return losses, metrics


def eval_batch(args, model, features, adj, labels, label_mask, scaler, corrs=False):
    model.eval()
    val_network, z_proto = model(features, adj)
    val_network = scaler.inverse_transform(val_network)
    labels = torch.nan_to_num(labels, nan=0.0)
    val_loss = masked_mae(val_network, labels, null_val=0.0)

    metrics = metric(val_network, labels, null_val=0.0, corrs=corrs)

    return val_network, val_loss.item(), metrics



