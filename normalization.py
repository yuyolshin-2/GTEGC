import numpy as np
import scipy.sparse as sp
import torch

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1)).flatten()
   d_inv_sqrt = np.power(row_sum, -0.5)
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
   return adj.tocoo()


def diff_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1)).flatten()
   d_inv = np.power(row_sum, -1.0)
   d_inv[np.isinf(d_inv)] = 0.
   d_mat_inv = sp.diags(d_inv)
   adj = d_mat_inv @ adj
   return adj.tocoo()


def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
       'DiffNormAdj': diff_normalized_adjacency, # A' = (D + I)^-1 * (A + I)
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def min_max_normalize(mx, minmax=None):
    """Min-Max Normalization column-wise"""
    if minmax is None:
        min_vals = torch.min(mx, dim=0, keepdim=True).values
        max_vals = torch.max(mx, dim=0, keepdim=True).values
    else:
        min_vals, max_vals = minmax

    normalized_tensor = (mx - min_vals) / (max_vals - min_vals)
    return normalized_tensor, min_vals, max_vals


def standardize(mx):
    """Standardization column-wise"""
    mean_vals = torch.mean(mx, dim=0, keepdim=True)
    std_vals = torch.std(mx, dim=0, keepdim=True)
    normalized_tensor = (mx - mean_vals) / (std_vals + 1e-5)
    return normalized_tensor, mean_vals, std_vals


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MinMaxScaler():
    """
    Standard the input
    """

    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return (data * (self.max - self.min)) + self.min

