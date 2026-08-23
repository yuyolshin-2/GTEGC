from __future__ import division
from __future__ import print_function
import random
import time
import argparse
from zipfile import error

import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
import glob
import sys

from models import GTEGC
from utils import get_A_r, load_dataset, metric, NetworkDataLoader, masked_mae, MultiNetworkDataLoader, extract_subgraph_from_indices, sparse_mx_to_torch_sparse_tensor
from train import fit_eval_reg, train_batch, eval_batch
from normalization import diff_normalized_adjacency
import warnings
warnings.filterwarnings('ignore')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--area', type=str, default=['Vienna'], help='Study area')
parser.add_argument('--transfer_area', type=str, default=['Zurich', 'Luzern', 'Birmingham', 'London',
                                                          'Munich', 'Paris', 'Lisbon', 'Stockholm',
                                                          'Madrid', 'Vienna'], help='Study area')
parser.add_argument('--experiment_name', type=str, default='outputs',)
parser.add_argument('--transductive', action='store_false')
parser.add_argument('--gnn_layer', type=str, default='GMLP', help='Choice of GNN layer')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data_directory', type=str, default='processed_data',
                    help='dataset to be used')
parser.add_argument('--model_directory', type=str,
                    default='results',)
parser.add_argument('--network_data', type=str, default='processed_data/processed_network.csv', help='network dataset')
parser.add_argument('--adj_data', type=str, default='processed_data/directed_adjacency_matrix.npz',
                    help='data directory for adjacency matrix')
parser.add_argument('--label_data', type=str, default='labels/Hourly_AvgSp.csv',
                    help='data directory for labels')
parser.add_argument('--batch_size', type=int, default=40000, help='batch size')
parser.add_argument('--order', type=int, default=3, help='to compute order-th power of adj')
parser.add_argument('--tau', type=float, default=2.0, help='temperature for Ncontrast loss')
parser.add_argument('--num_landuse_categories', type=int, default=15,
                    help='the number of land use categories')
parser.add_argument('--null_val', type=float, default=float('nan'), help='null val in labels data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def main():
    ## get data
    adj, features, labels, indices, trans_adj, trans_features, trans_labels = load_dataset(args, 'DiffNormAdj', True, indiv_norm=False)
    adj_labels = [get_A_r(a, args.order) for a in adj]
    trans_adj_labels = [get_A_r(tadj, args.order) for tadj in trans_adj]

    model_filepath = os.path.join(args.model_directory, '.'.join(args.area) + '(Train)', args.experiment_name)
    model_state_filenames = glob.glob(os.path.join(model_filepath, '*.pth'))
    tids = [s.split('_')[-1].split('.p')[0] for s in model_state_filenames]

    data_loaders = {a: [] for a in args.area}
    for i in range(len(tids)):
        if args.transductive:
            ids = {}
            ids['train'] = []
            ids['val'] = []
            ids['test'] = []
            ids['permutation'] = []
            for a in args.area:
                indices_filename = os.path.join(args.data_directory, a, 'subnetwork', 'indices.pkl')
                with open(indices_filename, 'rb') as f:
                    tmp = pickle.load(f)
                    ids['train'].extend(tmp['train'])
                    ids['val'].extend(tmp['val'])
                    ids['test'].extend(tmp['test'])
                    ids['permutation'].extend(tmp['permutation'])
        else:
            ids_directory = os.path.join(model_filepath, 'indices_' + tids[i] + '.pkl')
            with open(ids_directory, 'rb') as f:
                ids = pickle.load(f)

        for j in range(len(features)):
            data_loaders[args.area[j]].append(NetworkDataLoader(features[j][ids['test'][j], ...],
                                                                labels[j]['data'][ids['test'][j], ...],
                                                                # adj_labels[j][],
                                                                extract_subgraph_from_indices(adj_labels[j], ids['test'][j]),
                                                                labels[j]['mask'][ids['test'][j], ...],
                                                                batch_size=args.batch_size, pad_with_last_sample=False))
   
    areas = copy.deepcopy(args.area)
    tra = copy.deepcopy(areas)

    for i in range(len(trans_features)):
        if args.transfer_area[i] not in args.area:
            areas.append(args.transfer_area[i])
            data_loaders[args.transfer_area[i]] = [NetworkDataLoader(trans_features[i], trans_labels[i]['data'],
                                                                     trans_adj_labels[i], trans_labels[i]['mask'],
                                                                     batch_size=args.batch_size,
                                                                     pad_with_last_sample=False)]

    ## Model and optimizer
    model = GTEGC(nfeat=features[0].shape[1], landuse_categories=args.num_landuse_categories,
                nhid=args.hidden, dropout=args.dropout, gnn=args.gnn_layer)
   
    if args.cuda:
        model.cuda()

    print('\n'+'Configs', args)

    metric_strings = ['MAE: {:.4f}', 'MAPE: {:.4f}', 'RMSE: {:.4f}',
                      'R2: {:.4f}', 'Spearman: {:.4f}', 'Kendall: {:.4f}']

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print('Start evaluation...', flush=True)
    ts = time.time()

    cluster_results=None

    for i, fname in enumerate(model_state_filenames):
        model.load_state_dict(torch.load(fname))
        trans_outputs = []

        print('Transfer performance for avg. speed prediction for exp {} tid {}'.format(i, tids[i]))
        for j, a in enumerate(areas):
            perms = []
            trans_losses = []
            trans_mae = []
            trans_mape = []
            trans_rmse = []
            trans_r2 = []
            trans_batch_sizes = []
            trans_output = []
            if a in args.area:
                loader = data_loaders[a][i]
            else:
                loader = data_loaders[a][0]

            for iter, (x, y, adj_batch, m, idx) in enumerate(loader.get_iterator()):
                trans_out, trans_loss, trans_metrics = eval_batch(args, model, x, adj_batch, y, m, labels[0]['scaler'],
                                                                  corrs=False) # need to fix corrs
                trans_output.append(trans_out)
                trans_losses.append(trans_loss)
                perms.extend(idx)

                trans_losses.append(trans_loss)
                trans_mae.append(trans_metrics[0])
                trans_mape.append(trans_metrics[1])
                trans_rmse.append(trans_metrics[2])
                trans_r2.append(trans_metrics[3])
                trans_batch_sizes.append(x.size(0))

            perms = np.array(perms)
            trans_output = torch.concat(trans_output, dim=0)
            trans_output = trans_output[np.argsort(perms), ...]
            trans_outputs.append(trans_output.detach().cpu().numpy())

            num_large = len(np.where(np.abs(trans_outputs[-1]) > 200)[0])
            unum_large = len(np.unique(np.where(np.abs(trans_outputs[-1]) > 200)[0]))
            num_null = len(np.unique(np.where(np.isnan(trans_outputs[-1]))[0]))
            n_neg = len(np.where(trans_outputs[-1] < 0)[0])
            print('\t{}\tLoss: {:.2f}\tMAE: {:.2f}\tMAPE: {:.2f}\tRMSE: {:.2f}\tR2: {:.2f} \t#large: {}\tlg unique: {}\t#neg: {}\t#null: {}'.format(areas[j],
                np.mean(trans_losses), np.mean(trans_mae), np.mean(trans_mape) * 100, np.mean(trans_rmse), np.mean(trans_r2),
                num_large, unum_large, n_neg, num_null))

            save_direc = os.path.join(args.model_directory, '.'.join(tra) + '(Train)', args.experiment_name)
            np.save(os.path.join(save_direc, areas[j] + '_preds_' + str(tids[i]) + '.npy'), trans_outputs[-1])

    tid = time.time()
    print('total time spent: {} min {:.2f} sec'.format((tid - ts) // 60, (tid - ts) % 60))

if __name__ == '__main__':
    main()
