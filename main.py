from __future__ import division
from __future__ import print_function
import random
import time
import argparse
from zipfile import error

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
import sys

from models_ssl import PCL, run_kmeans, ablation_noTE, GNNmodel, MLPmodel
from utils_ssl import get_A_r, load_dataset, metric, NetworkDataLoader, masked_mae, MultiNetworkDataLoader, extract_subgraph_from_indices, sparse_mx_to_torch_sparse_tensor
from train_ssl import fit_eval_reg, train_batch, eval_batch
from normalization import diff_normalized_adjacency
import warnings
warnings.filterwarnings('ignore')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--area', type=str, default=['Zurich', 'Munich', 'Vienna'], help='Study area')
parser.add_argument('--transfer_area', type=str, default=['Luzern'], help='Study area')
parser.add_argument('--save_foldername', type=str, default='output_path')
parser.add_argument('--transductive', action='store_false')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--gnn_layer', type=str, default='GMLP', help='Choice of GNN layer')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data_directory', type=str, default='processed_data',
                    help='dataset to be used')
parser.add_argument('--network_data', type=str, default='processed_network_250213.csv', help='network dataset')
parser.add_argument('--adj_data', type=str, default='directed_adjacency_matrix.npz',
                    help='data directory for adjacency matrix')
parser.add_argument('--label_data', type=str, default='labels/Hourly_AvgSp.csv',
                    help='data directory for labels')
parser.add_argument('--alpha', type=float, default=1.0, help='To control the ratio of Ncontrast loss')
parser.add_argument('--batch_size', type=int, default=40000, help='batch size') # larger than the size of the network
parser.add_argument('--order', type=int, default=3, help='to compute order-th power of adj')
parser.add_argument('--tau', type=float, default=2.0,  help='temperature for Ncontrast loss')
parser.add_argument('--num_landuse_categories', type=int, default=15,
                    help='the number of land use categories')
parser.add_argument('--null_val', type=float, default=float('nan'), help='null val in labels data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def main():
    ## get data
    adj, features, labels, indices_, trans_adj, trans_features, trans_labels = load_dataset(args, 'DiffNormAdj', True, indiv_norm=False)
    if args.transductive:
        indices = {}
        indices['train'] = []
        indices['val'] = []
        indices['test'] = []
        indices['permutation'] = []
        for a in args.area:
            indices_filename = os.path.join(args.data_directory, a, 'subnetwork', 'indices.pkl')
            with open(indices_filename, 'rb') as f:
                tmp = pickle.load(f)
                indices['train'].extend(tmp['train'])
                indices['val'].extend(tmp['val'])
                indices['test'].extend(tmp['test'])
                indices['permutation'].extend(tmp['permutation'])

    adj_labels = [get_A_r(a, args.order) for a in adj]
    trans_adj_labels = [get_A_r(tadj, args.order) for tadj in trans_adj]

    train_idx = indices['train']
    val_idx = indices['val']
    test_idx = indices['test']

    train_loader = MultiNetworkDataLoader([fmx[train_idx[i], ...] for i, fmx in enumerate(features)],
                                          [lbs['data'][train_idx[i], ...] for i, lbs in enumerate(labels)],
                                          [extract_subgraph_from_indices(a, train_idx[i]) for i, a in enumerate(adj_labels)],
                                          [lbs['mask'][train_idx[i], ...] for i, lbs in enumerate(labels)],
                                          batch_size=args.batch_size, pad_with_last_sample=False)

    valid_loader = MultiNetworkDataLoader([fmx[val_idx[i], ...] for i, fmx in enumerate(features)],
                                          [lbs['data'][val_idx[i], ...] for i, lbs in enumerate(labels)],
                                          [extract_subgraph_from_indices(a, val_idx[i]) for i, a in enumerate(adj_labels)],
                                          [lbs['mask'][val_idx[i], ...] for i, lbs in enumerate(labels)],
                                          batch_size=args.batch_size, pad_with_last_sample=False)

    if args.gnn_layer == "GMLP":
        contrast=True
    else:
        contrast=False

    test_loaders = []
    for i in range(len(features)):
        tloader = NetworkDataLoader(features[i][test_idx[i], ...], labels[i]['data'][test_idx[i], ...],
                                    extract_subgraph_from_indices(adj_labels[i], test_idx[i]),
                                    labels[i]['mask'][test_idx[i], ...],
                                    batch_size=args.batch_size, pad_with_last_sample=False)
        test_loaders.append(tloader)

    transfer_loaders = []
    for idx in range(len(trans_features)):
        transfer_loaders.append(NetworkDataLoader(trans_features[idx], trans_labels[idx]['data'],
                                                  trans_adj_labels[idx], trans_labels[idx]['mask'],
                                                  batch_size=args.batch_size, pad_with_last_sample=False))



    ## Model and optimizer
    model = GTEGC(nfeat=features[0].shape[1], landuse_categories=args.num_landuse_categories,
                nhid=args.hidden, dropout=args.dropout, gnn=args.gnn_layer)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
                           
    if args.cuda:
        model.cuda()

    best_val_mae = 1000
    best_val_mse = 1000
    best_val_r2 = -100
    print('\n'+'training configs', args)

    metric_strings = ['MAE: {:.4f}', 'MAPE: {:.4f}', 'RMSE: {:.4f}',
                      'R2: {:.4f}', 'Spearman: {:.4f}', 'Kendall: {:.4f}']

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print('Start training...', flush=True)
    ts = time.time()
    his_loss = []
    his_ncLoss = []
    his_pcLoss = []
    his_p1loss = []
    his_p2loss = []
    his_p3loss = []
    his_mae = []
    his_mape = []
    his_rmse = []
    his_r2 = []

    his_val_loss = []
    his_val_mae = []
    his_val_mape = []
    his_val_rmse = []
    his_val_r2 = []
    train_time = []
    
    for epoch in range(args.epochs):
        es = time.time()
        train_loader.shuffle()

        train_loss = []
        # train_ncLoss = []
        train_pcLoss = []
        train_p1loss = []
        train_p2loss = []
        train_p3loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_r2 = []

        for iter, (x, y, adj_batch, m, idx, aid) in enumerate(train_loader.get_iterator()):

            losses, metrics = train_batch(args, model, x, adj_batch, optimizer, idx, y, m, labels[aid]['scaler'],
                                          cluster_results=cluster_results, contrast=contrast)

            train_loss.append(losses[0])
            train_ncLoss.append(losses[1])
            
            train_mae.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_r2.append(metrics[3])


        his_mae.append(np.mean(train_mae))
        his_mape.append(np.mean(train_mape))
        his_rmse.append(np.mean(train_rmse))
        his_r2.append(np.mean(train_r2))

        pred_outcome_eval = []
        network_features_eval = []
        perms = []
        for iter, (x, y, adj_batch, m, idx, aid) in enumerate(valid_loader.get_iterator()):
            model.eval()
            val_network, _ = model(x, adj_batch)
            val_network = labels[aid]['scaler'].inverse_transform(val_network)

            pred_outcome_eval.append(val_network)
            perms.extend(idx)

        pred_outcome_eval = torch.concat(pred_outcome_eval, dim=0)
        pred_outcome_eval = pred_outcome_eval[np.argsort(perms), ...]
        val_label = torch.concat([labels[i]['data'][val_idx[i], ...] for i in range(len(args.area))], dim=0)
        val_label = torch.nan_to_num(val_label, nan=0.0)

        val_loss = masked_mae(pred_outcome_eval, val_label, null_val=0.0)

        val_mae, val_mape, val_rmse, val_r2 = metric(pred_outcome_eval, val_label, null_val=0.0, corrs=False)

        his_val_loss.append(val_loss.item())

        his_val_mae.append(val_mae)
        his_val_mape.append(val_mape)
        his_val_rmse.append(val_rmse)
        his_val_r2.append(val_r2)

        ee = time.time()
        train_time.append(ee - es)

        if (epoch == 0) or ((epoch + 1) % 10 == 0):
            txt = ('Regression performance for avg. speed prediction et epoch {} (time spent: {:.2f} sec): Loss: {:.4f} \t val loss: {:.4f}\n'
                   '\tTrain: MAE: {:.4f}\tMAPE: {:.4f}\tRMSE: {:.4f}\tR2: {:.4f}\n'
                   '\tValid: MAE: {:.4f}\tMAPE: {:.4f}\tRMSE: {:.4f}\tR2: {:.4f}'.format(
                epoch + 1, ee - es, np.mean(train_loss), val_loss.item(),
                his_mae[-1], his_mape[-1], his_rmse[-1], his_r2[-1],
                his_val_mae[-1], his_val_mape[-1], his_val_rmse[-1], his_val_r2[-1]))
            print(txt)


        his_loss.append(np.mean(train_loss))
        his_ncLoss.append(np.mean(train_ncLoss))

        if val_loss < best_val_mse:
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_mse = val_loss
            best_epoch = epoch
            best_pred = pred_outcome_eval.detach().cpu().numpy()

            best_val_mae = val_mae
            best_val_mape = val_mape
            best_val_rmse = val_rmse
            best_val_r2 = val_r2

    model.load_state_dict(best_model_state)

    print('Best performance for avg. speed prediction at epoch {}\n'
          '\tValid ({}): Loss: {:.4f}\tMAE: {:.4f}\tMAPE: {:.4f}\tRMSE: {:.4f}\tR2: {:.4f}'.format(
        best_epoch, args.transfer_area[0], best_val_mse, best_val_mae, best_val_mape, best_val_rmse, best_val_r2))

    perms = []
    trans_losses = []
    trans_mae = []
    trans_mape = []
    trans_rmse = []
    trans_r2 = []
    trans_outputs = []

    print('Transfer performance for avg. speed prediction')
    for i in range(len(transfer_loaders)):
        perms = []
        trans_losses = []
        trans_mae = []
        trans_mape = []
        trans_rmse = []
        trans_r2 = []
        trans_batch_sizes = []
        trans_output = []

        for iter, (x, y, adj_batch, m, idx) in enumerate(transfer_loaders[i].get_iterator()):
            trans_out, trans_loss, trans_metrics = eval_batch(args, model, x, adj_batch, y, m, labels[0]['scaler'], corrs=False) # need to fix corrs
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

        print('\t{}\tLoss: {:.4f}\tMAE: {:.4f}\tMAPE: {:.4f}\tRMSE: {:.4f}\tR2: {:.4f}'.format(args.transfer_area[i],
            np.mean(trans_losses), np.mean(trans_mae), np.mean(trans_mape), np.mean(trans_rmse), np.mean(trans_r2)))

    tid = time.time()
    print('total time spent: {} min {:.2f} sec'.format((tid - ts) // 60, (tid - ts) % 60))

    save_directory = os.path.join('results_ssl', '.'.join(args.area) + '(Train)', args.save_foldername)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    if cluster_results is not None:
        with open('results_ssl/' + '.'.join(args.area) + '(Train)/' + args.save_foldername +
                  '/cluster_results_' + str(tid) + '.pkl', 'wb') as f:
            pickle.dump(cluster_results, f)

    loss_table = pd.DataFrame({'epoch': np.arange(args.epochs),
                               'train_loss': his_loss,
                               # 'train_ncloss': his_ncLoss,
                               'train_pcloss': his_pcLoss,
                               'train_p1loss': his_p1loss,
                               'train_p2loss': his_p2loss,
                               # 'train_p3loss': his_p3loss,
                               'train_mae': his_mae,
                               'train_mape': his_mape,
                               'train_rmse': his_rmse,
                               'train_r2': his_r2,
                               'valid_mae': his_val_mae,
                               'valid_mape': his_val_mape,
                               'valid_rmse': his_val_rmse,
                               'valid_r2': his_val_r2
                               })
    
    loss_table.to_csv(os.path.join(save_directory, 'loss_table_' + str(tid) + '.csv'), index=False)

    
    np.save(os.path.join(save_directory, 'predictions_' + str(tid) + '.npy'), best_pred)
    for i, tr_out in enumerate(trans_outputs):
        np.save(os.path.join(save_directory, args.transfer_area[i] + '_preds_' + str(tid) + '.npy'), tr_out)
    
    with open(os.path.join(save_directory, 'indices_' + str(tid) + '.pkl'), 'wb') as f:
        pickle.dump(indices, f)

    torch.save(model.state_dict(), os.path.join(save_directory, 'best_model_state_' + str(tid) + '.pth'))


if __name__ == '__main__':

    for i in range(5):
        main()
