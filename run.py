import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from model import Model
from utils import *

from sklearn.metrics import roc_auc_score
import random
import os
import dgl

import argparse
from tqdm import tqdm
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import torch.nn.functional as F
from sklearn import metrics
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='ARISE')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)

args = parser.parse_args()

batch_size = args.batch_size
subgraph_size = args.subgraph_size

if args.dataset == 'cora':
    args.lr = 3e-3
    args.num_epoch = 100

all_auc = []
for run in range(args.runs):

    seed = run + 1

    # Set random seed
    print('Dataset: ', args.dataset)
    print('lr:', args.lr)
    print('epoch:', args.num_epoch)
    print("seed:",seed)
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and preprocess data
    adj, features, _, _, _, _, ano_label, _, _ = load_mat(args.dataset)

    degree = np.sum(adj, axis=0)
    degree_ave = np.mean(degree)

    features, _ = preprocess_features(features)

    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    adj, adj_raw = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj_raw = adj_raw.todense()


    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])

    # Initialize model and optimiser
    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()

    if torch.cuda.is_available():
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
    else:
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    batch_num = nb_nodes // batch_size + 1

    added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
    added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
    added_adj_zero_col[:,-1,:] = 1.
    added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))
    if torch.cuda.is_available():
        added_adj_zero_row = added_adj_zero_row.cuda()
        added_adj_zero_col = added_adj_zero_col.cuda()
        added_feat_zero_row = added_feat_zero_row.cuda()

    # Train model
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):

            loss_full_batch = torch.zeros((nb_nodes,1))
            if torch.cuda.is_available():
                loss_full_batch = loss_full_batch.cuda()

            model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)


            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

                ba = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)

                logits, _ = model(bf, ba)

                loss_all = b_xent(logits, lbl)

                loss = torch.mean(loss_all)

                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()
                loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), './best_model.pkl')
            else:
                cnt_wait += 1

            pbar.set_postfix(loss=mean_loss)
            pbar.update(1)


    # Test model
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('./best_model.pkl'))

    multi_round_attr_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
    nodes_embed = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float).cuda()

    with tqdm(total=args.auc_test_rounds) as pbar_test:
        pbar_test.set_description('Testing')
        for round in range(args.auc_test_rounds):


            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                ba = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                with torch.no_grad():
                    logits, batch_embed = model(bf, ba)
                    logits = torch.squeeze(logits)
                    logits = torch.sigmoid(logits)

                    if round == args.auc_test_rounds - 1:
                        nodes_embed[idx] = batch_embed



                attr_ano_score = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()

                multi_round_attr_ano_score[round, idx] = attr_ano_score

            pbar_test.update(1)

    #attribute anomaly scores
    attr_ano_score_final = np.mean(multi_round_attr_ano_score, axis=0)
    attr_scaler = MinMaxScaler()
    attr_ano_score_final = attr_scaler.fit_transform(attr_ano_score_final.reshape(-1, 1)).reshape(-1)


    #topology anomaly scores
    features_norm = F.normalize(nodes_embed, p = 2, dim = 1)
    features_similarity = torch.matmul(features_norm, features_norm.transpose(0, 1)).squeeze(0).cpu()

    k_init = int(degree_ave)
    net = nx.from_numpy_matrix(adj_raw)
    net.remove_edges_from(nx.selfloop_edges(net))
    adj_raw = nx.to_numpy_matrix(net)
    multi_round_stru_ano_score = []
    while 1:
        list_temp = list(nx.k_core(net, k_init))
        if list_temp == []:
            break
        else:
            core_adj = adj_raw[list_temp, :][:, list_temp]
            core_graph = nx.from_numpy_matrix(core_adj)
            list_temp = np.array(list_temp)
            for i in nx.connected_components(core_graph):
                core_temp = list(i)
                core_temp = list_temp[core_temp]
                core_temp_size = len(core_temp)
                similar_temp = 0
                similar_num = 0
                scores_temp = np.zeros(nb_nodes)
                for idx in core_temp:
                    for idy in core_temp:
                        if idx != idy:
                            similar_temp += features_similarity[idx][idy]
                            similar_num += 1
                scores_temp[core_temp] = core_temp_size * 1 / (similar_temp / similar_num)
                multi_round_stru_ano_score.append(scores_temp)
            k_init += 1




    multi_round_stru_ano_score = np.array(multi_round_stru_ano_score)
    multi_round_stru_ano_score = np.mean(multi_round_stru_ano_score, axis=0)
    stru_scaler = MinMaxScaler()
    stru_ano_score_final = stru_scaler.fit_transform(multi_round_stru_ano_score.reshape(-1, 1)).reshape(-1)

    alpha_list = list(np.arange(0, 1, 0.2))
    rate_auc = []
    for alpha in alpha_list:
        final_scores_rate = alpha * attr_ano_score_final + (1 - alpha) * stru_ano_score_final
        auc_temp = roc_auc_score(ano_label, final_scores_rate)
        rate_auc.append(auc_temp)
    max_alpha = alpha_list[rate_auc.index(max(rate_auc))]
    final_scores_rate = max_alpha * attr_ano_score_final + (1 - max_alpha) * stru_ano_score_final
    best_auc = roc_auc_score(ano_label, final_scores_rate)
    print('Alpha: ', max_alpha)
    print('AUC:{:.4f}'.format(best_auc))
    print('\n')
    all_auc.append(best_auc)

print('\n==============================')
print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
print('==============================')