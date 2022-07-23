import random

import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from utils import *
from model import MNGACDA
import json


def main(edge_idx_dict, n_drug, n_cir, drug_sim, cir_sim, args_config, device):
    lr = args_config['lr']
    weight_decay = args_config['weight_decay']
    kfolds = args_config['kfolds']
    residual = args_config['residual']
    add_layer_attn = args_config['add_layer_attn']
    num_epoch = args_config['num_epoch']
    knn_nums = args_config['knn_nums']
    num_hidden_layers = args_config['num_hidden_layers']
    num_heads_per_layer = [args_config['num_heads_per_layer'] for _ in range(num_hidden_layers)]
    num_embedding_features = [args_config['num_embedding_features'] for _ in range(num_hidden_layers)]
    pos_edges = edge_idx_dict['pos_edges']
    neg_edges = edge_idx_dict['neg_edges']
    metrics_tensor = np.zeros((1, 7))
    temp_drug_cir = np.zeros((n_drug, n_cir))
    temp_drug_cir[pos_edges[0], pos_edges[1]] = 1
    drug_sim, cir_sim = get_syn_sim(temp_drug_cir, drug_sim, cir_sim, 1)
    drug_adj = k_matrix(drug_sim, knn_nums)
    cir_adj = k_matrix(cir_sim, knn_nums)
    edge_idx_drug, edge_idx_cir = np.array(tuple(np.where(drug_adj != 0))), np.array(tuple(np.where(cir_adj != 0)))
    edge_idx_drug = torch.tensor(edge_idx_drug, dtype=torch.long, device=device)
    edge_idx_cir = torch.tensor(edge_idx_cir, dtype=torch.long, device=device)
    model = MNGACDA(
        n_drug + n_cir, num_hidden_layers, num_embedding_features, num_heads_per_layer,
        n_drug, n_cir, add_layer_attn, residual).to(device)
    num_u, num_v = n_drug, n_cir
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    base_lr = 5e-5
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr=lr, step_size_up=200,
                                            step_size_down=200, mode='exp_range', gamma=0.99, scale_fn=None,
                                            cycle_momentum=False, last_epoch=-1)
    het_mat = construct_het_mat(temp_drug_cir, cir_sim, drug_sim)
    adj_mat = construct_adj_mat(temp_drug_cir)
    drug_sim = torch.tensor(drug_sim, dtype=torch.float, device=device)
    cir_sim = torch.tensor(cir_sim, dtype=torch.float, device=device)
    edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
    het_mat_device = torch.tensor(het_mat, dtype=torch.float32, device=device)
    for epoch in range(num_epoch):
        model.train()
        pred_mat = model(het_mat_device, edge_idx_device, drug_sim, edge_idx_drug, cir_sim, edge_idx_cir).cpu().reshape(
            num_u, num_v)
        loss = calculate_loss(pred_mat, pos_edges, neg_edges)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print('------EOPCH {} of {}------'.format(epoch + 1, args_config['num_epoch']))
            print('Loss: {}'.format(loss))
    model.eval()
    with torch.no_grad():
        pred_mat = model(het_mat_device, edge_idx_device, drug_sim, edge_idx_drug, cir_sim,
                         edge_idx_cir).cpu().reshape(num_u, num_v)
        metrics = calculate_evaluation_metrics(pred_mat.detach(), pos_edges, neg_edges)
        return pred_mat


if __name__ == '__main__':
    set_seed(666)
    repeat_times = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyperparam_dict = {
        'kfolds':5,
        'num_heads_per_layer': 3,
        'num_embedding_features':128,
        'num_hidden_layers': 2,
        'num_epoch': 1000,
        'knn_nums':25,
        'lr': 1e-3,
        'weight_decay': 5e-3,
        'add_layer_attn': True,
        'residual': True,
    }

    for i in range(repeat_times):
        print(f'********************{i + 1} of {repeat_times}********************')
        drug_sim, cir_sim, edge_idx_dict, drug_dis_matrix = load_data()
        diag = np.diag(cir_sim)
        if np.sum(diag) != 0:
            cir_sim = cir_sim - np.diag(diag)
        diag = np.diag(drug_sim)
        if np.sum(diag) != 0:
            drug_sim = drug_sim - np.diag(diag)
        pred_mat=main(edge_idx_dict, drug_sim.shape[0], cir_sim.shape[0], drug_sim, cir_sim, hyperparam_dict, device)