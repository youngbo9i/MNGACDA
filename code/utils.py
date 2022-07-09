import numpy as np
from scipy.sparse import coo_matrix
import random
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling
import torch
import pandas as pd
import os
def set_seed(seed):
    torch.manual_seed(seed)
    #进行随机搜索的这个要注释掉
    # random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def get_syn_sim(A, seq_sim, str_sim, mode):
    """

    :param A:
    :param seq_sim:
    :param str_sim:
    :param mode: 0 = GIP kernel sim
    :return:
    """
    GIP_c_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)


    if mode == 0:
        return GIP_c_sim, GIP_d_sim

    syn_c = np.zeros((A.shape[0], A.shape[0]))
    syn_d = np.zeros((A.shape[1], A.shape[1]))

    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if seq_sim[i, j] == 0:
                syn_c[i, j] = GIP_c_sim[i, j]
            else:
                syn_c[i, j] = (GIP_c_sim[i, j] + seq_sim[i, j]) / 2

    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            if str_sim[i, j] == 0:
                syn_d[i, j] = GIP_d_sim[i, j]
            else:
                syn_d[i, j] = (GIP_d_sim[i, j] + str_sim[i, j]) / 2
    return syn_c, syn_d

def GIP_kernel(asso_drug_cir):
    # the number of row
    nc = asso_drug_cir.shape[0]
    # initate a matrix as results matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(asso_drug_cir)
    # calculate the results matrix
    for i in range(nc):
        for j in range(nc):
            # calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(asso_drug_cir[i, :] - asso_drug_cir[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix

def getGosiR(asso_drug_cir):
    # calculate the r in GOsi Kerel
    nc = asso_drug_cir.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(asso_drug_cir[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r
def GIP_kernel(asso_drug_cir):
    # the number of row
    nc = asso_drug_cir.shape[0]
    # initate a matrix as results matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(asso_drug_cir)
    # calculate the results matrix
    for i in range(nc):
        for j in range(nc):
            # calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(asso_drug_cir[i, :] - asso_drug_cir[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix

def k_matrix(matrix, k=20):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        #将第i行最大的前k个值赋值给knn_graph(确保是对称矩阵)
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def random_index(index_matrix, k_fold):
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.shuffle(random_index)
    k_folds = k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp


def crossval_index(drug_mic_matrix, k_flod):
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    neg_index = np.random.choice(neg_index_matrix.shape[1], pos_index_matrix.shape[1], replace=False)
    neg_index_matrix = neg_index_matrix[:, neg_index]
    pos_index = random_index(neg_index_matrix, k_flod)
    neg_index = random_index(pos_index_matrix,k_flod)
    index = [pos_index[i] + neg_index[i] for i in range(k_flod)]
    return index

def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    # np.savetxt(roc_path.format(i), ROC_dot_matrix)

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    # np.savetxt(pr_path.format(i), PR_dot_matrix)

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)
    # plt.plot(x_ROC, y_ROC)
    # plt.plot(x_PR,y_PR)
    # plt.show()
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print( ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format( auc[0, 0],aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]


def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
    test_index = np.where(train_matrix == 0)
    real_score = interaction_matrix[test_index]
    predict_score = predict_matrix[test_index]
    return get_metrics(real_score, predict_score)


# turn dense matrix into a sparse foramt
def dense2sparse(matrix: np.ndarray):
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data

def construct_het_mat(rna_dis_mat, dis_mat, rna_mat):
    mat1 = np.hstack((rna_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret

def construct_adj_mat(training_mask):
    adj_tmp = training_mask.copy()
    rna_mat = np.zeros((training_mask.shape[0], training_mask.shape[0]))
    dis_mat = np.zeros((training_mask.shape[1], training_mask.shape[1]))

    mat1 = np.hstack((rna_mat, adj_tmp))
    mat2 = np.hstack((adj_tmp.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret

def calculate_loss(pred, pos_edge_idx, neg_edge_idx):
    pos_pred_socres = pred[pos_edge_idx[0], pos_edge_idx[1]]
    neg_pred_socres = pred[neg_edge_idx[0], neg_edge_idx[1]]
    pred_scores = torch.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = torch.hstack((torch.ones(pos_pred_socres.shape[0]), torch.zeros(neg_pred_socres.shape[0])))
    loss_fun=torch.nn.BCELoss(reduction='mean')
    # loss_fun=torch.nn.BCEWithLogitsLoss(reduction='mean')
    return loss_fun(pred_scores, true_labels)

def calculate_evaluation_metrics(pred_mat, pos_edges, neg_edges):
    pos_pred_socres = pred_mat[pos_edges[0], pos_edges[1]]
    neg_pred_socres = pred_mat[neg_edges[0], neg_edges[1]]
    pred_labels = np.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = np.hstack((np.ones(pos_pred_socres.shape[0]), np.zeros(neg_pred_socres.shape[0])))
    return get_metrics(true_labels, pred_labels)

def load_data():
    cir_sim = pd.read_csv("../data/gene_seq_sim.csv", index_col=0, dtype=np.float32).to_numpy()
    drug_sim=pd.read_csv("../data/drug_str_sim.csv", index_col=0, dtype=np.float32).to_numpy()
    drug_cir_ass =pd.read_csv("../data/association.csv", index_col=0).to_numpy().T
    diag = np.diag(cir_sim)
    if np.sum(diag) != 0:
        cir_sim = cir_sim - np.diag(diag)

    # get the edge idx of positives samplese
    rng = np.random.default_rng(1)
    pos_samples, edge_attr = dense2sparse(drug_cir_ass)
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    # get the edge index of negative samples
    rng = np.random.default_rng(1)
    neg_samples = np.where(drug_cir_ass == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]
    # split positive samples into training message samples, training supervision samples, test samples
    edge_idx_dict = dict()
    edge_idx_dict['pos_edges'] = pos_samples_shuffled
    edge_idx_dict['neg_edges'] = neg_samples_shuffled

    return drug_sim,cir_sim, edge_idx_dict,drug_cir_ass