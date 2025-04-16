import os

import numpy as np
import random
import torch
import pandas as pd
import dgl
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import torch as th
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, roc_curve, auc

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

device = torch.device('cuda')


def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj


def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def get_data(args):
    data = dict()

    # drf = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
    # drg = pd.read_csv(args.data_dir + 'DrugGIP.csv').iloc[:, 1:].to_numpy()
    #
    # dip = pd.read_csv(args.data_dir + 'DiseasePS.csv').iloc[:, 1:].to_numpy()
    # dig = pd.read_csv(args.data_dir + 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()
    adj = pd.read_csv(args.data_dir + 'adj.csv', index_col=0).to_numpy()
    drs = pd.read_csv(args.data_dir + 'Rwmeta_t.csv', header=None).to_numpy()
    dis = pd.read_csv(args.data_dir + 'Rwdis_t.csv', header=None).to_numpy()

    # drs = pd.read_csv(args.data_dir + 'average_metabolite_similarity.csv', header=None).to_numpy()
    # dis = pd.read_csv(args.data_dir + 'average_disease_similarity.csv', header=None).to_numpy()

    data['drug_number'] = int(drs.shape[0])
    data['disease_number'] = int(dis.shape[0])

    # data['drf'] = drf
    # data['drg'] = drg
    # data['dip'] = dip
    # data['dig'] = dig
    data['adj'] = adj

    data['drdi'] = pd.read_csv(args.data_dir + 'MetaDiseaseAssociationNumber.csv', dtype=int).to_numpy()
    # data['drpr'] = pd.read_csv(args.data_dir + 'DrugProteinAssociationNumber.csv', dtype=int).to_numpy()
    # data['dipr'] = pd.read_csv(args.data_dir + 'ProteinDiseaseAssociationNumber.csv', dtype=int).to_numpy()
    #
    # data['drugfeature'] = pd.read_csv(args.data_dir + 'Drug_mol2vec.csv', header=None).iloc[:, 1:].to_numpy()
    # data['diseasefeature'] = pd.read_csv(args.data_dir + 'DiseaseFeature.csv', header=None).iloc[:, 1:].to_numpy()
    # data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM.csv', header=None).iloc[:, 1:].to_numpy()
    # data['protein_number'] = data['proteinfeature'].shape[0]

    return data


def data_processing(data, args):
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    unsamples = zero_index[int(args.negative_rate * len(one_index)):]
    data['unsample'] = np.array(unsamples)

    zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    index = np.array(one_index + zero_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]

    # drs_mean = (data['drf'] + data['drg']) / 2
    # dis_mean = (data['dip'] + data['dig']) / 2
    #
    # drs = np.where(data['drf'] == 0, data['drg'], drs_mean)
    # dis = np.where(data['dip'] == 0, data['dip'], dis_mean)
    drs = pd.read_csv(args.data_dir + 'Rwmeta_t.csv', header=None).to_numpy()
    dis = pd.read_csv(args.data_dir + 'Rwdis_t.csv', header=None).to_numpy()

    # drs = pd.read_csv(args.data_dir + 'average_metabolite_similarity.csv', header=None).to_numpy()
    # dis = pd.read_csv(args.data_dir + 'average_disease_similarity.csv', header=None).to_numpy()

    data['drs'] = drs
    data['dis'] = dis
    data['all_samples'] = samples
    data['all_drdi'] = samples[:, :2]
    data['all_drdi_p'] = drdi_p
    data['all_drdi_n'] = drdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p

    return data


def k_fold(data, args):
    k = args.k_fold
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    X = data['all_drdi']
    Y = data['all_label']
    # n = skf.get_n_splits(X, Y)
    X_train_all, X_test_all, Y_train_all, Y_test_all = [], [], [], []
    for train_index, test_index in skf.split(X, Y):
        # print('Train:', train_index, 'Test:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    for i in range(k):
        X_train1 = pd.DataFrame(data=np.concatenate((X_train_all[i], Y_train_all[i]), axis=1),
                                columns=['drug', 'disease', 'label'])
        X_train1.to_csv(args.data_dir + 'fold/' + str(i) + '/data_train.csv')
        X_test1 = pd.DataFrame(data=np.concatenate((X_test_all[i], Y_test_all[i]), axis=1),
                               columns=['drug', 'disease', 'label'])
        X_test1.to_csv(args.data_dir + 'fold/' + str(i) + '/data_test.csv')

    data['X_train'] = X_train_all
    data['X_test'] = X_test_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    return data


def dgl_similarity_graph(data, args):
    drdr_matrix = k_matrix(data['drs'], args.neighbor)
    didi_matrix = k_matrix(data['dis'], args.neighbor)
    drdr_nx = nx.from_numpy_array(drdr_matrix)
    didi_nx = nx.from_numpy_array(didi_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['drs'])
    didi_graph.ndata['dis'] = torch.tensor(data['dis'])

    return drdr_graph, didi_graph, data


def dgl_heterograph(data, drdi, args):
    drdi_list, drpr_list, dipr_list = [], [], []
    for i in range(drdi.shape[0]):
        drdi_list.append(drdi[i])
    for i in range(data['drpr'].shape[0]):
        drpr_list.append(data['drpr'][i])
    for i in range(data['dipr'].shape[0]):
        dipr_list.append(data['dipr'][i])

    node_dict = {
        'drug': args.drug_number,
        'disease': args.disease_number,
        'protein': args.protein_number
    }

    heterograph_dict = {
        ('drug', 'association', 'disease'): (drdi_list),
        ('drug', 'association', 'protein'): (drpr_list),
        ('disease', 'association', 'protein'): (dipr_list)
    }

    data['feature_dict'] = {
        'drug': torch.tensor(data['drugfeature']),
        'disease': torch.tensor(data['diseasefeature']),
        'protein': torch.tensor(data['proteinfeature'])
    }

    drdipr_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)

    return drdipr_graph, data


def drdi_heterograph(data, drdi, args):
    drdi_list = []
    for i in range(drdi.shape[0]):
        drdi_list.append(drdi[i])

    node_dict = {
        'drug': args.drug_number,
        'disease': args.disease_number,
    }

    heterograph_dict = {
        ('drug', 'association', 'disease'): (drdi_list),

    }

    data['feature_dict'] = {
        'drug': torch.tensor(data['drs']),
        'disease': torch.tensor(data['dis']),
    }

    drdi_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)
    drdi_graph.ndata['h'] = data['feature_dict']

    return drdi_graph, data


def construct_adj_mat(training_mask):
    adj_tmp = training_mask.copy()
    rna_mat = np.zeros((training_mask.shape[0], training_mask.shape[0]))
    dis_mat = np.zeros((training_mask.shape[1], training_mask.shape[1]))

    mat1 = np.hstack((rna_mat, adj_tmp))
    mat2 = np.hstack((adj_tmp.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret


def hero(embd1, embd2, distance):
    beta, alpha = 1, 1
    coe2 = 1.0 / beta
    res = torch.mm(torch.transpose(embd1, 0, 1), embd1)  # H.T* H
    inv = torch.inverse(torch.eye(embd1.shape[1]).to(device) + coe2 * res)  # Q中的逆矩阵
    res = torch.mm(inv, res)  # B中第二项的后面一部分
    res = coe2 * embd1 - coe2 * coe2 * torch.mm(embd1, res)  # B
    tmp = torch.mm(torch.transpose(embd1, 0, 1), res)  # H.T * B
    part1 = torch.mm(embd1, tmp)

    part2 = (- alpha / 2) * torch.mm(distance, res)  # / self.args.alpha
    embs_all = part1 + part2
    embs_hom = embs_all
    embs_het = embd2
    return embs_het, embs_hom


def pairwise_distance(x, y=None):
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1)  # [B, N, f]
    A = -2 * th.bmm(y, x)  # [B, N, N]
    A += th.sum(y ** 2, dim=2, keepdim=True)  # [B, N, 1]
    A += th.sum(x ** 2, dim=1, keepdim=True)  # [B, 1, N]
    return A.squeeze()


def get_edge_index_torch(matrix):
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)

    non_zero = torch.nonzero(matrix, as_tuple=True)
    edge_index = torch.stack(non_zero)

    return edge_index


def to_hyperboloid(x, c=1.0):
    """
    将欧几里得空间的向量投影到超双曲模型上
    x: 输入向量 (batch_size, dim)
    c: 双曲空间的曲率（默认为1）
    """
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)

    # 将 c 转换为与 x 相同设备和数据类型的标量张量
    c_tensor = torch.tensor(c, dtype=x.dtype, device=x.device)

    return torch.cat([torch.sqrt(1 + c_tensor * x_norm ** 2), torch.sqrt(c_tensor) * x], dim=-1)


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def contrastive_loss(h1, h2, tau=0.7):
    sim_matrix = sim(h1, h2)
    f = lambda x: torch.exp(x / tau)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag()
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()
    return loss


def plot_roc_curves(root_path,fold_results):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, 5))

    mean_fpr = np.linspace(0, 1, 5064)
    tprs = []
    aucs = []

    for i, (y_true, y_pred_proba) in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, color=colors[i], lw=1,
                 label=f'Fold {i + 1} (AUC = {roc_auc:.4f})')

    # 计算并绘制平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # np.save(os.path.join('results', 'GHTMDA', 'mean_fpr.npy'), mean_fpr)
    # np.save(os.path.join('results',  'GHTMDA', 'mean_tpr.npy'), mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='navy', lw=2, linestyle='--',
             label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')

    plt.plot([0, 1], [0, 1], linestyle=':', lw=2, color='r')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    save_path = os.path.join(root_path, 'roc_curves.png')
    # plt.savefig(os.path.join('results', 'test', 'roc_curves.png'))
    plt.savefig(save_path)
    plt.close()

def plot_pr_curves(root_path,fold_results):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, 5))

    mean_recall = np.linspace(0, 1, 8392)
    precisions = []
    aucs = []

    for i, (y_true, y_pred_proba) in enumerate(fold_results):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
        pr_auc = average_precision_score(y_true, y_pred_proba)
        aucs.append(pr_auc)
        plt.plot(recall, precision, color=colors[i], lw=1,
                 label=f'Fold {i + 1} (AUPR = {pr_auc:.4f})')

    mean_precision = np.mean(precisions, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(mean_recall, mean_precision, color='navy', lw=2, linestyle='--',
             label=f'Mean (AUPR = {round(mean_auc, 4):.4f} ± {round(std_auc, 4):.4f})')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    save_path = os.path.join(root_path, 'pr_curves.png')
    # plt.savefig(os.path.join('results', 'test', 'pr_curves.png'))
    # 保存图片
    plt.savefig(save_path)
    plt.close()


def plot_combined_curves(root_path,fold_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # ROC Curve
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, 5))
    mean_fpr = np.linspace(0, 1, 8392)
    tprs = []
    roc_aucs = []

    for i, (y_true, y_pred_proba) in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        ax1.plot(fpr, tpr, color=colors[i], lw=1,
                 label=f'Fold {i + 1} (AUC = {roc_auc:.4f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = auc(mean_fpr, mean_tpr)
    std_roc_auc = np.std(roc_aucs)

    ax1.plot(mean_fpr, mean_tpr, color='navy', lw=2, linestyle='--',
             label=f'Mean (AUC = {round(mean_roc_auc, 4):.4f} ± {round(std_roc_auc, 4):.4f})')
    ax1.plot([0, 1], [0, 1], linestyle=':', lw=2, color='r')
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")

    # Precision-Recall Curve
    mean_recall = np.linspace(0, 1, 8392)
    precisions = []
    pr_aucs = []

    for i, (y_true, y_pred_proba) in enumerate(fold_results):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
        pr_auc = average_precision_score(y_true, y_pred_proba)
        pr_aucs.append(pr_auc)
        ax2.plot(recall, precision, color=colors[i], lw=1,
                 label=f'Fold {i + 1} (AUPR = {pr_auc:.4f})')

    mean_precision = np.mean(precisions, axis=0)
    mean_pr_auc = np.mean(pr_aucs)
    std_pr_auc = np.std(pr_aucs)

    ax2.plot(mean_recall, mean_precision, color='navy', lw=2, linestyle='--',
             label=f'Mean (AUPR = {round(mean_pr_auc, 4):.4f} ± {round(std_pr_auc, 4):.4f})')
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")

    plt.tight_layout()

    save_path = os.path.join(root_path, 'combined_curves.png')
    # plt.savefig(os.path.join('results', 'test', 'combined_curves.png'))
    plt.savefig(save_path)
    plt.close()