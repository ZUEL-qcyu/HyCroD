"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.datasets as dt
import torch_geometric.transforms as T
import torch_geometric.utils as tu
from torch_geometric.utils.convert import to_networkx
from utils.polblogs import PolBlogs
from ogb.nodeproppred import PygNodePropPredDataset



def load_data(args, datapath=None): 
    #if isinstance(datapath, str):
    # data, labels, idx_train, idx_val, idx_test = load_data_lp(args.dataset, args.use_feats, datapath)
    if args.dataset in ['amazon','coauthor','ogbn-arxiv','arxiv-year']:
        data = load_other_data(args.dataset,'data/',args.use_feats,args.seed)
    else:
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.seed)
    #else:
    #    data = {'adj_train': sp.coo_matrix(datapath['adj'].squeeze(0).numpy()), 'features': datapath['features'].squeeze(0).numpy()}
        
    # adj = data['adj_train']
    # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
    #         adj, args.val_prop, args.test_prop, args.split_seed
    # )
    # data['adj_train'] = adj_train
    # data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
    # data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
    # data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    #
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data

# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed','citeseer','PolBlogs']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data1(dataset, use_feats, data_path)
    elif dataset == 'disease_lp' or dataset == 'BA' or dataset == 'sbm':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data, labels, idx_train, idx_val, idx_test

def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora','citeseer', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data1(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'amazon':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data

# ############### DATA SPLITS #####################################################

def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  

def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


# ############### FEATURES PROCESSING ####################################

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATASETS ####################################
def load_citation_data1(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

def load_other_data(data_name, data_path, use_feats,split_seed=None):
    flag = 0
    year = 0
    # 加载数据集
    if data_name == 'amazon':
        path = os.path.join(data_path, 'Amazon')
        dataset = dt.Amazon(path, name='Photo')
    elif data_name in ['ogbn-arxiv','arxiv-year']:
        path = os.path.join(data_path, 'ogbn_arxiv')
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=path)
        flag = 1
        if data_name == 'arxiv-year':
            year = 1
    else:
        path = os.path.join(data_path, 'Coauthor')
        dataset = dt.Coauthor(path, name='CS')

    data = dataset[0]

    # 提取邻接矩阵
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    adj = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))

    # 提取特征矩阵
    features = data.x.numpy()
    features = sp.csr_matrix(features)

    # 如果不使用特征，则用单位矩阵代替
    if not use_feats:
        features = sp.eye(adj.shape[0])

    # 提取标签
    labels = data.y.numpy()
    labels = torch.LongTensor(labels)
    print(labels.shape)
    if flag == 1:
        labels = labels.squeeze(-1)
        if year == 1:
            labels = even_quantile_labels(labels.numpy(), 5, verbose=False)
            labels = torch.LongTensor(labels)

    # 划分训练集、验证集和测试集

    num_classes = len(np.unique(labels))
    idx_train = []

    # 遍历每个类别
    for class_id in range(num_classes):
        # 获取当前类别的所有样本索引
        class_indices = np.where(labels[:-1000] == class_id)[0]

        # 如果当前类别的样本数少于 20，则选择所有样本
        if len(class_indices) < 20:
            idx_train.extend(class_indices)
        else:
            # 随机选择 20 个样本
            idx_train.extend(np.random.choice(class_indices, size=20, replace=False))

    idx_test = np.arange(num_nodes - 1000 ,num_nodes)
    remaining_indices = np.setdiff1d(np.arange(num_nodes - 1000), idx_train)
    idx_val = np.random.choice(remaining_indices, size=500, replace=False)
    # 转换为列表
    idx_train = idx_train
    idx_val = idx_val.tolist()
    idx_test = idx_test.tolist()

    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val,
            'idx_test': idx_test}
    return data

def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    if dataset_str in ['citeseer','cora']:
        if dataset_str=='citeseer':
            d_name = 'Citeseer'
        elif dataset_str == 'cora':
            d_name = 'Cora'
        dataset = dt.Planetoid(root='data/'+d_name, name=d_name, transform=T.NormalizeFeatures())
    if dataset_str == 'PolBlogs':
        dataset = PolBlogs(root='data/'+dataset_str) 
    data = dataset.data
    graph = to_networkx(data)
    adj = nx.adjacency_matrix(graph)
    # print(data)
    # print(data.x)
    labels = data.y
    if (data.x!=None):
        features = data.x
        idx_train = data.train_mask.nonzero().t()[0]
        idx_val = data.val_mask.nonzero().t()[0]
        idx_test = data.test_mask.nonzero().t()[0]
    else:
        val_prop, test_prop = 0.1, 0.15
        idx_train, idx_val, idx_test = split_data(labels.numpy(), val_prop, test_prop, seed=split_seed)
        
    features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def load_synthetic_data(dataset_str, use_feats, data_path):
    if dataset_str == 'BA' or dataset_str == 'sbm':
        adj = torch.load(os.path.join(data_path, "{}_edge_index.pt".format(dataset_str)))
        
        adj = tu.to_scipy_sparse_matrix(adj)
        features = torch.load(os.path.join(data_path, "{}_features.pt".format(dataset_str)))
        labels = torch.load(os.path.join(data_path, "{}_labels.pt".format(dataset_str)))
        
    else:
        object_to_idx = {}
        idx_counter = 0
        edges = []
        with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
            all_edges = f.readlines()
        for line in all_edges:
            n1, n2 = line.rstrip().split(',')
            if n1 in object_to_idx:
                i = object_to_idx[n1]
            else:
                i = idx_counter
                object_to_idx[n1] = i
                idx_counter += 1
            if n2 in object_to_idx:
                j = object_to_idx[n2]
            else:
                j = idx_counter
                object_to_idx[n2] = j
                idx_counter += 1
            edges.append((i, j))
        adj = np.zeros((len(object_to_idx), len(object_to_idx)))
        for i, j in edges:
            adj[i, j] = 1.  # comment this line for directed adjacency matrix
            adj[j, i] = 1.
        if use_feats:
            features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
        else:
            features = sp.eye(adj.shape[0])
        labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    print(features)
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()
