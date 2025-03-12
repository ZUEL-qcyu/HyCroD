# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
import random
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from utils import process
from utils import data_utils
from utils import aug
from modules.gcn import GCNLayer
from net.mview import agg_contrastive_loss,HypMVIEW
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,precision_score, recall_score, f1_score
from nc_train import train
from diff import hyperdiff

from manifolds.hyperboloid import Hyperboloid


os.environ['LOG_DIR'] = 'logs/'
os.environ['DATAPATH'] = 'data/'
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--dataset',default='cora',choices=\
    ['cora','citeseer','pubmed','amazon','coauthor','ogbn-arxiv','arxiv-year'])
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sample_size', type=int, default=2000)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--sparse', type=str_to_bool, default=True)

parser.add_argument('--input_dim', type=int, default=1433)##
parser.add_argument('--gnn_dim', type=int, default=512)
parser.add_argument('--proj_dim', type=int, default=512)
parser.add_argument('--proj_hid', type=int, default=4096)
parser.add_argument('--pred_dim', type=int, default=512)
parser.add_argument('--pred_hid', type=int, default=4096)
parser.add_argument('--momentum', type=float, default=0.7)
parser.add_argument('--beta', type=float, default=0.3)

parser.add_argument('--drop_edge', type=float, default=0.15)
parser.add_argument('--drop_feat1', type=float, default=0.35)
parser.add_argument('--drop_feat2', type=float, default=0.35)
#diffusion args
parser.add_argument('--diff_epoc',type=int,default=150)
parser.add_argument('--c',type=float,default=0.6)
parser.add_argument('--r',type=float,default=2.)
parser.add_argument('--t',type=float,default=1.)
parser.add_argument('--model',type=str,default='HGCN')
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--cuda',type=int,default=0)
parser.add_argument('--enco_epoch',type=int,default=250)
parser.add_argument('--weight-decay',type=float,default=0.00001)
parser.add_argument('--optimizer',type=str,default='Adam')
# parser.add_argument('--momentum',type=float,default=0.99)
#parser.add_argument('--seed',type=int,default=1432)
parser.add_argument('--log-freq',type=int,default=5)
parser.add_argument('--eval-freq',type=int,default=1)
parser.add_argument('--save-dir',type=str,default=None)
parser.add_argument('--sweep-c',default=0)
parser.add_argument('--lr-reduce-freq',default=None)
parser.add_argument('--gamma',default=0.2)
parser.add_argument('--print-epoch',default=True)
parser.add_argument('--grad-clip',default=10)
parser.add_argument('--min-epochs',default=None)
parser.add_argument('--dim',default=512)
parser.add_argument('--manifold',default='PoincareBall')
parser.add_argument('--hid2',default=32)
parser.add_argument('--hid1',default=64)
parser.add_argument('--pos-weight',default=0)
parser.add_argument('--num-layers',default=2)
parser.add_argument('--bias',default=0)
parser.add_argument('--act',default='relu')
parser.add_argument('--n-heads',default=4)
parser.add_argument('--alpha',default=0.6)
parser.add_argument('--double-precision',default=0)
parser.add_argument('--use-att',default=0)
parser.add_argument('--local-agg',default=0)

parser.add_argument('--val-prop',default=0.05)
parser.add_argument('--test-prop',default=0.1)
parser.add_argument('--use-feats',default=1)
parser.add_argument('--normalize-feats',default=1)
parser.add_argument('--normalize-adj',default=1)
parser.add_argument('--split-seed',default=1)

args = parser.parse_args()
torch.set_num_threads(4)

def evaluation(adj, num_classes, feat, gnn, idx_train, idx_test, sparse):
    clf = LogisticRegression(random_state=0, max_iter=2000)
    model = GCNLayer(input_size, gnn_output_size)  # 1-layer
    model.load_state_dict(gnn.state_dict())
    with torch.no_grad():
        embeds1 = model(feat, adj, sparse)
        train_embs = embeds1[0, idx_train] #+ embeds2[0, idx_train]
        test_embs = embeds1[0, idx_test] #+ embeds2[0, idx_test]
        train_labels = labels[0, idx_train]
        test_labels = labels[0, idx_test]

    clf.fit(train_embs, train_labels)
    pred_test_labels = clf.predict(test_embs)
    acc = accuracy_score(test_labels, pred_test_labels)
    prec = precision_score(test_labels, pred_test_labels, average='macro')
    rec = recall_score(test_labels, pred_test_labels, average='macro')
    f1 = f1_score(test_labels, pred_test_labels, average='macro')

    return acc,prec,rec,f1


if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    n_runs = args.runs
    eval_every_epoch = args.eval_every

    dataset = args.dataset
    input_size = args.input_dim

    gnn_output_size = args.gnn_dim
    projection_size = args.proj_dim
    projection_hidden_size = args.proj_hid
    prediction_size = args.pred_dim
    prediction_hidden_size = args.pred_hid
    momentum = args.momentum
    beta = args.beta

    drop_edge_rate_1 = args.drop_edge
    drop_feature_rate_1 = args.drop_feat1
    drop_feature_rate_2 = args.drop_feat2

    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    sample_size = args.sample_size
    batch_size = args.batch_size
    patience = args.patience
    sparse = args.sparse
    # Loading dataset
    #adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    data = data_utils.load_data(args, os.path.join(os.environ['DATAPATH']))

    idx_train = torch.LongTensor(data['idx_train'])
    idx_val = torch.LongTensor(data['idx_val'])
    idx_test = torch.LongTensor(data['idx_test'])


    #features, _ = process.preprocess_features(features)
    features = data['features']

    nb_nodes = features.shape[0]

    ft_size = features.shape[1]

    features = torch.FloatTensor(features[np.newaxis])  ##升维
    labels = torch.FloatTensor(data['labels'].detach().cpu().numpy()[np.newaxis])

    nb_classes = labels.shape[1]

    train(args,data)

    diff_embedding = hyperdiff(args,data).to(torch.float32)

    # save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    # file_path = os.path.join(save_dir, 'embeddings.npy')
    # diff_embedding = torch.FloatTensor(np.load(file_path)).to(device)

    #diff_features = torch.FloatTensor(diff_features.detach().cpu().numpy()[np.newaxis])

    adj = data['adj_train']
    norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    #norm_adj = adj

    # norm_diff = sp.csr_matrix(diff)
    if sparse:
        eval_adj = process.sparse_mx_to_torch_sparse_tensor(norm_adj)
        #eval_diff = process.sparse_mx_to_torch_sparse_tensor(norm_diff)
    else:
        eval_adj = (norm_adj + sp.eye(norm_adj.shape[0])).todense()
        #eval_diff = (norm_diff + sp.eye(norm_diff.shape[0])).todense()
        eval_adj = torch.FloatTensor(eval_adj[np.newaxis])
        #eval_diff = torch.FloatTensor(eval_diff[np.newaxis])

    result_over_runs = []
    input_size = ft_size
    # Initiate models
    model = GCNLayer(input_size, gnn_output_size)
    hyp_mview = HypMVIEW(gnn=model,
                  feat_size=input_size,
                  projection_size=projection_size,
                  projection_hidden_size=projection_hidden_size,
                  prediction_size=prediction_size,
                  prediction_hidden_size=prediction_hidden_size,
                  moving_average_decay=momentum, beta=beta,args=args).to(device)

    opt = torch.optim.Adam(hyp_mview.parameters(), lr=lr, weight_decay=weight_decay)

    results = []

    # Training
    best = 0
    patience_count = 0
    for epoch in range(epochs):
        for _ in range(batch_size):
            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1)
            ba = adj[idx: idx + sample_size, idx: idx + sample_size]
            #bd = diff[idx: idx + sample_size, idx: idx + sample_size]
            #bd = sp.csr_matrix(np.matrix(bd))
            features = features.squeeze(0)
            bf = features[idx: idx + sample_size]
            #^^^^^
            #diff_features = diff_features.squeeze(0)
            #bf2 = diff_features[idx: idx + sample_size]
            diff_embedding = diff_embedding.squeeze(0)
            aug_emb = diff_embedding[idx: idx + sample_size]

            ori_adj=ba
            aug_adj1 = aug.aug_random_edge(ba, drop_percent=drop_edge_rate_1)
            #aug_adj2 = bd

            ori_features=bf
            aug_features1 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_1)
            #aug_features2 = aug.aug_feature_dropout(bf2, drop_percent=drop_feature_rate_2)
            aug_emb = aug.aug_feature_dropout(aug_emb, drop_percent=drop_feature_rate_2)

            ori_adj=process.normalize_adj(ori_adj+sp.eye(ori_adj.shape[0]))
            aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
            #aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

            if sparse:
                ori_adj=process.sparse_mx_to_torch_sparse_tensor(ori_adj).to(device)
                adj_1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1).to(device)
                #adj_2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2).to(device)
            else:
                ori_adj = (ori_adj + sp.eye(ori_adj.shape[0])).todense()
                aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
                #aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()

                ori_adj=torch.FloatTensor(ori_adj[np.newaxis]).to(device)
                adj_1 = torch.FloatTensor(aug_adj1[np.newaxis]).to(device)
                #adj_2 = torch.FloatTensor(aug_adj2[np.newaxis]).to(device)

            ori_features=ori_features.to(device)
            aug_features1 = aug_features1.to(device)
            #aug_features2 = aug_features2.to(device)

            opt.zero_grad()
            loss = hyp_mview(ori_adj, adj_1, aug_emb, ori_features, aug_features1, sparse)

            loss.backward()
            opt.step()
            hyp_mview.update_ma()

        #ema.update()
        if epoch % eval_every_epoch == 0:
            # acc = evaluation(eval_adj, eval_diff, features, model, idx_train, idx_test, sparse)
            acc,prec,rec,f1 = evaluation(eval_adj, nb_classes, features, model, idx_train, idx_test, sparse)
            if acc > best:
                best = acc
                patience_count = 0
            else:
                patience_count += 1
            results.append(acc)
            print('\t epoch {:03d} | loss {:.5f} | clf test acc {:.5f}, prec {:.5f}, rec {:.5f}, f1 {:.5f}'.format(epoch, loss.item(), acc, prec, rec ,f1))
            if patience_count >= patience:
                print('Early Stopping.')
                break
            
    result_over_runs.append(max(results))
    print('\t best acc {:.5f}'.format(max(results)))