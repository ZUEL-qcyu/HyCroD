import torch
import torch.nn as nn
import copy
import numpy as np
import torch.nn.functional as F
# from models.direction_diffusion import diffusion
# from hyperbolic_learning.hyperkmeans import hkmeanscom as Community_cluster
# import os
import manifolds

class MLP(nn.Module):
    '''对应论文中的p和predictor'''

    def __init__(self, inp_size, outp_size, hidden_size):
        '''
        Args:

        inp_size: 输入的维度
        outp_size: 输出的维度
        hidden_size: 隐藏层的维度
        '''
        super().__init__()
        # MLP的架构是一个输入层，接一个隐藏层，再接PReLU进行非线性化处理，最后
        # 全连接层进行输出
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):
        '''数据通过MLP时没有进行其他的处理，因此仅仅通过net()'''
        return self.net(x)


class GraphEncoder(nn.Module):
    '''encoder包含gnn(论文中的f)和MLP(论文中的p)'''

    def __init__(self,
                 gnn,
                 projection_hidden_size,
                 projection_size):
        '''
        Args:

        gnn: 事先定义好的gnn层
        projection_hidden_size: 通过MLP时，MLP的隐藏层维度
        projection_size: 输出维度
        '''
        super().__init__()

        self.gnn = gnn
        ## projector对应论文中的p，输入为512维，因为本文定义的嵌入表示的维度为512
        self.projector = MLP(512, projection_size, projection_hidden_size)

    def forward(self, adj, in_feats, sparse):
        representations = self.gnn(in_feats, adj, sparse)  # 初始的嵌入表示
        representations = representations.view(-1, representations.size(-1))
        projections = self.projector(representations)  # (batch, proj_dim)
        return projections


class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    '''参数更新方式，MOCO-like'''
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def sim(h1, h2):
    '''计算相似度'''
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())

####################################################################################


def agg_contrastive_loss(h, z):
    def f(x): return torch.exp(x)
    cross_sim = f(sim(h, z))
    return -torch.log(cross_sim.diag()/cross_sim.sum(dim=-1))


def interact_contrastive_loss(h1, h2):
    def f(x): return torch.exp(x)
    intra_sim = f(sim(h1, h1))
    inter_sim = f(sim(h1, h2))
    return -torch.log(inter_sim.diag() /
                      (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))

def contrastive_loss_wo_cross_network(h1, h2, z):
    f = lambda x: torch.exp(x)
    intra_sim = f(sim(h1, h1))
    inter_sim = f(sim(h1, h2))
    return -torch.log(inter_sim.diag() /
                     (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))


def contrastive_loss_wo_cross_view(h1, h2, z):
    f = lambda x: torch.exp(x)
    cross_sim = f(sim(h1, z))
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))

##################################################################################

class HypMVIEW(nn.Module):

    def __init__(self,
                 gnn,
                 feat_size,
                 projection_size,
                 projection_hidden_size,
                 prediction_size,
                 prediction_hidden_size,
                 moving_average_decay,
                 beta,
                 args):
        '''
        moving_average_decay: 权重更新的值
        beta: loss函数的组合
        alpha: 信息传递

        '''
        super().__init__()


        self.online_encoder = GraphEncoder(
            gnn, projection_hidden_size, projection_size)
        self.target_encoder1 = copy.deepcopy(self.online_encoder)
        self.target_encoder2 = copy.deepcopy(self.online_encoder)

        self.manifold = getattr(manifolds, args.manifold)()
        self.c = args.c

        set_requires_grad(self.target_encoder1, False)
        set_requires_grad(self.target_encoder2, False)

        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(
            projection_size, prediction_size, prediction_hidden_size)

        self.beta = beta


    def reset_moving_average(self):
        del self.target_encoder1
        del self.target_encoder2
        self.target_encoder1 = None
        self.target_encoder2 = None

    def update_ma(self):
        assert self.target_encoder1 or self.target_encoder2 is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater,
                              self.target_encoder1, self.online_encoder)
        update_moving_average(self.target_ema_updater,
                              self.target_encoder2, self.online_encoder)

    def set_requires_grad(module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad


    def forward(self, adj, aug_adj_1, aug_emb, feat, aug_feat_1, sparse):

        online_proj_one = self.online_encoder(aug_adj_1, aug_feat_1, sparse)

        eud_emb = self.manifold.proj_tan0(self.manifold.logmap0(aug_emb, c=self.c), c=self.c)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(eud_emb)

        with torch.no_grad():
            target_proj_one = self.target_encoder1(aug_adj_1, aug_feat_1, sparse)

            target_proj_two = eud_emb

        l1 = self.beta * contrastive_loss_wo_cross_network(online_pred_one, online_pred_two, target_proj_two.detach()) + \
             (1.0 - self.beta) * contrastive_loss_wo_cross_view(online_pred_one, online_pred_two, target_proj_two.detach())

        l2 = self.beta * contrastive_loss_wo_cross_network(online_pred_two, online_pred_one, target_proj_one.detach()) + \
             (1.0 - self.beta) * contrastive_loss_wo_cross_view(online_pred_two, online_pred_one, target_proj_one.detach())

        loss = 0.5 * (l1 + l2)
        return loss.mean()