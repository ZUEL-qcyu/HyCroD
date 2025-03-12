import torch
import torch.nn as nn

##############################################
# 这个模块是图神经网络层，我们这个模型是训练GCN层，
# 然后把数据通过GCN层得到嵌入表示，再用线性模型进行分类
# 对应论文里的f()#
##############################################

class GCNLayer(nn.Module):
    
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        '''因为GCNLayer继承了nn.Module，要想在forward中调用该类的所有方法，
        必须要找到这些方法，因此要super(GCNLayer)来找到上一级'''
        super(GCNLayer, self).__init__()
        
        self.fc = nn.Linear(in_ft, out_ft, bias=False) ## 输入和输出，默认不使用偏置
        self.act = nn.PReLU() if act == 'prelu' else nn.ReLU() ##激活函数
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0) ##用0来初始化参数
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m) 

    def weights_init(self, m):
        if isinstance(m, nn.Linear): ##判断m是否为nn.Linear类型
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        '''seq:属性
            adj:邻接矩阵
        '''
        seq_fts = self.fc(seq) ##先把seq转换为另一种维度的向量
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
            '''torch.unsqueeze将矩阵相乘的结果升维'''
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)