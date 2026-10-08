import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, lambda1=1, lambda2=0.5, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.lambda1 = lambda1  # 权重系数1
        self.lambda2 = lambda2  # 权重系数2
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))  
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj1, adj2):
        Wh = torch.mm(h, self.W)                                      # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)       # 每一个节点和其它所有节点拼在一起
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # 计算每一个节点与其它节点的注意力值
        zero_vec = 0*torch.ones_like(e)
        inf_vec = -9e15*torch.ones_like(e)

        attention1 = torch.where(adj1 > 0, e, zero_vec)
        attention2 = torch.where(adj2 > 0, e, zero_vec)
        attention = self.lambda1 * attention1 + self.lambda2 * attention2

        attention = torch.where(attention != 0, attention, inf_vec)
        attention = F.softmax(attention, dim=1)                       # 计算每一个节点与其它节点的注意力分数
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)                         # 聚合邻居节点特征更新自己

        if self.concat:
            return F.elu(h_prime)                                     # 激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):               # 两个节点拼在一起的全排列。
        N = Wh.size()[0] 
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] # 多头注意力
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)                  # 最后一层
        self.out_proj = nn.Linear(nhid * nheads, nhid)

    def forward(self, x, adj1, adj2):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj1, adj2) for att in self.attentions], dim=1)                    # 把每一个头计算出来的结果拼接一起
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj1, adj2))
        # x = F.elu(self.out_proj(x))

        return F.log_softmax(x, dim=1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        # 使用cross_entropy计算原始损失，注意targets应为类别索引
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
