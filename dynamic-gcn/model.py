import sys
import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import SAGEConv
# from torch_geometric.nn import GINConv


# References
# RvNN - https://github.com/majingCUHK/Rumor_RvNN
# BiGCN - https://github.com/TianBian95/BiGCN/
# Self-Attention - https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py


class TDRumorGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDRumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_index = data.root_index  # skip connection (residual connection)
        root_extend = torch.zeros(len(data.batch), x1.size(1))
        root_extend = root_extend.to(Network.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(
            len(data.batch), x2.size(1)).to(Network.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BURumorGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BURumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_index = data.root_index
        root_extend = torch.zeros(len(data.batch), x1.size(1))
        root_extend = root_extend.to(Network.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(
            len(data.batch), x2.size(1)).to(Network.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BiGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BiGCN, self).__init__()
        self.TDRumorGCN = TDRumorGCN(in_feats, hid_feats, out_feats)
        self.BURumorGCN = BURumorGCN(in_feats, hid_feats, out_feats)

    def forward(self, data):
        TD_x = self.TDRumorGCN(data)
        BU_x = self.BURumorGCN(data)
        x = torch.cat((TD_x, BU_x), 1)
        return x


class Network(nn.Module):
    # def __init__(self, in_feats, hid_feats, out_feats, snapshot_num, device):
    def __init__(self, in_feats, hid_feats, out_feats, settings):
        super(Network, self).__init__()
        # Network.snapshot_num = snapshot_num
        # Network.device = device

        Network.snapshot_num = settings['snapshot_num']
        Network.device = settings['cuda']
        Network.learning_sequence = settings['learning_sequence']

        self.rumor_GCN_0 = BiGCN(in_feats, hid_feats, out_feats)
        self.W_s1 = nn.Linear(out_feats * 2 * 4, 1)  # additive attention
        self.fc = nn.Linear((out_feats + hid_feats) * 2 * 2, 4)
        self.init_weights()

    def init_weights(self):  # Xavier Init
        init.xavier_normal_(self.rumor_GCN_0.TDRumorGCN.conv1.weight)
        init.xavier_normal_(self.rumor_GCN_0.TDRumorGCN.conv2.weight)
        init.xavier_normal_(self.rumor_GCN_0.BURumorGCN.conv1.weight)
        init.xavier_normal_(self.rumor_GCN_0.BURumorGCN.conv2.weight)
        init.xavier_normal_(self.W_s1.weight)
        init.xavier_normal_(self.fc.weight)

    def additive_attention(self, x, x_context):  # additive attention
        attn_w = []
        for current_x in x:
            attn_w.append(self.W_s1(torch.cat((current_x, x_context), 1)))
        attn_weights = torch.cat((attn_w), 1)  # B x 5
        attn_weights = F.softmax(attn_weights, dim=1)
        updated_x = []
        for index, current_x in enumerate(x):
            weighted_x = torch.bmm(
                current_x.unsqueeze(2),
                attn_weights[:, index].unsqueeze(1).unsqueeze(2)
            )
            updated_x.append(weighted_x)
        updated_x = torch.stack(updated_x, 1)
        return updated_x

    def self_attention(self, query, key, value, mask=None):  # Self-Attention
        dk = query.size()[-1]  # 256
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)

    def attention_module(self, x_stack):
        # MEAN
        # LSTM, GRU
        # self-attention

        if Network.learning_sequence == "additive":
            x_mean = x_stack.mean(dim=1)
            x_stack = self.additive_attention(x_stack, x_mean)  # query: context
        elif Network.learning_sequence == "dot_product":
            x_stack = self.self_attention(x_stack, x_stack, x_stack)
        else:
            pass

        return x_stack

    def forward(self, snapshots):
        x = []
        for s in snapshots:
            x.append(self.rumor_GCN_0(s))

        x_stack = torch.stack(x, 1)  # B x S x D - E.g.: (20, 3, 256)
        x_stack = self.attention_module(x_stack)

        x_mean = x_stack.mean(dim=1)  # mean pooling (nodes -> graph)
        x_max = torch.max(x_stack, dim=1)[0]  # max pooling (nodes -> graph)
        x_cat = torch.cat((x_mean, x_max), 1).squeeze(2)
        x = self.fc(x_cat)
        x = F.log_softmax(x, dim=1)
        return x
