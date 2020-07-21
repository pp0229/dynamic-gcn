import sys
import os
import copy
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
    def __init__(self, in_feats, hid_feats, out_feats, snapshot_num, device):
        super(Network, self).__init__()
        Network.snapshot_num = snapshot_num
        Network.device = device
        self.rumor_GCN_0 = BiGCN(in_feats, hid_feats, out_feats)
        self.W_s1 = nn.Linear(out_feats * 2 * 4, 1)
        self.fc = nn.Linear((out_feats + hid_feats) * 2 * 2, 4)
        self.init_weights()

    def init_weights(self):  # Xavier Init
        init.xavier_normal_(self.rumor_GCN_0.TDRumorGCN.conv1.weight)
        init.xavier_normal_(self.rumor_GCN_0.TDRumorGCN.conv2.weight)
        init.xavier_normal_(self.rumor_GCN_0.BURumorGCN.conv1.weight)
        init.xavier_normal_(self.rumor_GCN_0.BURumorGCN.conv2.weight)
        init.xavier_normal_(self.W_s1.weight)
        init.xavier_normal_(self.fc.weight)

    def attention_module(self, x, x_context):  # (Additive Attention)
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
        return updated_x

    def forward(self, snapshots):
        x = []
        for s in snapshots:
            x.append(self.rumor_GCN_0(s))

        x_stack = torch.stack(x, 1)
        x_mean = x_stack.mean(dim=1)
        x = self.attention_module(x, x_mean)  # query

        x_stack = torch.stack(x, 1)
        x_mean = x_stack.mean(dim=1)  # mean pooling (nodes -> graph)
        x_max = torch.max(x_stack, dim=1)[0]  # max pooling (nodes -> graph)
        x_cat = torch.cat((x_mean, x_max), 1).squeeze(2)
        x = self.fc(x_cat)
        x = F.log_softmax(x, dim=1)
        return x
