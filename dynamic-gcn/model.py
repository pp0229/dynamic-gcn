from dataset import GraphSnapshotDataset

import sys
import os
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

import copy


# from tools.earlystopping import EarlyStopping

"""
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
"""


# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)



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

        # root feature enhancement, ~ skip connection (residual connection)
        root_index = data.root_index
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(Network.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(Network.device)
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
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(Network.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(Network.device)
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
        BU_x = self.TDRumorGCN(data)
        x = torch.cat((TD_x, BU_x), 1)
        return x


class Network(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, snapshot_num, device):
        super(Network, self).__init__()
        Network.snapshot_num = snapshot_num
        Network.device = device

        # for gcn_index in range(Network.snapshot_num):
        #     gcn = eval("self.rumor_GCN_{0}".format(gcn_index))
        #     gcn = BiGCN(in_feats, hid_feats, out_feats)

        # shared vs unshared weights
        self.rumor_GCN_0 = BiGCN(in_feats, hid_feats, out_feats)

        # self.rumor_GCN_0 = BiGCN(in_feats, hid_feats, out_feats)
        # self.rumor_GCN_1 = BiGCN(in_feats, hid_feats, out_feats)
        # self.rumor_GCN_2 = BiGCN(in_feats, hid_feats, out_feats)
        # self.rumor_GCN_3 = BiGCN(in_feats, hid_feats, out_feats)
        # self.rumor_GCN_4 = BiGCN(in_feats, hid_feats, out_feats)

        # Attention Module (Additive Attention)
        # option 1
        self.W_s1 = nn.Linear(out_feats * 2 * 4, 1)
        self.fc = nn.Linear((out_feats + hid_feats) * 2, 4)

        self.init_weights()  # Xavier Init

    def init_weights(self):
        # for gcn_index in range(Network.snapshot_num):
        #     gcn = eval("self.rumor_GCN_{0}".format(gcn_index))
        #     init.xavier_normal_(gcn.TDrumorGCN.conv1.weight)
        #     init.xavier_normal_(gcn.TDrumorGCN.conv2.weight)
        #     init.xavier_normal_(gcn.BUrumorGCN.conv1.weight)
        #     init.xavier_normal_(gcn.BUrumorGCN.conv2.weight)

        init.xavier_normal_(self.rumor_GCN_0.TDRumorGCN.conv1.weight)
        init.xavier_normal_(self.rumor_GCN_0.TDRumorGCN.conv2.weight)
        init.xavier_normal_(self.rumor_GCN_0.BURumorGCN.conv1.weight)
        init.xavier_normal_(self.rumor_GCN_0.BURumorGCN.conv2.weight)
        init.xavier_normal_(self.W_s1.weight)
        init.xavier_normal_(self.fc.weight)

    # additive attention with mean key + 1 layer

    # def attention_module(self, x0, x1, x2, x3, x4, x_context):  # (Additive Attention)
    def attention_module(self, x, x_context):  # (Additive Attention)

        attn_w = []
        for current_x in x:
            attn_w.append(self.W_s1(torch.cat((current_x, x_context), 1)))

        # attn_w_0 = self.W_s1(torch.cat((x0, x_context), 1))
        # attn_w_1 = self.W_s1(torch.cat((x1, x_context), 1))
        # attn_w_2 = self.W_s1(torch.cat((x2, x_context), 1))
        # attn_w_3 = self.W_s1(torch.cat((x3, x_context), 1))
        # attn_w_4 = self.W_s1(torch.cat((x4, x_context), 1))
        attn_weights = torch.cat((attn_w), 1)  # B x 5
        attn_weights = F.softmax(attn_weights, dim=1)  # TODO: confirmed

        updated_x = []
        for index, current_x in enumerate(x):
            weighted_x = torch.bmm(current_x.unsqueeze(2), attn_weights[:, index].unsqueeze(1).unsqueeze(2))
            updated_x.append(weighted_x)

        # x0 = torch.bmm(x0.unsqueeze(2), attn_weights[:, 0].unsqueeze(1).unsqueeze(2))
        # x1 = torch.bmm(x1.unsqueeze(2), attn_weights[:, 1].unsqueeze(1).unsqueeze(2))
        # x2 = torch.bmm(x2.unsqueeze(2), attn_weights[:, 2].unsqueeze(1).unsqueeze(2))
        # x3 = torch.bmm(x3.unsqueeze(2), attn_weights[:, 3].unsqueeze(1).unsqueeze(2))
        # x4 = torch.bmm(x4.unsqueeze(2), attn_weights[:, 4].unsqueeze(1).unsqueeze(2))
        # return x0, x1, x2, x3, x4
        return updated_x

    def forward(self, snapshots):
        # unshared weights
        # x0 = self.rumor_GCN_0(s0)
        # x1 = self.rumor_GCN_1(s1)
        # x2 = self.rumor_GCN_2(s2)
        # x3 = self.rumor_GCN_3(s3)
        # x4 = self.rumor_GCN_4(s4)

        # shared weights
        x = []
        for s in snapshots:
            x.append(self.rumor_GCN_0(s))
        # x0 = self.rumor_GCN_0(s[0])
        # x1 = self.rumor_GCN_0(s[1])
        # x2 = self.rumor_GCN_0(s[2])
        # x3 = self.rumor_GCN_0(s[3])
        # x4 = self.rumor_GCN_0(s[4])
        # x_stack = torch.stack([x0, x1, x2, x3, x4], 1)  # 0402
        x_stack = torch.stack(x, 1)  # 0402
        x_mean = x_stack.mean(dim=1)  # option 1
        x_sum = x_stack.sum(dim=1)  # option 2
        # x_max = th.max(x_stack, dim=1)[0]  # option 3

        # x0, x1, x2, x3, x4 = self.attention_module(x0, x1, x2, x3, x4, x_mean)
        # x0, x1, x2, x3, x4 = self.attention_module(x, x_sum)
        x = self.attention_module(x, x_mean)

        # x_sum = th.stack([x0, x1, x2, x3, x4], 1)  # 0402
        # x = x_sum.sum(dim=1).squeeze(2)
        # x_stack = torch.stack([x0, x1, x2, x3, x4], 1)  # 0412
        x_stack = torch.stack(x, 1)  # 0412
        x_mean = x_stack.mean(dim=1)  # option 1
        x_max = x_stack.sum(dim=1)[0]


        # x = x.squeeze(2)

        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        # x = F.log_softmax(x, dim=1)  # 0512
        return x
