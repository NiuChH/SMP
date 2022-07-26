import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch.nn import init

# GNNs with torch_geometric implementations, from P-GNN source code (https://github.com/JiaxuanYou/P-GNN)
# including GraphSAGE(not used), GAT, and PGNN

####################### Basic Ops #############################

# # PGNN layer, only pick closest node for message passing
class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)
        else:
            self.dist_compute = None

        self.linear_hidden = nn.Linear(input_dim * 2, output_dim)
        self.linear_out_position = nn.Linear(output_dim, 1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1).to(feature.device)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages)  # n*m*d
        dim = len(messages.size())
        if dim == 3:
            messages = messages.squeeze(-1)
        elif dim == 2:
            messages = messages.unsqueeze(-1)

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        if len(out_position.size()) == 1:
            out_position = out_position.unsqueeze(-1)
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure


### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


####################### NNs #############################

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_first = nn.Linear(feature_dim, hidden_dim)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.linear_first(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        # x = F.normalize(x, p=2, dim=-1)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        # x = F.normalize(x, p=2, dim=-1)
        return x


class GAT(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, heads=1, **kwargs):
        super(GAT, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        gat_drop = 0.6
        if heads >= 2:
            hidden_dim //= heads
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GATConv(feature_dim, hidden_dim, heads=heads, dropout=gat_drop)
        else:
            self.conv_first = tg.nn.GATConv(input_dim, hidden_dim, heads=heads, dropout=gat_drop)
        self.conv_hidden = nn.ModuleList([
            tg.nn.GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=gat_drop) for _ in range(layer_num - 2)])
        self.conv_out = tg.nn.GATConv(hidden_dim * heads, output_dim, heads=1, concat=True, dropout=gat_drop)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        return self.x_forward(x, edge_index)

    def x_forward(self, x, edge_index):
        if self.feature_pre:
            x = self.linear_pre(x)
        if self.dropout:
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=0.6, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv_out(x, edge_index)
        # x = F.normalize(x, p=2, dim=-1)
        return x


class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, task='link', **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num > 1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim)
        self.task = task

    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax)
        if self.layer_num == 1:
            # return x
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)

        if 'link' in self.task:
            return x_position
        else:
            # for node cls
            return x
