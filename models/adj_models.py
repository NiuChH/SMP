import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros

from torch_sparse import SparseTensor


# GNNs with sparse-tensor implementation. Using sparse-tensor for large graphs like ogbl-ppa.
# including GCN and SMP-variants

def normalize_adj(adj):
    # normalize adj in GCN style
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    assert deg.min() > 0.5
    deg_inv_sqrt = deg.pow(-0.5)
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj


def get_adj_from_data(data, normalize=True):
    if hasattr(data, 'adj'):
        return data.adj
    edge_index = data.edge_index
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                       sparse_sizes=(data.x.size(0), data.x.size(0)))
    if normalize:
        adj = normalize_adj(adj)
    data.adj = adj
    return data.adj


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.ones([in_channels, out_channels], dtype=torch.float32), requires_grad=True)
        self.bias = Parameter(torch.ones([out_channels], dtype=torch.float32), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj):
        if self.in_channels > self.out_channels:
            return adj @ (x @ self.weight) + self.bias
        else:
            return adj @ x @ self.weight + self.bias


class AdjGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num,
                 dropout, **kwargs):
        super(AdjGCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(layer_num - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = 0.5 if dropout else 0

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x = data.x
        adj = get_adj_from_data(data, normalize=True)
        return self.x_forward(x, adj)

    def x_forward(self, x, adj):
        for conv in self.convs[:-1]:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x


class AdjMP(nn.Module):
    def __init__(self, layer_num, **kwargs):
        super().__init__()
        self.layer_num = layer_num

    def forward(self, data):
        x = data.x
        adj = get_adj_from_data(data, normalize=True)
        return self.x_forward(x, adj)

    def x_forward(self, x, adj):
        for _ in range(self.layer_num):
            x = adj @ x
        return x


class AdjSGC(nn.Module):
    def __init__(self, input_dim, output_dim, layer_num, **kwargs):
        super().__init__()
        self.linear_first = input_dim > output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.mp = AdjMP(layer_num)

    def forward(self, data):
        x = data.x
        adj = get_adj_from_data(data, normalize=True)
        return self.x_forward(x, adj)

    def x_forward(self, x, adj):
        if self.linear_first:
            x = self.linear(x)
        x = self.mp.x_forward(x, adj)
        if not self.linear_first:
            x = self.linear(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 layer_num=2, dropout=True, **kwargs):
        super().__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
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
        return x


def gen_stochastic_feature(size, dim, normalize=True, method='norm'):
    assert method in ('norm', 'uniform')
    if method == 'norm':
        ret = torch.randn([size, dim])
    elif method == 'uniform':
        ret = torch.rand([size, dim])
    if normalize:
        ret = ret * (dim ** (-0.5))
    return ret


class AdjSMP(nn.Module):
    """
    | Model_name          | gnn_type | last_layer |
    | ------------------- | -------- | ---------- |
    | SMP-Identity        | "SGC"    | "None"     |
    | SMP-Linear          | "SGC"    | "Linear"   |
    | SMP-MLP             | "SGC"    | "MLP"      |
    | SMP-Linear-GCN_feat | "GCN"    | "Linear"   |
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 layer_num, fixed_noise, stochastic_dim,
                 feature_pre, last_layer, dropout, gnn_type, stochastic_feature_dist='norm', **kwargs):
        super().__init__()
        self.mp = AdjMP(layer_num)
        if feature_pre:
            if gnn_type == 'SGC':
                self.gnn = AdjSGC(input_dim=input_dim, layer_num=layer_num, output_dim=hidden_dim)
            elif gnn_type == 'GCN':
                self.gnn = AdjGCN(input_dim=input_dim, layer_num=layer_num, output_dim=hidden_dim,
                                  hidden_dim=hidden_dim, dropout=dropout)
            elif gnn_type == 'None':
                self.gnn = self.mp
            else:
                raise NotImplementedError
            last_input_dim = hidden_dim + stochastic_dim
        else:
            self.gnn = self.mp
            last_input_dim = input_dim + stochastic_dim

        if last_layer == 'Linear':
            self.last_layer = nn.Linear(last_input_dim, output_dim)
        elif last_layer == 'MLP':
            self.last_layer = SimpleMLP(input_dim=last_input_dim, hidden_dim=hidden_dim,
                                        output_dim=output_dim, layer_num=2, dropout=dropout)
        elif last_layer == 'None':
            self.last_layer = None
        else:
            raise NotImplementedError

        self.fixed_noise = fixed_noise
        self.stochastic_dim = stochastic_dim
        self.stochastic_feature_dist = stochastic_feature_dist

    def forward(self, data):
        x = data.x
        adj = get_adj_from_data(data, normalize=True)
        if self.fixed_noise:
            if not hasattr(data, 'stochastic_feature'):
                data.stochastic_feature = gen_stochastic_feature(
                    size=x.size(0), dim=self.stochastic_dim, normalize=True, method=self.stochastic_feature_dist).to(x)
            noise_feature = data.stochastic_feature
        else:
            noise_feature = gen_stochastic_feature(
                size=x.size(0), dim=self.stochastic_dim, normalize=True, method=self.stochastic_feature_dist).to(x)

        noise_feature = self.mp.x_forward(noise_feature, adj)
        x = self.gnn.x_forward(x, adj)

        x = F.normalize(x, p=2, dim=-1)

        x = torch.cat((x, noise_feature), dim=-1)
        if self.last_layer is not None:
            x = self.last_layer(x)

        return x


class AdjSMPGCN(nn.Module):
    """
    SMP-Linear-GCN_concat (not reported in the paper)
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 layer_num, fixed_noise, stochastic_dim,
                 feature_pre, last_layer, dropout, gnn_type, stochastic_feature_dist='norm', **kwargs):
        super().__init__()
        self.gnn = AdjGCN(input_dim=input_dim + stochastic_dim, layer_num=layer_num, output_dim=output_dim,
                          hidden_dim=hidden_dim, dropout=dropout)
        self.fixed_noise = fixed_noise
        self.stochastic_dim = stochastic_dim
        self.stochastic_feature_dist = stochastic_feature_dist

    def forward(self, data):
        x = data.x
        adj = get_adj_from_data(data, normalize=True)
        if self.fixed_noise:
            if not hasattr(data, 'stochastic_feature'):
                data.stochastic_feature = gen_stochastic_feature(
                    size=x.size(0), dim=self.stochastic_dim, normalize=True, method=self.stochastic_feature_dist).to(x)
            noise_feature = data.stochastic_feature
        else:
            noise_feature = gen_stochastic_feature(
                size=x.size(0), dim=self.stochastic_dim, normalize=True, method=self.stochastic_feature_dist).to(x)
        x = torch.cat((x, noise_feature), dim=-1)
        x = self.gnn.x_forward(x, adj)
        return x


class AdjSMPGCNGCN(nn.Module):
    """
    SMP-Linear-GCN_both
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 layer_num, fixed_noise, stochastic_dim,
                 feature_pre, last_layer, dropout, gnn_type, stochastic_feature_dist='norm', **kwargs):
        super().__init__()
        self.gnn_e = AdjGCN(input_dim=stochastic_dim, layer_num=layer_num, output_dim=hidden_dim,
                            hidden_dim=hidden_dim, dropout=dropout)
        self.gnn_f = AdjGCN(input_dim=input_dim, layer_num=layer_num, output_dim=hidden_dim,
                            hidden_dim=hidden_dim, dropout=dropout)
        self.fixed_noise = fixed_noise
        self.stochastic_dim = stochastic_dim
        self.last_layer = nn.Linear(hidden_dim * 2, output_dim)
        self.stochastic_feature_dist = stochastic_feature_dist

    def forward(self, data):
        x = data.x
        adj = get_adj_from_data(data, normalize=True)
        if self.fixed_noise:
            if not hasattr(data, 'stochastic_feature'):
                data.stochastic_feature = gen_stochastic_feature(
                    size=x.size(0), dim=self.stochastic_dim, normalize=True, method=self.stochastic_feature_dist).to(x)
            noise_feature = data.stochastic_feature
        else:
            noise_feature = gen_stochastic_feature(
                size=x.size(0), dim=self.stochastic_dim, normalize=True, method=self.stochastic_feature_dist).to(x)
        noise_feature = self.gnn_e.x_forward(noise_feature, adj)
        x = self.gnn_f.x_forward(x, adj)
        x = torch.cat((x, noise_feature), dim=-1)
        x = self.last_layer(x)
        return x
