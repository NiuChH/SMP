import json
import logging
import pickle
import random
from pprint import pprint

import networkx as nx
import numpy as np
import torch
import torch.utils.data
import torch_geometric as tg
import torch_geometric.transforms as T
#
from networkx.readwrite import json_graph
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.utils import from_networkx

from args import make_args
from utils.arg_helper import mkdir
from utils.pgnn_utils import pre_compute_dist_data, get_link_mask, duplicate_edges, get_random_edge_mask_link_as_neg
from utils.visual_utils import plot_graphs_or_data_list

NO_NODE_FEATURE_DATASETS = {'grid', 'communities', 'usa', 'brazil', 'europe', 'email'}

TASK_TO_DATASET_DICT = {
    'link': [
        'communities',
        'grid',
        'email',

        'CiteSeer',
        'Cora',
        'PubMed',

        'ppi',
        'CS',

        'Physics',

        # 'ogbl-ppa',
    ],
    'link_pair': [
        'email',
        'communities',
        'CS',
        'Physics',
        'CiteSeer',
        'Cora',
        'PubMed',
    ],
    'node_cls': [
        'Cora',
        'CiteSeer',
        'PubMed',
        'communities',
        'CS',
        'Physics',
    ],
    'node_reg': [
        'Cora',
        'CiteSeer',
        'PubMed',
        'communities',
        'CS',
        'Physics',
        'email',
        'ppi'
    ]
}


def get_label_to_nodes(data, shuffle=True):
    n = data.num_nodes
    label_to_nodes = {i: [] for i in range(data.y.min(), data.y.max() + 1)}
    for i in range(n):
        label_to_nodes[data.y[i].item()].append(i)
    if shuffle:
        for k in label_to_nodes.keys():
            label_to_nodes[k] = np.random.permutation(label_to_nodes[k])
    logging.info(f"NodeLabel\tSize")
    for k, v in label_to_nodes.items():
        logging.info(f"{k}\t{len(v)}")
    return label_to_nodes


def get_mask_from_idx(idx, n):
    return np.array(idx, dtype=np.int)


def mask_node_by_label(data, train_size, val_size):
    """
        train_size: the number of nodes for training, among each group of nodes,
            where `group` means `nodes with the same labels`
        val_size: the number of nodes for validation, among each group of nodes

        Example:
            For a graph with 30 nodes, with 9 of them with label `A`,
            10 of them with label `B`, and 11 of them with label `C`,

            when `train_size=3, val_size=2`,

            the number of nodes belonging to `training_set|validation_set|testing_set` will be:
            3|2|4, for nodes with label `A`,
            3|2|5, for nodes with label `B`,
            3|2|6, for nodes with label `C`.

            Therefore, for the whole graph, the `training_set|validation_set|testing_set` split will be `9|6|15`

            When `train_size+val_size >= number_of_nodes_with_label_X`, all the nodes with label `X` will be removed
            from `training_set, validation_set, and testing_set` (but not from the graph).
    """
    label_to_nodes = get_label_to_nodes(data, shuffle=True)
    to_remove = []
    for k, v in label_to_nodes.items():
        if len(v) <= train_size + val_size:
            to_remove.append(k)
    for k in to_remove:
        label_to_nodes.pop(k)
    train_idx = []
    for v in label_to_nodes.values():
        train_idx.extend(v[:train_size])
    val_idx = []
    for v in label_to_nodes.values():
        val_idx.extend(v[train_size:train_size + val_size])
    test_idx = []
    for v in label_to_nodes.values():
        test_idx.extend(v[train_size + val_size:])
    data.train_mask = get_mask_from_idx(train_idx, data.num_nodes)
    data.val_mask = get_mask_from_idx(val_idx, data.num_nodes)
    data.test_mask = get_mask_from_idx(test_idx, data.num_nodes)


def mask_node_randomly(data, train_size, val_size):
    idx = np.random.permutation(data.x.size(0))
    data.train_mask = idx[:train_size]
    data.val_mask = idx[train_size:train_size + val_size]
    data.test_mask = idx[train_size + val_size:]


def add_pairwise_label_to_node_label_data(data, split):
    if hasattr(data, 'mask_link_positive') or hasattr(data, 'mask_link_positive_train'):
        return
    label_to_nodes = get_label_to_nodes(data, shuffle=False)

    edge_labels = []
    for k, v in label_to_nodes.items():
        for i_node1, node1 in enumerate(v):
            for node2 in v[:i_node1]:
                edge_labels.append([node1, node2])
    mask_link_positive = np.array(edge_labels).T

    mask_link_positive = mask_link_positive[:, np.random.permutation(mask_link_positive.shape[1])[:data.num_edges // 2]]
    num_negative_edges = int(data.num_edges / 2 * split)
    val_len = num_negative_edges // 2
    test_len = num_negative_edges - val_len
    train_len = data.num_edges // 2 - val_len - test_len
    mask_link_negative = np.zeros((2, num_negative_edges), dtype=mask_link_positive.dtype)
    for k in label_to_nodes:
        label_to_nodes[k] = set(label_to_nodes[k])
    for i in range(num_negative_edges):
        while True:
            mask_temp = tuple(np.random.choice(data.num_nodes, size=(2,), replace=False))
            if mask_temp[1] not in label_to_nodes[data.y[mask_temp[0]].item()]:
                mask_link_negative[:, i] = mask_temp
                break

    data.mask_link_positive = mask_link_positive

    data.mask_link_positive_train = mask_link_positive[:, :train_len]
    data.mask_link_positive_val = mask_link_positive[:, train_len:train_len + val_len]
    data.mask_link_positive_test = mask_link_positive[:, train_len + val_len:]

    data.mask_link_negative_val = mask_link_negative[:, :val_len]
    data.mask_link_negative_test = mask_link_negative[:, val_len:]
    logging.info(f'mask_link_positive: {mask_link_positive.shape}')


def load_from_edge_list(dataset_name):
    assert dataset_name in ['brazil', 'usa', 'europe']
    data_file_name = 'data/airports/' + dataset_name + '-airports.edgelist'
    label_file_name = 'data/airports/labels-' + dataset_name + '-airports.txt'
    g = nx.read_edgelist(data_file_name, nodetype=int)
    for line in open(label_file_name, 'r').readlines()[1:]:
        node, label = map(int, line.strip('\n').split(' '))
        assert label in (0, 1, 2, 3)
        g.nodes[node]['y'] = label
        # g.nodes[node]['y'] = min(g.degree[node], 3) - 1
    permute_idx = np.random.permutation(g.number_of_nodes())
    train_len = max(1, int(0.7 * g.number_of_nodes()))
    val_len = max(1, int(0.1 * g.number_of_nodes()))
    test_len = g.number_of_nodes() - train_len - val_len
    # add_one_hot_id_node_feature(g)
    add_one_hot_deg_node_feature(g)
    return [g], \
           None, \
           [permute_idx[:train_len]], \
           [permute_idx[train_len:train_len + val_len]], \
           [permute_idx[-test_len:]]


def get_dataset_info(data_list):
    info = {
        'num_graphs': len(data_list),
        'avg_num_nodes': np.mean([data.num_nodes for data in data_list]),
        'avg_num_edges': np.mean([data.num_edges for data in data_list]) / 2,
        'feat_len': data_list[0].x.size(1),
        'num_classes': max([data.y.max().item() for data in data_list]) + 1 if hasattr(data_list[0], 'y') and
                                                                               data_list[0].y is not None else None
    }
    print(info)
    return info


def get_tg_dataset(args, dataset_name,
                   remove_feature=False,  # set to True to run inductive communities/PubMed
                   ):
    if dataset_name.startswith('ogbl-'):
        dataset = PygLinkPropPredDataset(name=dataset_name)
    # "Cora", "CiteSeer" and "PubMed"
    elif dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = tg.datasets.Planetoid(root='datasets/', name=dataset_name,
                                        transform=T.NormalizeFeatures())
    elif dataset_name in ['CoraFull']:
        dataset = tg.datasets.CoraFull(root='datasets/' + dataset_name,
                                       transform=T.NormalizeFeatures())
    elif dataset_name in ['CS', 'Physics']:
        dataset = tg.datasets.Coauthor(root='datasets/' + dataset_name, name=dataset_name,
                                       transform=T.NormalizeFeatures())
    else:
        if dataset_name in ['grid', 'communities']:
            remove_feature = True
        elif dataset_name.endswith('-T'):
            remove_feature = False
            dataset_name = dataset_name.strip('-T')
        dataset = load_tg_dataset(dataset_name)
    dataset_list = list(dataset)
    print(dataset_name)
    data_list = []
    dists_list = []
    dists_removed_list = []
    links_train_list = []
    links_val_list = []
    links_test_list = []
    for i, data in enumerate(dataset_list):
        assert isinstance(data, tg.data.Data)
        data.x = data.x.to(torch.float)
        if 'node' in args.task:
            if 'node_cls' in args.task:
                if args.dataset in ['CS', 'Physics']:
                    mask_node_by_label(data, train_size=20, val_size=30)
                elif args.dataset in ['email']:
                    mask_node_by_label(data, train_size=5, val_size=5)
                elif args.dataset in ['communities']:
                    mask_node_by_label(data, train_size=5, val_size=5)
                elif args.dataset.endswith('_simulation'):
                    mask_node_by_label(data, train_size=int(args.train_size), val_size=int(args.val_size))
            elif 'node_reg' in args.task:  # node regression
                mask_node_randomly(data, train_size=int(data.x.size(0) * 0.6), val_size=int(data.x.size(0) * 0.2))
                g = tg.utils.to_networkx(data, node_attrs=None, to_undirected=True, remove_self_loops=True)
                centrality_method = args.task.split('_')[-1]
                assert centrality_method in {'degree', 'closeness', 'betweenness', 'eigenvector', 'kcore'}
                centrality_path = 'centralities/' + dataset_name + '_' + str(i) + '_' + centrality_method + '.pkl'
                try:
                    centrality = pickle.load(open(centrality_path, 'rb'))
                except FileNotFoundError:
                    if centrality_method == 'degree':
                        centrality = nx.degree_centrality(g)
                    elif centrality_method == 'closeness':
                        centrality = nx.closeness_centrality(g)
                    elif centrality_method == 'betweenness':
                        centrality = nx.betweenness_centrality(g)
                    elif centrality_method == 'eigenvector':
                        centrality = nx.eigenvector_centrality(g)
                    elif centrality_method == 'kcore':
                        centrality = nx.core_number(g)
                    else:
                        centrality = {}
                    mkdir("centralities")
                    if dataset_name not in {'communities', 'grid'}:
                        pickle.dump(centrality, open(centrality_path, 'wb'))
                data.y = torch.tensor([centrality[i_node] for i_node in range(data.x.size(0))], dtype=torch.float)
        if 'link' in args.task:
            if dataset_name.startswith('ogbl-'):
                split_edge = dataset.get_edge_split()
                data.mask_link_positive_train = split_edge["train"]["edge"].T.cpu()
                data.mask_link_positive_val = split_edge["valid"]["edge"].T.cpu()
                data.mask_link_positive_test = split_edge["test"]["edge"].T.cpu()
                if not args.resample_neg:
                    logging.warning("Forcing args.resample_neg == True")
                    args.resample_neg = True
                data.mask_link_negative_train = get_random_edge_mask_link_as_neg(
                    data.edge_index,
                    data.num_nodes,
                    data.mask_link_positive_train.shape[1])
                data.mask_link_negative_val = split_edge["valid"]["edge_neg"].T.cpu()
                data.mask_link_negative_test = split_edge["test"]["edge_neg"].T.cpu()
            elif args.task == 'link_pair' and dataset_name not in ['communities', 'grid', 'ppi', 'email']:
                add_pairwise_label_to_node_label_data(data, split=args.remove_link_ratio)
            elif args.task != 'link_pair' or dataset_name in ['communities', 'grid', 'ppi', 'email']:
                get_link_mask(data, args.remove_link_ratio, resplit=True,
                              infer_link_positive=True if args.task == 'link' else False)
            print(data.mask_link_positive_train.shape)
            print(data.mask_link_positive_val.shape)
            print(data.mask_link_positive_test.shape)
            print(data.mask_link_negative_val.shape)
            print(data.mask_link_negative_test.shape)
            links_train_list.append(data.mask_link_positive_train)
            links_val_list.append(data.mask_link_positive_val)
            links_test_list.append(data.mask_link_positive_test)
        if args.task == 'link':
            if dataset_name.startswith('ogbl-'):
                pass
                # data.edge_index = data.mask_link_positive_train
            else:
                data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()
        if args.approximate != 0:
            if args.task == 'link':
                dists_removed = pre_compute_dist_data(data.mask_link_positive_train, data.num_nodes,
                                                      approximate=args.approximate)
                dists_removed_list.append(dists_removed)
                data.dists = torch.from_numpy(dists_removed).float()
            else:
                dists = pre_compute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=args.approximate)
                dists_list.append(dists)
                data.dists = torch.from_numpy(dists).float()

        if remove_feature:
            data.ori_x = data.x
            data.x = torch.ones((data.x.shape[0], 1))

            # one_hot_deg_adder(data)
        # add_one_hot = False
        # if add_one_hot:
        #     one_hot = torch.eye(data.num_nodes).to(data.x)[:, np.random.permutation(data.num_nodes)]
        #     max_node = max([data.num_nodes for data in dataset_list])
        #     data.x = torch.cat([
        #         data.x, one_hot, torch.zeros([data.num_nodes, max_node - data.num_nodes]).to(data.x)
        #     ], dim=-1)
        # add_one_hot_id_node_feature(data, True, max([data.num_nodes for data in dataset_list]))
        #             print(data.x.numpy())
        data_list.append(data)
    get_dataset_info(data_list)
    return data_list


def nx_to_tg_data(graphs,
                  edge_labels=None,
                  idx_train=None, idx_val=None, idx_test=None):
    data_list = []
    for i in range(len(graphs)):
        graph = graphs[i].copy()
        graph.remove_edges_from(nx.selfloop_edges(graph))

        data = from_networkx(graph)

        # get edge_labels
        if edge_labels and edge_labels[0] is not None:
            edge_label = edge_labels[i]
            mask_link_positive = np.stack(np.nonzero(edge_label))
            data.mask_link_positive = mask_link_positive
        if idx_train is not None:
            g_idx_train, g_idx_val, g_idx_test = idx_train[i], idx_val[i], idx_test[i]
            for phase, idx in {
                'train': g_idx_train,
                'val': g_idx_val,
                'test': g_idx_test
            }.items():
                flags = np.zeros(graph.number_of_nodes(), dtype=np.bool)
                flags[idx] = True
                setattr(data, f'{phase}_mask', torch.from_numpy(flags).bool())
        data_list.append(data)
    for data in data_list:
        # Row-normalizes node features to sum-up to one.
        # print(data.x.numpy())
        data.x = data.x / data.x.sum(1, keepdim=True).clamp(min=1)
    return data_list


def add_one_hot_id_node_feature(graph, permute=True, max_num=None):
    assert isinstance(graph, nx.Graph)
    n = graph.number_of_nodes()
    idx = np.identity(n, dtype=np.float32)
    if max_num is not None and n < max_num:
        idx = np.concatenate([idx, np.zeros([n, max_num - n])], axis=-1)
    if permute:
        idx = idx[np.random.permutation(n), :]
    feat_map = {}
    for i, n in enumerate(graph.nodes()):
        feat_map[n] = idx[i, :]
    nx.set_node_attributes(graph, feat_map, 'x')


def add_constant_node_feature(graph, constant=1.0):
    assert isinstance(graph, nx.Graph)
    nx.set_node_attributes(graph, [constant], 'x')


def add_one_hot_deg_node_feature_to_tg_data(data, max_deg=None):
    g = tg.utils.to_networkx(data, node_attrs=None, to_undirected=True, remove_self_loops=True)
    add_one_hot_deg_node_feature(g, max_deg=max_deg)
    data_ = tg.utils.from_networkx(g)
    data.x = torch.cat([data.x, data_.x], dim=-1)


def add_one_hot_deg_node_feature(graph, max_deg=None):
    assert isinstance(graph, nx.Graph)
    n = graph.number_of_nodes()

    # noinspection PyCallingNonCallable
    node_to_degree = dict(graph.degree())
    if max_deg is None:
        max_deg = max(node_to_degree.values())
    feat_map = {}
    for i, n in enumerate(graph.nodes()):
        feat_map[n] = np.zeros(max_deg + 1, dtype=np.float32)
        feat_map[n][node_to_degree[n]] = 1.0
    nx.set_node_attributes(graph, feat_map, 'x')


# main data load function
def load_graphs(dataset_str):
    edge_labels = None
    if dataset_str == 'grid':
        graphs = []
        for _ in range(1):
            graph = nx.grid_2d_graph(20, 20)
            graph = nx.convert_node_labels_to_integers(graph)
            add_one_hot_id_node_feature(graph, permute=True)
            graphs.append(graph)

    elif dataset_str == 'communities':
        graphs = []
        edge_labels = []
        for i in range(1):
            community_size = 20
            community_num = 20
            p = 0.01

            graph = nx.connected_caveman_graph(community_num, community_size)
            # graph = nx.caveman_graph(community_num, community_size)

            count = 0

            for (u, v) in graph.edges():
                if random.random() < p:  # rewire the edge
                    x = random.choice(list(graph.nodes))
                    if graph.has_edge(u, x):
                        continue
                    graph.remove_edge(u, v)
                    graph.add_edge(u, x)
                    count += 1
            print('rewire:', count)

            n = graph.number_of_nodes()
            label = np.zeros((n, n), dtype=int)
            node_labels_dict = {}
            for u in list(graph.nodes):
                node_labels_dict[u] = int(u // community_size)
                for v in list(graph.nodes):
                    if u // community_size == v // community_size and u > v:
                        label[u, v] = 1
            add_one_hot_id_node_feature(graph, permute=True)
            nx.set_node_attributes(graph, node_labels_dict, 'y')
            graphs.append(graph)
            edge_labels.append(label)

    elif dataset_str == 'email':

        with open('data/email.txt', 'rb') as f:
            graph = nx.read_edgelist(f)

        label_all = np.loadtxt('data/email_labels.txt')
        graph_label_all = label_all.copy()
        graph_label_all[:, 1] = graph_label_all[:, 1] // 6

        #  to split the whole graph to 6 connected_components
        for edge in list(graph.edges()):
            if graph_label_all[int(edge[0])][1] != graph_label_all[int(edge[1])][1]:
                graph.remove_edge(edge[0], edge[1])

        add_constant_node_feature(graph)

        comps = [comp for comp in nx.connected_components(graph) if len(comp) > 10]
        graphs = [graph.subgraph(comp) for comp in comps]

        edge_labels = []

        for g in graphs:
            n = g.number_of_nodes()
            add_constant_node_feature(g)

            label = np.zeros((n, n), dtype=int)
            node_labels_dict = {}
            for i, u in enumerate(g.nodes()):
                node_labels_dict[u] = int(label_all[int(u)][1])
                for j, v in enumerate(g.nodes()):
                    if label_all[int(u)][1] == label_all[int(v)][1] and i > j:
                        label[i, j] = 1
            edge_labels.append(label)
            nx.set_node_attributes(g, node_labels_dict, 'y')

    elif dataset_str == 'ppi':
        dataset_dir = 'data/ppi'
        print("Loading data...")
        G = json_graph.node_link_graph(json.load(open(dataset_dir + "/ppi-G.json")))
        # edge_labels_internal = json.load(open(dataset_dir + "/ppi-class_map.json"))
        # edge_labels_internal = {int(i): l for i, l in edge_labels_internal.items()}

        train_ids = [n for n in G.nodes()]
        # train_labels = np.array([edge_labels_internal[i] for i in train_ids])
        # if train_labels.ndim == 1:
        #     train_labels = np.expand_dims(train_labels, 1)

        print("Using only features..")
        feats = np.load(dataset_dir + "/ppi-feats.npy").astype(dtype=np.float32)
        # Logistic gets thrown off by big counts, so log transform num comments and score
        feats[:, 0] = np.log(feats[:, 0] + 1.0)
        feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
        feat_id_map = json.load(open(dataset_dir + "/ppi-id_map.json"))
        feat_id_map = {int(i): val for i, val in feat_id_map.items()}
        train_feats = feats[[feat_id_map[i] for i in train_ids]]

        node_to_idx = {}
        feat_map = {}
        for i_node, node in enumerate(G.nodes()):
            node_to_idx[node] = i_node
            feat_map[node] = train_feats[i_node] + 0.1
        nx.set_node_attributes(G, feat_map, 'x')

        comps = [comp for comp in nx.connected_components(G) if len(comp) > 10]
        graphs = [G.subgraph(comp) for comp in comps]

    else:
        raise NotImplementedError

    return graphs, edge_labels


def gen_simulation_datasets(name='communities_simulation'):
    assert name.endswith('_simulation')
    if name == 'communities_simulation':
        return gen_communities_simulation()
    raise NotImplementedError()


def gen_communities_simulation():
    # just an example
    graphs = []
    for i in range(1):
        community_size = 20
        community_num = 20
        p = 0.01

        graph = nx.connected_caveman_graph(community_num, community_size)
        count = 0

        for (u, v) in graph.edges():
            if random.random() < p:  # rewire the edge
                x = random.choice(list(graph.nodes))
                if graph.has_edge(u, x):
                    continue
                graph.remove_edge(u, v)
                graph.add_edge(u, x)
                count += 1
        print('rewire:', count)

        n = graph.number_of_nodes()
        label = np.zeros((n, n), dtype=int)
        node_labels_dict = {}
        for u in list(graph.nodes):
            node_labels_dict[u] = int(u // community_size)
            for v in list(graph.nodes):
                if u // community_size == v // community_size and u > v:
                    label[u, v] = 1

        # when args.rm_feature==True, these one-hot features will be removed later in function `get_tg_dataset`
        add_one_hot_id_node_feature(graph, permute=True)

        nx.set_node_attributes(graph, node_labels_dict, 'y')
        graphs.append(graph)
    return graphs


def load_tg_dataset(name='communities'):
    if name in ['brazil', 'usa', 'europe']:
        graphs, edge_labels, idx_train, idx_val, idx_test = load_from_edge_list(name)
        return nx_to_tg_data(graphs, edge_labels, idx_train, idx_val, idx_test)
    elif name.endswith('_simulation'):
        graphs = gen_simulation_datasets(name)
        return nx_to_tg_data(graphs)
    else:
        graphs, edge_labels = load_graphs(name)
        return nx_to_tg_data(graphs, edge_labels)


def plot_graph(data_list, dataset_name, fig_dir='data-sample-figs'):
    mkdir(fig_dir)
    for i, data in enumerate(data_list):
        plot_graphs_or_data_list([data],
                                 title=dataset_name + f'_{i}.pdf', rows=1, cols=1,
                                 save_dir=fig_dir,
                                 node_label=False, fig_size=10,
                                 layout_function=nx.fruchterman_reingold_layout
                                 )


if __name__ == '__main__':
    args = make_args()
    args.approximate = 0
    args.task = 'node_cls'
    args.dataset = 'usa'
    args.rm_feature = False
    if args.task == 'All':
        task_list = [
            'node_cls',
            'link',
            'link_pair'
        ]
    else:
        task_list = [args.task]

    info_list = {}

    for task in task_list:
        if args.dataset != 'All':
            dataset_name_list = [args.dataset]
        else:
            dataset_name_list = []
            for v in TASK_TO_DATASET_DICT.values():
                dataset_name_list.extend(v)
            print(dataset_name_list)
        for dataset_name in dataset_name_list:
            try:
                if dataset_name in info_list:
                    continue
                loaded_data_list = get_tg_dataset(args, dataset_name, remove_feature=args.rm_feature)
                info = get_dataset_info(loaded_data_list)
                if info['avg_num_nodes'] < 500:
                    plot_graph(loaded_data_list, dataset_name)
                info['name'] = dataset_name
                info_list[dataset_name] = info
                pprint(info)
            except NotImplementedError:
                print('NotImplementedError:', dataset_name)

    pickle.dump(info_list, open('dataset_info.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
