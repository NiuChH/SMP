import logging
import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import to_networkx

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib
import matplotlib.pyplot as plt
import torch_geometric as tg

warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)

options = {
    'node_size': 8,
    'line_color': 'grey',
    'linewidths': 0,
    'width': 0.1,
    'with_labels': False,
}


def plot_graphs_or_data_list(graphs, title='title', rows=1, cols=1,
                             save_dir=None, node_label=True, fig_size=10, layout_function=None, **kwargs):
    if isinstance(graphs[0], tg.data.Data):
        if hasattr(graphs[0], 'y') and graphs[0].y is not None:
            graphs = [to_networkx(g, node_attrs=['y']).to_undirected() for g in graphs]
        else:
            graphs = [to_networkx(g).to_undirected() for g in graphs]
            node_label = False
    batch_size = len(graphs)
    max_num = min(batch_size, rows * cols)
    figure = plt.figure(figsize=(cols * fig_size, rows * fig_size))

    for i in range(max_num):
        idx = i * (batch_size // max_num)
        G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        ax = plt.subplot(rows, cols, i + 1)
        if layout_function is not None:
            pos = layout_function(G)
        else:
            pos = None
        now_options = options.copy()
        now_options.update(kwargs)
        if node_label:
            values = [data['y'] for node, data in G.nodes(data=True)]
            nx.draw_networkx(G,
                             pos=pos,
                             cmap=plt.get_cmap('viridis'),
                             node_color=values,
                             **now_options)
        else:
            nx.draw_networkx(G,
                             pos,
                             **now_options)
        ax.axis('on')

    save_fig(save_dir=save_dir, title=title)


def save_fig(save_dir=None, title='fig', dpi=300, fig_dir='fig'):
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(save_dir, fig_dir)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, title),
                    bbox_inches='tight',
                    dpi=dpi,
                    transparent=True)
        plt.close()
    return
