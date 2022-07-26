import copy

from args import make_args
from main import main

if __name__ == "__main__":
    ori_args = make_args()
    ori_args.__dict__.update({
        # run
        'gpu': True,
        'cuda': '0',
        'epoch_log': 1,
        'epoch_num': 61,
        'repeat_num': 3,

        # output
        'result_dir': 'ppa_results',
        'plot_table': False,
        'comment': '',
        'save_code': True,

        # model
        'use_predictor': True,

        'model': 'AdjGCN',
        'dropout': True,
        'feature_dim': 256,
        'feature_pre': True,  #
        'hidden_dim': 256,
        'layer_num': 3,
        'output_dim': 256,  # only for pairwise tasks

        # SMP
        'last_layer': 'Linear',
        'fixed_noise': True,
        'trainable_sigma': False,  #
        'stochastic_dim': 64,
        'gnn_type': 'SGC',

        # PGNN
        'anchor_num': 64,
        'approximate': 0,
        'permute': True,

        # train
        'batch_size': 8,
        'lr': 0.01,

        # dataset
        'dataset': 'ogbl-ppa',
        'task': 'link',
        'cache': False,
        'remove_link_ratio': 0.2,
        'rm_feature': False,
        'test_split': 0.0,
        'resample_neg': True,
        # 'edge_batch_size': None,
        'edge_batch_size': 64 * 1024,
        'metric': 'hit',
    })

    args = copy.deepcopy(ori_args)
    main(args)
