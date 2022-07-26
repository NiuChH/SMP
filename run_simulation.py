import copy

from args import make_args
from run_all import run_all_smp, run_all_datasets

if __name__ == "__main__":
    ori_args = make_args()
    ori_args.__dict__.update({
        # run
        'gpu': True,
        'cuda': '0',
        'epoch_log': 5,
        'epoch_num': 1001,
        'repeat_num': 5,

        # output
        'result_dir': 'results',

        'plot_table': False,
        'comment': '-',
        'save_code': True,

        # model
        'use_predictor': False,

        'model': 'All',
        'dropout': True,
        'feature_dim': 32,
        'feature_pre': True,  #
        'hidden_dim': 32,
        'layer_num': 2,
        'output_dim': 32,  # only for pairwise tasks

        # SMP
        'last_layer': 'Linear',
        'fixed_noise': False,
        'trainable_sigma': False,  #
        'stochastic_dim': 32,
        'gnn_type': 'GCN',
        'stochastic_feature_dist': 'norm',
        'heads': 4,

        # PGNN
        'anchor_num': 64,
        'approximate': 0,
        'permute': True,

        # train
        'batch_size': 8,
        'lr': 0.01,

        # dataset
        'dataset': 'communities_simulation',

        'task': 'node_cls',
        'cache': False,
        'remove_link_ratio': 0.0,
        'rm_feature': True,  # `True` means constant features
        'test_split': 0.2,
        'resample_neg': True,
        'edge_batch_size': None,
        'metric': 'auc',

        # see function `mask_node_by_label` in `dataset_utils.py` for more details
        'train_size': 5,
        'val_size': 5
    })

    args = copy.deepcopy(ori_args)
    if ori_args.model == 'All':
        model_list = [
            # all available models
            'AdjSMP',
            'AdjGCN',
            # 'AdjSMPGCN', # not reported
            'AdjSMPGCNGCN',
            'AdjSGC',
            'GAT',
            'PGNN',
        ]
    else:
        model_list = [ori_args.model]
    for model in model_list:
        ori_args.model = model
        if model == 'PGNN':
            ori_args.approximate = 2
        else:
            ori_args.approximate = 0
        if model.endswith('SMP'):
            run_all_smp(copy.deepcopy(ori_args))
        else:
            run_all_datasets(copy.deepcopy(ori_args))
