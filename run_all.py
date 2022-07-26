from args import make_args
from dataset_utils import TASK_TO_DATASET_DICT, NO_NODE_FEATURE_DATASETS
from main import main
import copy
import traceback


def run_all_datasets(args):
    _ori_args = copy.deepcopy(args)
    if _ori_args.task == 'All':
        task_list = [
            'link',
            'link_pair',
            'node_cls',
        ]
    else:
        task_list = [ori_args.task]
    for task in task_list:
        if _ori_args.dataset == 'All':
            dataset_list = TASK_TO_DATASET_DICT[task]
        elif isinstance(_ori_args.dataset, str):
            dataset_list = [_ori_args.dataset]
        else:
            dataset_list = _ori_args.dataset
        print(f"task: {task}, datasets: {dataset_list}")
        for dataset in dataset_list:
            if dataset in NO_NODE_FEATURE_DATASETS:
                rm_feature_range = [True]
            else:
                rm_feature_range = [False]
            for rm_feature in rm_feature_range:
                if args.model.endswith('SMP') and args.feature_pre and rm_feature:
                    pass
                args.__dict__.update({
                    'dataset': dataset,
                    'task': task,
                    'rm_feature': rm_feature,
                })
                if dataset in ['email', 'ppi']:
                    args.__dict__.update({
                        'test_split': 0.2,
                        'fixed_noise': False,
                    })
                else:
                    args.__dict__.update({
                        'test_split': 0.0,
                        'fixed_noise': True,
                    })
                args.comment = _ori_args.comment + '-'.join(
                    map(str, [task, not rm_feature])
                )
                try:
                    main(copy.deepcopy(args))
                except Exception:
                    print(traceback.format_exc())


def run_all_smp(args):
    for gnn_type in [
        'SGC',
        'GCN'
    ]:
        if gnn_type == 'None':
            feature_pre = False
        else:
            feature_pre = True
        if gnn_type == 'SGC':
            last_layer_choices = [
                'None',
                'Linear',
                'MLP',
            ]
        else:
            last_layer_choices = ['Linear']
        for last_layer in last_layer_choices:
            if last_layer == 'None' and not args.feature_pre:
                epoch_num = 11
                epoch_log = 1
            else:
                epoch_num = 1001
                epoch_log = 5
            args.__dict__.update({
                'epoch_num': epoch_num,
                'epoch_log': epoch_log,
                'last_layer': last_layer,
                'feature_pre': feature_pre,
                'gnn_type': gnn_type
            })
            run_all_datasets(copy.deepcopy(args))


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
        'dataset': 'All',
        'task': 'All',
        'cache': False,
        'remove_link_ratio': 0.2,
        'rm_feature': False,
        'test_split': 0.2,
        'resample_neg': True,
        'edge_batch_size': None,
        'metric': 'auc',
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
