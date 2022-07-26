from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()

    parser.add_argument('--config_file', dest='config_file',
                        type=str, default='config/train.yaml', help="Path of config file")
    parser.add_argument('--log_level', dest='log_level', type=str,
                        default='INFO', help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument('--result_dir', dest='result_dir',
                        type=str,
                        default='results', help="where to save the results")

    # general
    parser.add_argument('--comment', dest='comment', default='0', type=str,
                        help='comment')
    parser.add_argument('--task', dest='task', default='link', type=str,
                        help='link; link_pair')
    parser.add_argument('--model', dest='model', default='GCN', type=str,
                        help='model class name. E.g., GCN, PGNN, SGAT,...')
    parser.add_argument('--dataset', dest='dataset', default='All', type=str,
                        help='All; Cora; grid; communities; ppi')
    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='whether use gpu')
    parser.add_argument('--cache_no', dest='cache', action='store_false',
                        help='whether use cache')
    parser.add_argument('--cpu', dest='gpu', action='store_false',
                        help='whether use cpu')
    parser.add_argument('--cuda', dest='cuda', default='', type=str)

    # dataset
    parser.add_argument('--remove_link_ratio', dest='remove_link_ratio', default=0.2, type=float)
    parser.add_argument('--rm_feature', dest='rm_feature', action='store_true',
                        help='whether rm_feature')
    parser.add_argument('--rm_feature_no', dest='rm_feature', action='store_false',
                        help='whether rm_feature')
    parser.add_argument('--permute', dest='permute', action='store_true',
                        help='whether permute subsets')
    parser.add_argument('--permute_no', dest='permute', action='store_false',
                        help='whether permute subsets')
    parser.add_argument('--feature_pre', dest='feature_pre', action='store_true',
                        help='whether pre transform feature')
    parser.add_argument('--feature_pre_no', dest='feature_pre', action='store_false',
                        help='whether pre transform feature')
    parser.add_argument('--dropout', dest='dropout', action='store_true',
                        help='whether dropout, default 0.5')
    parser.add_argument('--dropout_no', dest='dropout', action='store_false',
                        help='whether dropout, default 0.5')
    parser.add_argument('--approximate', dest='approximate', default=0, type=int,
                        help='k-hop shortest path distance. '
                             '-1 means exact shortest path, '
                             '0 means not computing distance')  # -1, 2

    parser.add_argument('--batch_size', dest='batch_size', default=8, type=int)  # implemented via accumulating gradient
    parser.add_argument('--layer_num', dest='layer_num', default=2, type=int)
    parser.add_argument('--feature_dim', dest='feature_dim', default=32, type=int)
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=32, type=int)
    parser.add_argument('--output_dim', dest='output_dim', default=32, type=int)
    parser.add_argument('--anchor_num', dest='anchor_num', default=64, type=int)

    parser.add_argument('--lr', dest='lr', default=1e-2, type=float)
    parser.add_argument('--epoch_num', dest='epoch_num', default=2001, type=int)
    parser.add_argument('--repeat_num', dest='repeat_num', default=2, type=int)  # 10
    parser.add_argument('--epoch_log', dest='epoch_log', default=10, type=int)

    parser.add_argument('--stochastic_dim', dest='stochastic_dim', default=32, type=int)
    parser.add_argument('--fixed_noise', dest='fixed_noise',
                        action='store_true')
    parser.add_argument('--fixed_noise_no', dest='fixed_noise', action='store_false',
                        help='whether use cache')
    parser.add_argument('--last_layer', dest='last_layer', default='None', type=str)
    parser.add_argument('--test_split', dest='test_split', default=0.0, type=float,
                        help='split proportion of the test set. '
                             '0 means transductive setting'
                             'the validation set has the same length of the test set.',
                        )
    parser.add_argument('--resample_neg', dest='resample_neg', action='store_true')
    parser.add_argument('--resample_neg_no', dest='resample_neg', action='store_false')
    parser.add_argument('--stochastic_feature_dist', dest='stochastic_feature_dist', default='norm', type=str)
    parser.add_argument('--train_size', dest='train_size', default=5, type=int)
    parser.add_argument('--val_size', dest='val_size', default=5, type=int)

    parser.set_defaults(gpu=False, cuda='', task=None, model=None, dataset=None,
                        cache=False, rm_feature=False,
                        permute=True,
                        feature_pre=True, dropout=True,
                        fixed_noise=False,
                        approximate=0,
                        resample_neg=False)
    args = parser.parse_args()
    print(args)
    if args.model == 'PGNN':
        assert args.permute
        if args.approximate == 0:
            args.approximate = 2
    return args
