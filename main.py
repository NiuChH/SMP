import json
import logging
import os
import time
from pprint import pformat

import torch.optim.lr_scheduler
from tensorboardX import SummaryWriter

from args import make_args
from dataset_utils import get_tg_dataset, TASK_TO_DATASET_DICT
# noinspection PyUnresolvedReferences
from models.adj_models import *
# noinspection PyUnresolvedReferences
from models.model import *
from train_steps import train_step, test_step
from utils.arg_helper import set_seed_and_logger, get_config, mkdir
from utils.load_helper import log_model_params
# args
from utils.pgnn_utils import preselect_anchor
import numpy as np


def format_hits(hits):
    return ', '.join([f'{k:03d}: {v:.4f}' for k, v in hits.items()])


def run_train(args, device, repeat_id=0):
    dataset_name, task = args.dataset, args.task

    writer_train = SummaryWriter(
        comment=args.task + '_' + args.model + '_' + args.comment + '_' + args.result_dir + '_train')
    writer_val = SummaryWriter(
        comment=args.task + '_' + args.model + '_' + args.comment + '_' + args.result_dir + '_val')
    writer_test = SummaryWriter(
        comment=args.task + '_' + args.model + '_' + args.comment + '_' + args.result_dir + '_test')

    run_tag_str = f"{dataset_name}_{args.model}_{repeat_id}"
    result_val = []
    result_dicts = []
    data_list = get_tg_dataset(args, dataset_name, remove_feature=args.rm_feature)

    num_features = data_list[0].x.shape[1]
    num_node_classes = None
    num_graph_classes = None
    if hasattr(data_list[0], 'y') and data_list[0].y is not None:
        num_node_classes = max([data.y.max().item() for data in data_list]) + 1
    if hasattr(data_list[0], 'y_graph') and data_list[0].y_graph is not None:
        num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list]) + 1
    logging.info(f"Dataset: {dataset_name}|"
                 f"Graph: {len(data_list)}|"
                 f"Feature: {num_features}|"
                 f"Node Class: {num_node_classes}|"
                 f"Graph Class: {num_graph_classes}|")
    nodes = [data.num_nodes for data in data_list]
    edges = [data.num_edges for data in data_list]
    logging.info('Node: max{}, min{}, mean{}'.format(max(nodes), min(nodes), sum(nodes) / len(nodes)))
    logging.info('Edge: max{}, min{}, mean{}'.format(max(edges), min(edges), sum(edges) / len(edges)))

    args.batch_size = min(args.batch_size, len(data_list))
    logging.info('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

    # model
    input_dim = num_features
    if 'node_cls' in task:
        output_dim = num_node_classes
        predictor = None
    elif 'node_reg' in task:
        output_dim = 1
        predictor = None
    else:
        output_dim = args.output_dim
        if args.use_predictor:
            if args.last_layer == 'None':
                output_dim += args.stochastic_dim
            predictor = SimpleMLP(input_dim=output_dim,
                                  hidden_dim=256,
                                  output_dim=1,
                                  layer_num=3, dropout=args.dropout).to(device)
            log_model_params(predictor)
        else:
            def predictor(x):
                return torch.sum(x, dim=-1)

    model = eval(args.model)(input_dim=input_dim, feature_dim=args.feature_dim,
                             hidden_dim=args.hidden_dim, output_dim=output_dim,
                             feature_pre=args.feature_pre, layer_num=args.layer_num,
                             dropout=args.dropout,
                             stochastic_dim=args.stochastic_dim,
                             task=args.task,
                             fixed_noise=args.fixed_noise,
                             last_layer=args.last_layer,
                             gnn_type=args.gnn_type,
                             heads=args.heads,
                             stochastic_feature_dist=args.stochastic_feature_dist).to(device)
    model.device = device
    log_model_params(model)

    # data
    for i, data in enumerate(data_list):
        if isinstance(model, PGNN):
            preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
        data.x = data.x.to(torch.float).to(device)
        if hasattr(data, 'y') and data.y is not None:
            data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)
        data_list[i] = data
    maybe_test_len = int(args.test_split * len(data_list))
    maybe_train_len = len(data_list) - 2 * maybe_test_len
    if maybe_test_len < 1 or maybe_train_len < 1:
        # to train and test all graphs
        train_data_list, val_data_list, test_data_list = data_list, data_list, data_list
    else:
        # to split the graphs to distinct sets
        val_len = maybe_test_len
        train_len = maybe_train_len
        train_data_list = data_list[:train_len]
        val_data_list = data_list[train_len:train_len + val_len]
        test_data_list = data_list[train_len + val_len:]
    args.batch_size = min(args.batch_size, len(train_data_list))

    # loss
    param_list = list(model.parameters())
    if isinstance(predictor, nn.Module):
        param_list = param_list + list(predictor.parameters())
    if len(param_list) == 0:
        optimizer = None
    else:
        optimizer = torch.optim.Adam(param_list, lr=args.lr,
                                     # weight_decay=5e-4,
                                     weight_decay=0 if args.dataset.startswith('ogbl') else 5e-4
                                     )

    if 'link' in task:  # 'link' or 'link_pair'
        loss_func = nn.BCEWithLogitsLoss()
    elif 'node_cls' in task:  # 'node_cls'
        loss_func = nn.NLLLoss()
    elif 'node_reg' in task:
        loss_func = nn.MSELoss()
    else:
        raise NotImplementedError()

    train_time_list = []
    test_time_list = []

    for epoch in range(args.epoch_num):
        train_start_time = time.time()
        train_step(args, epoch, train_data_list, model, loss_func, optimizer, 'train', predictor=predictor)
        train_time_list.append(time.time() - train_start_time)

        if epoch % args.epoch_log == 0:
            # evaluate
            test_start_time = time.time()
            with torch.no_grad():
                loss_train, auc_train, hits_train = test_step(args, train_data_list, model, loss_func, 'train',
                                                              predictor=predictor)
                loss_val, auc_val, hits_val = test_step(args, val_data_list, model, loss_func, 'val',
                                                        predictor=predictor)
                loss_test, auc_test, hits_test = test_step(args, test_data_list, model, loss_func, 'test',
                                                           predictor=predictor)
            test_time_list.append(time.time() - test_start_time)
            mean_train_time = np.mean(train_time_list)
            mean_test_time = np.mean(test_time_list)
            eta = (mean_test_time + args.epoch_log * mean_train_time) / args.epoch_log * (args.epoch_num - epoch)

            logging.info(f'{run_tag_str}{args.comment}|{epoch}|'
                         f'ETA: {time.strftime("%H:%M:%S", time.gmtime(eta))}|'
                         f'Loss: {loss_train:.4f}|'
                         f'Train: {auc_train:.4f}|'
                         f'Val: {auc_val:.4f}|'
                         f'Test: {auc_test:.4f}|\n'
                         f'Train Hits: {format_hits(hits_train)}|\n'
                         f'Val Hits: {format_hits(hits_val)}|\n'
                         f'Test Hits: {format_hits(hits_test)}|'
                         )
            writer_train.add_scalar('repeat_' + str(repeat_id) + '/auc_' + dataset_name, auc_train, epoch)
            writer_train.add_scalar('repeat_' + str(repeat_id) + '/loss_' + dataset_name, loss_train, epoch)

            writer_val.add_scalar('repeat_' + str(repeat_id) + '/auc_' + dataset_name, auc_val, epoch)
            writer_val.add_scalar('repeat_' + str(repeat_id) + '/loss_' + dataset_name, loss_val, epoch)

            writer_test.add_scalar('repeat_' + str(repeat_id) + '/auc_' + dataset_name, auc_test, epoch)
            writer_test.add_scalar('repeat_' + str(repeat_id) + '/loss_' + dataset_name, loss_test, epoch)

            res_dict = {
                'auc_val': auc_val,
                'auc_test': auc_test,
                'auc_train': auc_train,
                'loss_val': loss_val,
                'loss_test': loss_test,
                'loss_train': loss_train,
            }
            res_dict.update({
                f'train_hit{k}': float(v) for k, v in hits_train.items()
            })
            res_dict.update({
                f'test_hit{k}': float(v) for k, v in hits_test.items()
            })
            res_dict.update({
                f'val_hit{k}': float(v) for k, v in hits_val.items()
            })

            result_dicts.append(res_dict)

            result_val.append(auc_val)

    # export scalar data to JSON for external processing
    writer_train.export_scalars_to_json("./all_scalars.json")
    writer_train.close()
    writer_val.export_scalars_to_json("./all_scalars.json")
    writer_val.close()
    writer_test.export_scalars_to_json("./all_scalars.json")
    writer_test.close()
    # return result_dicts[-1]  # not early stop
    result_val = np.array(result_val)
    if 'node_reg' in args.task:
        early_stop_result = result_dicts[np.argmin(result_val).item()]
    else:
        early_stop_result = result_dicts[np.argmax(result_val).item()]
    early_stop_result.update({
        'train_time': np.mean(train_time_list[3:]),
        'test_time': np.mean(test_time_list[3:]),
        'final_loss': float(result_dicts[-1]['loss_train']),
        'final_auc': float(result_dicts[-1]['auc_train']),
    })
    return early_stop_result  # early stop


def main(args):
    mkdir(args.result_dir)
    args = get_config(args)
    set_seed_and_logger(args)

    if args.dataset.startswith('ogbl'):
        assert args.task == 'link'
        if args.edge_batch_size is None:
            args.edge_batch_size = 16 * 1024
        args.batch_size = 1
        args.resample_neg = True
    # else:
    #     args.edge_batch_size = None

    logging.info('config: \n' + pformat(args))
    # set up gpu
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        logging.info('Using CPU')
    device = torch.device('cuda:' + str(args.cuda) if args.gpu else 'cpu')

    if args.dataset == 'All':
        dataset_name_list = TASK_TO_DATASET_DICT[args.task]
    else:
        # assert args.dataset in TASK_TO_DATASET_DICT[args.task]
        dataset_name_list = [args.dataset]

    # torch.autograd.set_detect_anomaly(True)

    for dataset_name in dataset_name_list:
        args.dataset = dataset_name
        results = []
        for repeat_id in range(args.repeat_num):
            results.append(run_train(args, device, repeat_id))
            torch.cuda.empty_cache()
        final_result = {k: [r[k] for r in results] for k in results[0].keys()}
        for k, v in final_result.items():
            final_result[k] = (np.mean(v).item(), np.std(v).item())

        logging.info('-----------------Final-------------------')
        logging.info('config: \n' + pformat(args))
        logging.info(json.dumps(final_result, indent=2))

        final_result.update(args.__dict__)
        with open(
                args.result_dir +
                '/{}_{}_{}_layer{}_{}_{}_{}.txt'.format(
                    args.task, args.model,
                    dataset_name, args.layer_num,
                    args.comment,
                    args.run_id,
                    np.random.randint(0, 65536)),
                'w') as f:
            f.write(json.dumps(final_result, indent=2, sort_keys=True))
        if len(dataset_name_list) == 1:
            return final_result


if __name__ == "__main__":
    ori_args = make_args()
    main(ori_args)
