from random import shuffle

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import PGNN
from utils.pgnn_utils import preselect_anchor, get_random_edge_mask_link_as_neg


def eval_hits(y_pred_pos, y_pred_neg, K=100):
    """
        compute Hits@K
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    """

    if len(y_pred_neg) < K:
        return 1.
    if not isinstance(y_pred_pos, torch.Tensor):
        y_pred_pos = torch.tensor(y_pred_pos)
        y_pred_neg = torch.tensor(y_pred_neg)
    y_pred_pos, y_pred_neg = y_pred_pos.squeeze(), y_pred_neg.squeeze()
    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    return hitsK


def get_label_and_pred(model_out, data, phase, task='link', return_embed=False,
                       predictor=None, MAX_PRED_LEN=64 * 1024):
    device = model_out.device
    assert phase in ('train', 'test', 'val')
    hits = {
        10: 0,
        50: 0,
        100: 0,
    }
    if 'link' in task:
        mask_link_positive = getattr(data, f'mask_link_positive_{phase}')
        mask_link_negative = getattr(data, f'mask_link_negative_{phase}')
        assert predictor is not None
        if not isinstance(mask_link_positive, torch.Tensor):
            mask_link_positive = torch.tensor(mask_link_positive, dtype=torch.long, device=model_out.device)
        else:
            mask_link_positive = mask_link_positive.to(device)
        if not isinstance(mask_link_negative, torch.Tensor):
            mask_link_negative = torch.tensor(mask_link_negative, dtype=torch.long, device=model_out.device)
        else:
            mask_link_negative = mask_link_negative.to(device)

        label_positive = torch.ones([mask_link_positive.shape[1], ], dtype=torch.float32, device=model_out.device)
        label_negative = torch.zeros([mask_link_negative.shape[1], ], dtype=torch.float32, device=model_out.device)
        tmp_label = torch.cat((label_positive, label_negative))

        edge_mask = torch.cat((mask_link_positive, mask_link_negative), dim=-1)
        if isinstance(predictor, torch.nn.Module):
            normalized_out = model_out
        else:
            normalized_out = F.normalize(model_out, p=2, dim=-1)
        if edge_mask.size(1) <= MAX_PRED_LEN:
            nodes_first = normalized_out[edge_mask[0]]
            nodes_second = normalized_out[edge_mask[1]]
            pred = predictor(nodes_first * nodes_second).squeeze(-1)
            label = tmp_label
        else:
            pred_list = []
            label_list = []
            # noinspection PyTypeChecker
            for perm in DataLoader(range(edge_mask.size(1)), MAX_PRED_LEN):
                nodes_first = normalized_out[edge_mask[0, perm]]
                nodes_second = normalized_out[edge_mask[1, perm]]
                tmp_pred = predictor(nodes_first * nodes_second).squeeze(-1)
                pred_list.append(tmp_pred)
                label_list.append(tmp_label[perm])
            pred = torch.cat(pred_list, dim=0)
            label = torch.cat(label_list, dim=0)
        for k in hits.keys():
            hits[k] = eval_hits(y_pred_pos=pred[:mask_link_positive.size(1)],
                                y_pred_neg=pred[mask_link_positive.size(1):],
                                K=k)

    elif 'node_cls' in task:
        mask = getattr(data, f'{phase}_mask')
        pred = model_out
        pred = F.log_softmax(pred, dim=-1)[mask]
        label = data.y[mask]
    elif 'node_reg' in task:
        mask = getattr(data, f'{phase}_mask')
        pred = F.selu(model_out[mask])
        label = data.y[mask].unsqueeze(-1)
    else:  # 'graph' in task
        raise NotImplementedError()
    return label, pred, hits


def get_metric(pred, label, task):
    if 'link' in task:
        return roc_auc_score(label.flatten().cpu().numpy(),
                             torch.sigmoid(pred).flatten().data.cpu().numpy())
    elif 'node_cls' in task:
        return pred.max(1)[1].eq(label).sum().item() / pred.size(0)


def train_step(args, epoch, train_data_list, model, loss_func, optimizer,
               phase_tag='train', predictor=None):
    param_list = list(model.parameters())
    if isinstance(predictor, torch.nn.Module):
        param_list = param_list + list(predictor.parameters())
    model.train()
    if isinstance(predictor, torch.nn.Module):
        predictor.train()
    if optimizer is not None:
        if epoch == 200:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        optimizer.zero_grad()
    shuffle(train_data_list)
    for i_data, data in enumerate(train_data_list):
        if args.permute and isinstance(model, PGNN):
            preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=model.device)
        if args.edge_batch_size is not None:
            assert args.task == 'link' and 'ogbl' in args.dataset
            pos_train_edge = getattr(data, f'mask_link_positive_{phase_tag}')
            # noinspection PyTypeChecker

            for perm in tqdm(DataLoader(
                    range(pos_train_edge.shape[1]),
                    args.edge_batch_size, shuffle=True), desc='train: '):
                optimizer.zero_grad()
                h = model(data)
                pos_edge = pos_train_edge[:, perm]
                neg_edge = get_random_edge_mask_link_as_neg(
                    data.edge_index,
                    data.num_nodes,
                    pos_edge.shape[1])
                pos_pred = predictor(h[pos_edge[0]] * h[pos_edge[1]]).squeeze(-1)
                neg_pred = predictor(h[neg_edge[0]] * h[neg_edge[1]]).squeeze(-1)
                label = torch.cat((torch.ones_like(pos_pred), torch.zeros_like(neg_pred)), dim=0)
                pred = torch.cat((pos_pred, neg_pred), dim=0)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label)
                # print(len(perm), loss.requires_grad)
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(param_list, 1.0)
                    optimizer.step()
        else:
            if args.resample_neg and 'link' in args.task:
                data.mask_link_negative_train = get_random_edge_mask_link_as_neg(
                    data.edge_index,
                    data.num_nodes,
                    data.mask_link_positive_train.shape[1])
            out = model(data)
            label, pred, _ = get_label_and_pred(out, data, phase_tag, task=args.task, predictor=predictor)
            loss = loss_func(pred, label)

            # update
            if loss.requires_grad:
                loss.backward()
            else:
                # for non-trainable model
                continue
            optimizer.step()
            optimizer.zero_grad()


def test_step(args, data_list, model, loss_func,
              phase_tag, predictor=None):
    model.eval()
    if isinstance(predictor, torch.nn.Module):
        predictor.eval()
    loss, metric = 0, 0
    hits = {
        10: 0,
        50: 0,
        100: 0,
    }
    for i_data, data in enumerate(data_list):
        out = model(data)

        if args.dataset.startswith('ogbl'):
            # assert evaluator is not None
            pos_edge = getattr(data, f'mask_link_positive_{phase_tag}')
            if phase_tag == 'train':
                neg_edge = data.mask_link_negative_val
            else:
                neg_edge = getattr(data, f'mask_link_negative_{phase_tag}')

            def get_pred(edges):
                pred_list = []
                # noinspection PyTypeChecker
                for perm in tqdm(DataLoader(range(edges.size(1)), args.edge_batch_size), desc='test: '):
                    edge = edges[:, perm]
                    pred_list += [predictor(out[edge[0]] * out[edge[1]]).squeeze().cpu()]
                return torch.cat(pred_list, dim=0)

            pos_pred = get_pred(pos_edge)
            neg_pred = get_pred(neg_edge)

            label = torch.cat((torch.ones_like(pos_pred), torch.zeros_like(neg_pred)), dim=0)
            for k in hits.keys():
                hits[k] += eval_hits(y_pred_pos=pos_pred, y_pred_neg=neg_pred, K=k)

            if 'node_reg' in args.task:
                # Since the scales of different centralities are significantly different,
                # we rescale the MSE value by dividing it by the corresponding averaged
                # centrality values of all nodes.
                metric += (loss / data.y.mean().cpu().data.numpy())
            else:
                metric += get_metric(pred=torch.cat((pos_pred.sigmoid(), neg_pred.sigmoid()), dim=0),
                                     label=label, task='link')

            pred = torch.cat((pos_pred, neg_pred), dim=0)
            loss += torch.nn.functional.binary_cross_entropy_with_logits(pred, label)
        else:
            label, pred, data_hits = get_label_and_pred(out, data, phase_tag,
                                                        task=args.task, predictor=predictor)
            loss += loss_func(input=pred, target=label).cpu().data.numpy()
            if 'node_reg' in args.task:
                # Since the scales of different centralities are significantly different,
                # we rescale the MSE value by dividing it by the corresponding averaged
                # centrality values of all nodes.
                metric += (loss / data.y.mean().cpu().data.numpy())
            else: # 'link*' and 'node_cls'
                metric += get_metric(pred=pred, label=label, task=args.task)
            for k in hits.keys():
                hits[k] += data_hits[k]
    loss /= len(data_list)
    metric /= len(data_list)
    for k in hits.keys():
        hits[k] /= len(data_list)

    return loss, metric, hits
