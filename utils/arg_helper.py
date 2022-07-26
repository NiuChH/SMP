import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict
import shutil
import glob

SEED = 0


def parse_arguments(default_config="config/train.yaml"):
    parser = argparse.ArgumentParser(description="Running Experiments")
    parser.add_argument('-c', '--config_file', type=str,
                        default=default_config, help="Path of config file")
    parser.add_argument('-l', '--log_level', type=str,
                        default='INFO', help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument('-d', '--dataset', type=str,
                        default=None, help="dataset name")
    parser.add_argument('-f', '--file', type=str,
                        default=None, help="file name of model")
    parser.add_argument('-m', '--comment', type=str,
                        default="", help="Experiment comment")
    args = parser.parse_args()

    return args


def get_config(args):
    """ Construct and snapshot hyper parameters """
    config = EasyDict(yaml.load(open(args.config_file, 'r')))
    config.update(args.__dict__)
    # noinspection PyTypeChecker
    process_config(config, comment=args.comment)
    return config


def process_config(config, comment=''):
    # create hyper parameters
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.comment = comment
    # config.dev = dev
    config.run_id = str(os.getpid())
    config.folder_name = '_'.join([
        config.model, config.dataset, comment.replace(' ', '_'),
        time.strftime('%b-%d-%H-%M-%S'), config.run_id
    ])

    if 'save_dir' not in config:
        config.save_dir = os.path.join(config.exp_dir, config.exp_name, config.folder_name)
        config.model_save_dir = os.path.join(config.save_dir, 'models')

    # config.dataset.stochastic_dim = config.model.options.stochastic_dim

    mkdir(config.exp_dir)
    mkdir(config.save_dir)
    mkdir(config.model_save_dir)

    # snapshot hyper-parameters and code
    if config.save_code:
        code_dir = config.save_dir + '/code'
        mkdir(code_dir)
        COPY_DIRS = ['utils', 'models']
        for d in COPY_DIRS:
            shutil.copytree(f'./{d}', os.path.join(code_dir, d))
        for f in glob.glob('./*.py'):
            shutil.copy(f, code_dir)
    save_name = os.path.join(config.save_dir, 'train.yaml')
    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)
    return config


def edict2dict(edict_obj):
    dict_obj = {}
    for key, vals in edict_obj.items():
        if isinstance(vals, EasyDict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals
    return dict_obj


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def set_seed():
    global SEED
    random.seed(SEED)
    np.random.RandomState(SEED)
    torch.manual_seed(SEED)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(SEED)


def set_seed_and_logger(args):
    global SEED
    SEED = args.seed
    random.seed(args.seed)
    np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(args.seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    log_file = os.path.join(args.save_dir, args.log_level.lower() + ".log")
    FORMAT = '|%(asctime)s|%(message)s'
    fh = logging.FileHandler(log_file)
    fh.setLevel(args.log_level)
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.DEBUG, format=FORMAT,
                        datefmt='%m-%d %H:%M:%S',
                        handlers=[
                            fh,
                            logging.StreamHandler(sys.stdout)
                        ])
    logging.info('EXPERIMENT BEGIN: ' + args.comment)
    logging.info('logging into %s', log_file)
