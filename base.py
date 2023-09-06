"""
This file provides basic classes and functions to accelerate the development of deep learning tasks.
Some APIs are designed by taking 'pytorch-lightning' and 'transformers' packages as references.
"""

import os
import json
import sys
import torch
import random
import shutil
import numpy as np
import logging as lg
from tqdm import tqdm
from abc import ABC, abstractmethod
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from collections.abc import Iterable
from tensorboardX import SummaryWriter


lg.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=lg.INFO)

# keys to obtain batch output info
LOSS_KEY = 'loss'  # for training loss
BAR_KEY = 'progress_bar'  # for progress bar postfix
SCALAR_LOG_KEY = 'scalar_log'  # for tensorboard
VAL_SCORE_KEY = 'val_score'  # for choosing the best checkpoint
# directories to save checkpoints, model outputs, and tensorboard summaries
CPT_DIR_NAME = 'Checkpoint'  # for checkpoints
OUT_DIR_NAME = 'Output'  # for model outputs
LOG_DIR_NAME = 'Tensorboard'  # for tensorboard summaries
# runtime environment variables
RUNTIME_LOG_DIR = 'RUNTIME_LOG_DIR'  # logging tensorboard info into a runtime directory, and copy to exp dir at last
RUNTIME_MODEL_DIR = 'RUNTIME_MODEL_DIR'  # dumping model checkpoints into a runtime directory, and copy to exp dir at last
# TODO: impl logics for runtime model directory

# Helper functions
def dump_json_to(obj, fpath, indent=2, ensure_ascii=False, **kwargs):
    """The helper for dumping json into the given file path"""
    with open(fpath, 'w') as fout:
        json.dump(obj, fout, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def load_json_from(fpath, **kwargs):
    """The helper for loading json from the given file path"""
    with open(fpath, 'r') as fin:
        obj = json.load(fin, **kwargs)

    return obj


def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif str_val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))


def add_config_to_argparse(config, arg_parser):
    """The helper for adding configuration attributes to the argument parser"""
    for key, val in config.to_dict().items():
        if isinstance(val, bool):
            arg_parser.add_argument('--' + key, type=strtobool, default=val)
        elif isinstance(val, (int, float, str)):
            arg_parser.add_argument('--' + key, type=type(val), default=val)
        else:
            raise Exception('Do not support value ({}) type ({})'.format(val, type(val)))


class BaseConfig:
    """The config class to set up the task"""

    def __init__(self):
        # the maximum number of epochs to run
        self.max_epochs = 1000
        # the initial random seed for data sampler and parameter initialization
        self.random_seed = 0
        # the number of gradient accumulation steps (for large-batch training with limited memory)
        self.grad_accum_steps = 1
        # whether to use cuda for GPU training
        self.use_cuda = True
        # -1: single-gpu training, positive integer: the local rank for distributed training
        self.local_rank = -1
        # distributed bankend, 'gloo' is slower than 'nccl'
        self.dist_backend = 'nccl'
        # only permit the master node to log information
        self.only_master_log = True
        # the directory to save checkpotins and model outputs
        self.exp_dir = os.path.join(os.getcwd(), 'Exp')
        # the file name of the config json
        self.cfg_fname = 'config.json'
        # the file name of the model checkpoint
        self.cpt_fname = 'model.cpt'
        # the file name of the model output
        self.out_fname = 'out.cpt'
        # Options for dumping checkpoints and model outputs
        # 0: no dump, 1: latest, 2: latest & best, 3: latest & best & every epoch
        self.dump_option = 2
        # whether trying to recover from the latest checkpoints before the start of training
        self.try_recover = False
        # whether to skip training
        self.skip_train = False
        # whether to evaluate on test set during the model fitting
        self.eval_test_dur_fit = False
        # enable early stopping
        self.enable_early_stop = True
        # the maximum number of epochs with no validation improvement before early stopping
        self.early_stop_epochs = 30
        # for DistributedDataParallel
        self.find_unused_parameters = False

    def update_by_dict(self, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)

    def to_dict(self):
        return dict(self.__dict__)
