import sys
import os
import time
import datetime
import random
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader

sys.path.insert(0, './dynamic-gcn/')

from preparation.preprocess_dataset import load_resource_labels
from preparation.preprocess_dataset import load_resource_trees

from tools.random_folds import load_k_fold_train_val_test
from tools.random_folds import print_folds_labels

from tools.early_stopping import EarlyStopping
from tools.evaluation import evaluation
from tools.evaluation import merge_batch_eval_list
from dataset import GraphSnapshotDataset

from utils import print_dict
from utils import save_json_file
from utils import append_json_file
from utils import load_json_file
from utils import ensure_directory

from model import Network

# TODO:

device = torch.device("cuda:1")
settings = {}
settings['snapshot_num'] = 3
settings['cuda'] = "cuda:1"
settings['learning_sequence'] = 'dot_product'

model = Network(5000, 64, 64, settings).to(device)

model.load_state_dict(torch.load("./results/GCN_Twitter16_sequential_dot_product_3_2020_0723_2319_model.pt"))
model.to(device)



