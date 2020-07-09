import sys
import os
import time
import datetime
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from preparation.prepare_snapshots import load_trees
from preparation.prepare_snapshots import load_labels

from tools.random_folds import load_k_fold_train_val_test
from tools.random_folds import count_folds_labels

from tools.early_stopping import EarlyStopping
from tools.evaluation import evaluation
from tools.evaluation import merge_batch_eval_list
from dataset import GraphSnapshotDataset


# from project_settings import *
from utils import print_dict
from utils import save_json_file
from utils import load_json_file
from utils import ensure_directory


def write_results(string):  # TODO:
    with open(RESULTS_FILE, 'a') as out_file:
        out_file.write(str(string) + '\n')

def append_json_file(path, data):
    # ensure_directory(path)
    with open(path, 'a') as json_file:
        json_file.write(json.dumps(data))

# TODO:
arg_names = [
    'command',
    'dataset_name', 'dataset_type', 'snapshot_num',
    'learning_sequence', 'model'
]
args = dict(zip(arg_names, sys.argv))
print("----------------")
print(args)
print("----------------")


"""
model = 'GCN'  # TODO:
dataset_name = args['dataset_name']
dataset_type = args['dataset_type']
snapshot_num = args['snapshot_num']
learning_sequence = args['learning_sequence']
# TODO: CUDA
"""
# -----------------
#     OPTION 1)
# -----------------
# from model import Network  # Dev

from model_mean_sum_concat import Network  # GCN (ICLR 2017)


# from model_additive_attention_gcn import Network  # GCN (ICLR 2017)
# from model_additive_attention_sage import Network  # GraphSAGE (NIPS 2017)
# from model_additive_attention_gin import Network  # Graph Isomorphic Network (ICLR 2019)


# from model_self_attention_gcn import Network  # <== # TODO: HERE
# from model_self_attention_sage import Network  #
# from model_multi_head_attention import Network  #
# from model_additive_attention_gcn import Network  # GCN (ICLR 2017)
# from model_lstm import Network
# from model_gru import Network
# from model_mean import Network
# from model_concat_fc import Network

# TODO: OPT
# from model_self_attention_gcn_opt import Network  #


# =======================
#     Variations
# =======================

# TODO: argv
# dataset = sys.argv[1]  # "Twitter15"、"Twitter16"
# iterations = int(sys.argv[2])

model = "GCN"  # GCN, GraphSAGE, GIN
# model = "GraphSAGE"  # GCN, GraphSAGE, GIN
# dataset_name = "Twitter16"  # Twitter15, Twitter16
dataset_name = "Twitter15"  # Twitter15, Twitter16

dataset_type = "sequence"  # sequence, temporal
# dataset_type = "temporal"  # sequence, temporal


sequence_learning_type = "self_attention"  # additive_attention, self_attention
# sequence_learning_type = "additive_attention"  # additive_attention, self_attention
# sequence_learning_type = "multi_head_attention"  # additive_attention, self_attention
# sequence_learning_type = "LSTM"  # additive_attention, self_attention
# sequence_learning_type = "GRU"  # additive_attention, self_attention
# sequence_learning_type = "mean"  # additive_attention, self_attention
# sequence_learning_type = "concat"

# snapshot_num = 2
snapshot_num = 3
# snapshot_num = 5
# snapshot_num = 8



current = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

ensure_directory("./results")
RESULTS_FILE = "./results/{0}_{1}_{2}_{3}_{4}_{5}_results.txt".format(
    dataset_name, model, dataset_type, sequence_learning_type, snapshot_num, current
)
FOLDS_FILE = "./results/{0}_{1}_{2}_{3}_{4}_{5}_folds.json".format(
    dataset_name, model, dataset_type, sequence_learning_type, snapshot_num, current
)
MODEL_PATH = "./results/{0}_{1}_{2}_{3}_{4}_{5}_model.pt".format(
    dataset_name, model, dataset_type, sequence_learning_type, snapshot_num, current
)



# =======================
#     Hyperparameters
# =======================

iterations = 2
iterations = 10
num_epochs = 200
batch_size = 150
batch_size = 20
# batch_size = 64

lr = 0.0005
# lr = 0.0003

weight_decay = 1e-4
patience = 10
# patience = 20  # TODO:

td_droprate = 0.2
bu_droprate = 0.2
info = {
    'iterations': iterations,
    'num_epochs': num_epochs, 'batch_size': batch_size,
    'lr': lr, 'weight_decay': weight_decay, 'patience': patience,
    'td_droprate': td_droprate, 'bu_droprate': bu_droprate,
    "model": model, "dataset_name": dataset_name, "dataset_type": dataset_type,
    "sequence_learning_type": sequence_learning_type, "snapshot_num": snapshot_num,
    "td_droprate": td_droprate, "bu_droprate":bu_droprate,
    "current": current,
    "sys.argv": sys.argv,
}
write_results(info)  # Dev

counters = {'iter': 0, 'CV': 0}
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

"""
def load_snapshot_dataset(dataset_name, tree_dict, fold_x_train, fold_x_val, fold_x_test):
    data_path = "./data/graph/{0}/{1}_snapshot/".format(dataset_name, dataset_type)
    train_dataset = GraphSnapshotDataset(
        tree_dict, fold_x_train, data_path=data_path, snapshot_num=snapshot_num,
        td_droprate=td_droprate, bu_droprate=bu_droprate,
    )
    val_dataset = GraphSnapshotDataset(
        tree_dict, fold_x_val, data_path=data_path, snapshot_num=snapshot_num)

    test_dataset = GraphSnapshotDataset(
        tree_dict, fold_x_test, data_path=data_path, snapshot_num=snapshot_num
    )
    print("train count:", len(train_dataset))
    print("val count:", len(val_dataset))
    print("test count:", len(test_dataset))
    return train_dataset, val_dataset, test_dataset
"""

def load_snapshot_dataset_train(dataset_name, tree_dict, fold_x_train):
    data_path = "./data/graph/{0}/{1}_snapshot".format(dataset_name, dataset_type)
    train_dataset = GraphSnapshotDataset(
        tree_dict, fold_x_train, data_path=data_path, snapshot_num=snapshot_num,
        td_droprate=td_droprate, bu_droprate=bu_droprate,  # stochastic
    )
    print("train count:", len(train_dataset))
    return train_dataset


def load_snapshot_dataset_val_or_test(dataset_name, tree_dict, fold_x_val_or_test):
    data_path = "./data/graph/{0}/{1}_snapshot".format(dataset_name, dataset_type)
    val_or_test_dataset = GraphSnapshotDataset(
        tree_dict, fold_x_val_or_test, data_path=data_path, snapshot_num=snapshot_num
    )
    print("val or test count:", len(val_or_test_dataset))
    return val_or_test_dataset




def train_GCN(tree_dict, x_train, x_val, x_test, counters):

    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    test_losses, test_accuracies = [], []

    # -------------
    #     MODEL
    # -------------
    model = Network(5000, 64, 64, snapshot_num, device).to(device)
    # model = Network(5000, 128, 128, snapshot_num, device).to(device)

    # -----------------
    #     OPTIMIZER
    # -----------------

    """
    BU_params = []
    for gcn_index in range(snapshot_num):
        gcn = eval("model.rumor_GCN_{0}".format(gcn_index))
        BU_params += list(map(id, gcn.BURumorGCN.conv1.parameters()))
        BU_params += list(map(id, gcn.BURumorGCN.conv2.parameters()))
    base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.rumor_GCN_0.BURumorGCN.conv1.parameters(), 'lr': lr/5},
        {'params': model.rumor_GCN_0.BURumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    """

    BU_params = []
    BU_params += list(map(id, model.rumor_GCN_0.BURumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.rumor_GCN_0.BURumorGCN.conv2.parameters()))
    base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.rumor_GCN_0.BURumorGCN.conv1.parameters(), 'lr': lr/5},
        {'params': model.rumor_GCN_0.BURumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)

    # optimizer = torch.optim.Adam(
    #     [{'params': model.parameters()}],
    #     lr=lr, weight_decay=weight_decay
    # )

    early_stopping = EarlyStopping(patience=patience, verbose=True, model_path=MODEL_PATH)
    # criterion = nn.CrossEntropyLoss()

    val_dataset = load_snapshot_dataset_val_or_test(dataset_name, tree_dict, x_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    for epoch in range(num_epochs):
        # train_dataset, val_dataset, test_dataset = load_snapshot_dataset(dataset_name, tree_dict, x_train, x_val, x_test)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=5)  # TODO: move out epoch
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

        # for dropedge

        with torch.cuda.device(device):
            torch.cuda.empty_cache()  # TODO: CHECK
        train_dataset = load_snapshot_dataset_train(dataset_name, tree_dict, x_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

        # TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: 
        # del train_dataset
        # del train_loader
        # TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: 

        # ---------------------
        #         TRAIN
        # ---------------------
        model.train()
        batch_train_losses = []
        batch_train_accuracies = []

        for batch_index, batch_data in enumerate(train_loader):
            snapshots = []
            for i in range(snapshot_num):
                snapshots.append(batch_data[i].to(device))
            out_labels = model(snapshots)
            loss = F.nll_loss(out_labels, batch_data[0].y)
            del snapshots
            # nn.CrossEntropyLoss = nn.LogSoftmax + nn.NLLLoss
            # loss = criterion(out_labels, batch_data[0].y)  # TODO:

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_train_loss = loss.item()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(batch_data[0].y).sum().item()
            batch_train_acc = correct / len(batch_data[0].y)

            batch_train_losses.append(batch_train_loss)
            batch_train_accuracies.append(batch_train_acc)
            print("Iter {:02d} | CV {:02d} | Epoch {:03d} | Batch {:02d} | Train_Loss {:.4f} | Train_Accuracy {:.4f}".format(
                counters['iter'], counters['CV'], epoch, batch_index, batch_train_loss, batch_train_acc))

        train_losses.append(np.mean(batch_train_losses))  # epoch
        train_accuracies.append(np.mean(batch_train_accuracies))

        del train_dataset
        del train_loader

        # ------------------------
        #         VALIDATE
        # ------------------------
        batch_val_losses = []  # epoch
        batch_val_accuracies = []
        batch_eval_results = []

        model.eval()
        # TODO: no_grad
        # with torch.no_grad():

        for batch_data in val_loader:
            snapshots = []
            for i in range(snapshot_num):
                snapshots.append(batch_data[i].to(device))
            with torch.no_grad():  # HERE
                val_out = model(snapshots)
                val_loss = F.nll_loss(val_out, batch_data[0].y)
            del snapshots

            batch_val_loss = val_loss.item()
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(batch_data[0].y).sum().item()
            batch_val_acc = correct / len(batch_data[0].y)

            batch_val_losses.append(batch_val_loss)
            batch_val_accuracies.append(batch_val_acc)
            batch_eval_results.append(evaluation(val_pred, batch_data[0].y))

        validation_losses.append(np.mean(batch_val_losses))
        validation_accuracies.append(np.mean(batch_val_accuracies))
        validation_eval_result = merge_batch_eval_list(batch_eval_results)

        print("---------------------" * 3)
        print("eval_result:", validation_eval_result)
        print("---------------------" * 3)

        print("Iter {:03d} | CV {:02d} | Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(
            counters['iter'], counters['CV'], epoch, np.mean(batch_val_losses), np.mean(batch_val_accuracies))
        )
        # write_results("eval result: " + str(eval_result))
        write_results(
            "Iter {:03d} | CV {:02d} | Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(
                counters['iter'], counters['CV'], epoch, np.mean(batch_val_losses), np.mean(batch_val_accuracies)
            )
        )

        early_stopping(validation_losses[-1], model, 'BiGCN', dataset_name, validation_eval_result)
        if early_stopping.early_stop:
            print("Early Stopping")
            validation_eval_result = early_stopping.eval_result
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            break

    del val_dataset
    del val_loader

    # --------------------
    #         TEST
    # --------------------

    with torch.cuda.device(device):
        torch.cuda.empty_cache()  # TODO: CHECK
    test_dataset = load_snapshot_dataset_val_or_test(dataset_name, tree_dict, x_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    batch_test_losses = []  # epoch
    batch_test_accuracies = []
    batch_test_eval_results = []

    model.eval()
    for batch_data in test_loader:
        snapshots = []
        for i in range(snapshot_num):
            snapshots.append(batch_data[i].to(device))

        with torch.no_grad():
            test_out = model(snapshots)  # early stopped model
            test_loss = F.nll_loss(test_out, batch_data[0].y)
        del snapshots

        batch_test_loss = test_loss.item()
        _, test_pred = test_out.max(dim=1)

        correct = test_pred.eq(batch_data[0].y).sum().item()
        batch_test_acc = correct / len(batch_data[0].y)

        batch_test_losses.append(batch_test_loss)
        batch_test_accuracies.append(batch_test_acc)
        batch_test_eval_results.append(evaluation(test_pred, batch_data[0].y))

    test_losses.append(np.mean(batch_test_losses))
    test_accuracies.append(np.mean(batch_test_accuracies))
    test_eval_result = merge_batch_eval_list(batch_test_eval_results)

    accs = test_eval_result['acc_all']
    F0 = test_eval_result['C0'][3]
    F1 = test_eval_result['C1'][3]
    F2 = test_eval_result['C2'][3]
    F3 = test_eval_result['C3'][3]

    counters['CV'] += 1
    losses = [train_losses, validation_losses, test_losses]
    accuracies = [train_accuracies, validation_accuracies, test_accuracies]

    write_results("losses: " + str(losses))
    write_results("accuracies: " + str(accuracies))
    write_results("val eval result: " + str(validation_eval_result))
    write_results("test eval result: " + str(test_eval_result))
    print("Test_Loss {:.4f} | Test_Accuracy {:.4f}".format(
        np.mean(test_losses), np.mean(test_accuracies))
    )

    return losses, accuracies, accs, [F0, F1, F2, F3]


def main():

    # dataset_name = sys.argv[1]  # "Twitter15"、"Twitter16"
    # iterations = int(sys.argv[2])

    TREE_PATH = './resources/BiGCN/{0}/data.TD_RvNN.vol_5000.txt'.format(dataset_name)
    LABEL_PATH = './resources/BiGCN/{0}/{0}_label_All.txt'.format(dataset_name)
    tree_dict = load_trees(TREE_PATH)
    id_label_dict, _ = load_labels(LABEL_PATH)

    test_accs = []
    TR_F1, FR_F1, UN_F1, NR_F1 = [], [], [], []  # F1 score

    for iter_counter in range(iterations):
        counters['iter'] = iter_counter
        folds = load_5_fold_data_train_val_test_sets(dataset_name)

        append_json_file(FOLDS_FILE, folds)  # Dev

        fold0_x_train, fold0_x_val, fold0_x_test = folds[0][0], folds[1][0], folds[2][0]
        fold1_x_train, fold1_x_val, fold1_x_test = folds[0][1], folds[1][1], folds[2][1]
        fold2_x_train, fold2_x_val, fold2_x_test = folds[0][2], folds[1][2], folds[2][2]
        fold3_x_train, fold3_x_val, fold3_x_test = folds[0][3], folds[1][3], folds[2][3]
        fold4_x_train, fold4_x_val, fold4_x_test = folds[0][4], folds[1][4], folds[2][4]

        for fold_index in range(5):
            counts = count_fold_labels_train_val_test(
                id_label_dict, folds[0][fold_index], folds[1][fold_index], folds[2][fold_index]
            )
            print(fold_index, counts)

        _, _, accs_f0, F1_f0 = train_GCN(tree_dict, fold0_x_train, fold0_x_val, fold0_x_test, counters)  # fold 0
        _, _, accs_f1, F1_f1 = train_GCN(tree_dict, fold1_x_train, fold1_x_val, fold1_x_test, counters)  # fold 1
        _, _, accs_f2, F1_f2 = train_GCN(tree_dict, fold2_x_train, fold2_x_val, fold2_x_test, counters)  # fold 2
        _, _, accs_f3, F1_f3 = train_GCN(tree_dict, fold3_x_train, fold3_x_val, fold3_x_test, counters)  # fold 3
        _, _, accs_f4, F1_f4 = train_GCN(tree_dict, fold4_x_train, fold4_x_val, fold4_x_test, counters)  # fold 4

        print("Test Accuracies (k-folds):", accs_f0, accs_f1, accs_f2, accs_f3, accs_f4)
        test_accs.append((accs_f0 + accs_f1 + accs_f2 + accs_f3 + accs_f4) / 5)

        write_results("Test Accuracies Iter: {}/{}, k-folds: {} {} {} {} {} -> {}".format(
            iter_counter, iterations, accs_f0, accs_f1, accs_f2, accs_f3, accs_f4,
            np.mean([accs_f0, accs_f1, accs_f2, accs_f3, accs_f4]))
        )

        # Unpack F1 scores for folds
        F1_C0_0, F1_C1_0, F1_C2_0, F1_C3_0 = F1_f0  # f1 for fold 0
        F1_C0_1, F1_C1_1, F1_C2_1, F1_C3_1 = F1_f1
        F1_C0_2, F1_C1_2, F1_C2_2, F1_C3_2 = F1_f2
        F1_C0_3, F1_C1_3, F1_C2_3, F1_C3_3 = F1_f3
        F1_C0_4, F1_C1_4, F1_C2_4, F1_C3_4 = F1_f4

        # F1 scores by classes
        TR_F1.append((F1_C0_0 + F1_C0_1 + F1_C0_2 + F1_C0_3 + F1_C0_4) / 5)
        FR_F1.append((F1_C1_0 + F1_C1_1 + F1_C1_2 + F1_C1_3 + F1_C1_4) / 5)
        UN_F1.append((F1_C2_0 + F1_C2_1 + F1_C2_2 + F1_C2_3 + F1_C2_4) / 5)
        NR_F1.append((F1_C3_0 + F1_C3_1 + F1_C3_2 + F1_C3_3 + F1_C3_4) / 5)

    sums = [sum(test_accs), sum(TR_F1), sum(FR_F1), sum(UN_F1), sum(NR_F1)]
    sums = [s / iterations for s in sums]
    print("\n")
    print("INFO:", str(info))
    print("Total Test Accuray: {:.4f} | TR_F1: {:.4f}, FR_F1: {:.4f}, UN_F1: {:.4f}, NR_F1: {:.4f}".format(*sums))
    write_results("Total Test Accuray: {:.4f} | TR_F1: {:.4f}, FR_F1: {:.4f}, UN_F1: {:.4f}, NR_F1: {:.4f}".format(*sums))



print("=========================")
print("     Rumor Detection     ")
print("=========================\n\n")

if __name__ == '__main__':
    start_time = time.time()  # Timer Start
    main()
    end_time = time.time()
    print("Elapsed Time: {0} seconds".format(round(end_time - start_time, 3)))
