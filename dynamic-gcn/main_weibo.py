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

# from preparation.preprocess_dataset import load_resource_labels
# from preparation.preprocess_dataset import load_resource_trees

from preparation.preprocess_dataset import load_resource_labels_weibo as load_resource_labels
from preparation.preprocess_dataset import load_resource_trees_weibo as load_resource_trees

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

from model_weibo import Network  # GCN (ICLR 2017) ++


def append_results(string):  # TODO:
    with open(RESULTS_FILE, 'a') as out_file:
        out_file.write(str(string) + '\n')


# -------------------------------
#         PARSE ARGUMENTS
# -------------------------------
parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--model', '-m', type=str, default='GCN', help='GCN, GraphSAGE, GIN')
parser.add_argument('--learning-sequence', '-ls', type=str, help='additive, dot_product')
parser.add_argument('--dataset-name', '-dn', type=str, help='Twitter15, Twitter16')
parser.add_argument('--dataset-type', '-dt', type=str, help='sequential, temporal')
parser.add_argument('--snapshot-num', '-sn', type=int, help='2, 3, 5, ...')
parser.add_argument('--cuda', '-c', type=str, default='cuda:2', help='cuda:3')
args = parser.parse_args()
print(args)

model = args.model
learning_sequence = args.learning_sequence
dataset_name = args.dataset_name
dataset_type = args.dataset_type
snapshot_num = args.snapshot_num
current = datetime.datetime.now().strftime("%Y_%m%d_%H%M")


# -------------------------------
#         Validate Inputs
# -------------------------------
assert model in ['GCN']
assert learning_sequence in ['additive', 'dot_product']
assert dataset_name in ['Twitter15', 'Twitter16', 'Weibo']
assert dataset_type in ['sequential', 'temporal']
assert snapshot_num in [2, 3, 5]


# --------------------------
#         INIT PATHS
# --------------------------
path_info = [model, dataset_name, dataset_type, learning_sequence, snapshot_num, current]
ensure_directory("./results/")
RESULTS_FILE = "./results/{0}_{1}_{2}_{3}_{4}_{5}_results.txt".format(*path_info)
FOLDS_FILE = "./results/{0}_{1}_{2}_{3}_{4}_{5}_folds.json".format(*path_info)
MODEL_PATH = "./results/{0}_{1}_{2}_{3}_{4}_{5}_model.pt".format(*path_info)
LABEL_PATH = './resources/{0}/{0}_label_all.txt'.format(dataset_name)
TREE_PATH = './resources/{0}/data.TD_RvNN.vol_5000.txt'.format(dataset_name)

if dataset_name == 'Weibo':
    LABEL_PATH = './resources/{0}/weibo_id_label.txt'.format(dataset_name)
    TREE_PATH = './resources/{0}/weibotree.txt'.format(dataset_name)


# -------------------------------
#         Hyperparameters
# -------------------------------
iterations = 10
num_epochs = 200
batch_size = 20
# batch_size = 10
# batch_size = 2
lr = 0.0005
weight_decay = 1e-4
patience = 10
td_droprate = 0.2
bu_droprate = 0.2
device = torch.device(args.cuda if torch.cuda.is_available() else exit())
# device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

settings = {
    "model": model, "dataset_name": dataset_name, "dataset_type": dataset_type,
    "learning_sequence": learning_sequence, "snapshot_num": snapshot_num,
    "iterations": iterations, "num_epochs": num_epochs, "batch_size": batch_size,
    "lr": lr, "weight_decay": weight_decay, "patience": patience,
    "td_droprate": td_droprate, "bu_droprate": bu_droprate,
    "current": current, "sys.argv": sys.argv, "cuda": args.cuda,
}
append_results(settings)  # Dev
counters = {'iter': 0, 'CV': 0}


# TODO:
remove_list = ['3501902090262385', '3907580407356244', '3907742282069764', '3909081075061253', '3909155720971721', '3914408365363135', '3684095995971132', '3466379833885944', '3500947630475466', '3523166905046601', '3547825524904328']

# Train: with DropEdge
def load_snapshot_dataset_train(dataset_name, tree_dict, fold_x_train):

    fold_x_train = [x for x in fold_x_train if x not in remove_list]

    data_path = "./data/graph/{0}/{1}_snapshot".format(dataset_name, dataset_type)
    train_dataset = GraphSnapshotDataset(
        tree_dict, fold_x_train, data_path=data_path, snapshot_num=snapshot_num,
        td_droprate=0, bu_droprate=0,  # weibo without dropedge - stochastic (x)
    )
    print("train count:", len(train_dataset))
    return train_dataset

# Inference: without DropEdge
def load_snapshot_dataset_val_or_test(dataset_name, tree_dict, fold_x_val_or_test):

    fold_x_val_or_test = [x for x in fold_x_val_or_test if x not in remove_list]

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
    model = Network(5000, 64, 64, settings).to(device)


    # -----------------
    #     OPTIMIZER
    # -----------------
    BU_params = []
    BU_params += list(map(id, model.rumor_GCN_0.BURumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.rumor_GCN_0.BURumorGCN.conv2.parameters()))
    base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.rumor_GCN_0.BURumorGCN.conv1.parameters(), 'lr': lr/5},
        {'params': model.rumor_GCN_0.BURumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)

    early_stopping = EarlyStopping(patience=patience, verbose=True, model_path=MODEL_PATH)


    # without dropedge
    train_dataset = load_snapshot_dataset_train(dataset_name, tree_dict, x_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

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

            with torch.cuda.device(device):
                torch.cuda.empty_cache()  # TODO: CHECK

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
            print("Iter {:02d} | CV {:02d} | Epoch {:03d} | Batch {:03d} | Train_Loss {:.4f} | Train_Accuracy {:.4f}".format(
                counters['iter'], counters['CV'], epoch, batch_index, batch_train_loss, batch_train_acc))

            del batch_data

        train_losses.append(np.mean(batch_train_losses))  # epoch
        train_accuracies.append(np.mean(batch_train_accuracies))

        # del train_dataset
        # del train_loader

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
        # append_results("eval result: " + str(eval_result))
        append_results(
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

    del train_dataset
    del train_loader

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
    F0 = test_eval_result['C0']['F1']  # F1
    F1 = test_eval_result['C1']['F1']
    F2 = test_eval_result['C2']['F1']
    F3 = test_eval_result['C3']['F1']

    counters['CV'] += 1
    losses = [train_losses, validation_losses, test_losses]
    accuracies = [train_accuracies, validation_accuracies, test_accuracies]

    append_results("losses: " + str(losses))
    append_results("accuracies: " + str(accuracies))
    append_results("val eval result: " + str(validation_eval_result))
    append_results("test eval result: " + str(test_eval_result))
    print("Test_Loss {:.4f} | Test_Accuracy {:.4f}".format(
        np.mean(test_losses), np.mean(test_accuracies))
    )

    return losses, accuracies, accs, [F0, F1, F2, F3]


def main():

    id_label_dict, label_id_dict = load_resource_labels(LABEL_PATH)
    tree_dict = load_resource_trees(TREE_PATH)

    test_accs = []
    TR_F1, FR_F1, UN_F1, NR_F1 = [], [], [], []  # F1 score

    for iter_counter in range(iterations):
        counters['iter'] = iter_counter
        folds = load_k_fold_train_val_test(label_id_dict, k=5)
        print_folds_labels(id_label_dict, folds)
        append_json_file(FOLDS_FILE, folds)

        fold0_x_train, fold0_x_val, fold0_x_test = folds[0][0], folds[1][0], folds[2][0]
        fold1_x_train, fold1_x_val, fold1_x_test = folds[0][1], folds[1][1], folds[2][1]
        fold2_x_train, fold2_x_val, fold2_x_test = folds[0][2], folds[1][2], folds[2][2]
        fold3_x_train, fold3_x_val, fold3_x_test = folds[0][3], folds[1][3], folds[2][3]
        fold4_x_train, fold4_x_val, fold4_x_test = folds[0][4], folds[1][4], folds[2][4]

        _, _, accs_f0, F1_f0 = train_GCN(tree_dict, fold0_x_train, fold0_x_val, fold0_x_test, counters)  # fold 0
        _, _, accs_f1, F1_f1 = train_GCN(tree_dict, fold1_x_train, fold1_x_val, fold1_x_test, counters)  # fold 1
        _, _, accs_f2, F1_f2 = train_GCN(tree_dict, fold2_x_train, fold2_x_val, fold2_x_test, counters)  # fold 2
        _, _, accs_f3, F1_f3 = train_GCN(tree_dict, fold3_x_train, fold3_x_val, fold3_x_test, counters)  # fold 3
        _, _, accs_f4, F1_f4 = train_GCN(tree_dict, fold4_x_train, fold4_x_val, fold4_x_test, counters)  # fold 4

        print("Test Accuracies (k-folds):", accs_f0, accs_f1, accs_f2, accs_f3, accs_f4)
        test_accs.append((accs_f0 + accs_f1 + accs_f2 + accs_f3 + accs_f4) / 5)

        append_results("Test Accuracies Iter: {}/{}, k-folds: {} {} {} {} {} -> {}".format(
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

    print("\nINFO:", str(settings))
    print("Total Test Accuray: {:.4f} | TR_F1: {:.4f}, FR_F1: {:.4f}, UN_F1: {:.4f}, NR_F1: {:.4f}".format(*sums))
    append_results("Total Test Accuray: {:.4f} | TR_F1: {:.4f}, FR_F1: {:.4f}, UN_F1: {:.4f}, NR_F1: {:.4f}".format(*sums))



print("=========================")
print("     Rumor Detection     ")
print("=========================\n\n")

if __name__ == '__main__':
    start_time = time.time()  # Timer Start
    main()
    end_time = time.time()
    print("Elapsed Time: {0} seconds".format(round(end_time - start_time, 3)))
