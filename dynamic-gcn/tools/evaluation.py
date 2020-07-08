# Calculate accuracy, precision, recall, and F1 score
# Author: Jiho Choi
#
# Reference:
#   https://github.com/majingCUHK/Rumor_RvNN/blob/master/model/evaluate.py

from operator import add

def evaluation(prediction, y):  # 4-class
    TP = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    FP = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    FN = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    TN = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    for i in range(len(y)):
        pred, real = prediction[i], y[i]
        for label in range(4):
            if label == pred and label == real:
                TP[label] += 1
            if label == pred and label != real:
                FP[label] += 1
            if label != pred and label == real:
                FN[label] += 1
            if label != pred and label != real:
                TN[label] += 1

    acc_all = round(sum(TP.values()) / float(len(y)), 4)
    accuracy = {0: None, 1: None, 2: None, 3: None}
    precision = {0: None, 1: None, 2: None, 3: None}
    recall = {0: None, 1: None, 2: None, 3: None}
    F1 = {0: None, 1: None, 2: None, 3: None}

    for l in range(4):
        label_all = (TP[l] + FP[l] + FN[l] + TN[l])
        accuracy[l] = round((TP[l] + TN[l]) / label_all, 4)

        if (TP[l] + FP[l]) == 0:
            precision[l] = 0
        else:
            precision[l] = round(TP[l] / (TP[l] + FP[l]), 4)

        relavant = (TP[l] + FN[l])
        recall[l] = round(TP[l] / relavant, 4) if relavant else 0

        PR = (precision[l] + recall[l])
        F1[l] = round(2 * precision[l] * recall[l] / PR, 4) if PR else 0

    results = {
        'acc_all': acc_all,
        'C0': [accuracy[0], precision[0], recall[0], F1[0]],
        'C1': [accuracy[1], precision[1], recall[1], F1[1]],
        'C2': [accuracy[2], precision[2], recall[2], F1[2]],
        'C3': [accuracy[3], precision[3], recall[3], F1[3]],
    }
    return results


def merge_batch_eval_list(batch_eval_results):
    eval_results = {}
    batch_num = len(batch_eval_results)

    # Initialize
    for key in batch_eval_results[0].keys():
        if key not in eval_results:
            if not isinstance(batch_eval_results[0][key], list):
                eval_results[key] = 0.0
            else:
                eval_results[key] = [0.0, 0.0, 0.0, 0.0]
    # Combine
    for batch_eval in batch_eval_results:
        for key in batch_eval.keys():
            if not isinstance(eval_results[key], list):
                eval_results[key] += batch_eval[key]  # ACC
            else:
                eval_results[key] = list(map(add, eval_results[key], batch_eval[key]))
    # Normalize
    for key in eval_results.keys():
        value = eval_results[key]
        if not isinstance(eval_results[key], list):
            eval_results[key] = round(value / batch_num, 4) # ACC
        else:
            eval_results[key] = [round(v / batch_num, 4) for v in value]

    return eval_results
