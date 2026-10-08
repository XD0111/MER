# -*- coding: utf-8 -*-#
import os
import numpy as np


# -------------------------------------------------------------------------------
# Name:         utils
# Description:
# Author:       梁超
# Date:         2024/5/18
# -------------------------------------------------------------------------------
def calculate_f1(pred, label):
    """
    计算f1
    :param pred:
    :param label:
    :return:
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(pred)):
        if pred[i] == label[i] and label[i] != 0:
            tp += 1
        elif pred[i] != label[i] and label[i] == 0:
            fp += 1
        elif pred[i] != label[i] and label[i] != 0:
            fn += 1
        else:
            tn += 1

    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

def calculate_macro_f1_3(pred_list, label_list):
    # 将列表转换为numpy数组
    pred_array = np.array(pred_list)
    label_list = [label.item() for label in label_list]
    label_array = np.array(label_list)

    # 创建空的TP、FP、FN计数器
    tp_counter = [0, 0, 0]
    fp_counter = [0, 0, 0]
    fn_counter = [0, 0, 0]

    # 遍历预测和真实标签，更新计数器
    for pred, label in zip(pred_array, label_array):
        tp_counter[label] += (pred == label)
        fp_counter[pred] += (pred != label)
        fn_counter[label] += (pred != label)

    # 计算每个类别的F1分数
    p_per_class = []
    r_per_class = []
    f1_scores_per_class = []
    for class_id in [1, 2]:
        precision = tp_counter[class_id] / (tp_counter[class_id] + fp_counter[class_id] + 1e-16)
        recall = tp_counter[class_id] / (tp_counter[class_id] + fn_counter[class_id] + 1e-16)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        p_per_class.append(precision)
        r_per_class.append(recall)
        f1_scores_per_class.append(f1)


    # 计算Macro-F1分数，即所有类别的F1分数的平均值
    macro_f1 = sum(f1_scores_per_class) / len(f1_scores_per_class)
    p = sum(p_per_class) / len(p_per_class)
    r = sum(r_per_class) / len(r_per_class)

    return p, r, macro_f1

def calculate_accuracy(pred_list, label_list):
    # 将列表转换为numpy数组
    pred_array = np.array(pred_list)
    label_list = [label.item() for label in label_list]
    label_array = np.array(label_list)
    return np.mean(pred_array == label_array)

def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

