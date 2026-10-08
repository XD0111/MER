import re
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def clean_and_evaluate(input_filepath):
    # 定义四个大类及对应的子标签
    categories = {
        "共指关系": ["none", "cof"],
        "子事件关系": ["belong to", "include"],
        "时间关系": ["before", "after"],
        "因果关系": ["cause", "caused by"]
    }

    # 汇总所有合法标签
    valid_labels = set()
    for labels in categories.values():
        valid_labels.update(labels)

    y_true = []
    y_pred = []

    # 1. 逐行读取并清洗数据
    with open(input_filepath, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            # 清理可能存在的类似 这样的前端提示符
            line = re.sub(r'^\\s*', '', line)
            if not line:
                continue

            # 优先用制表符分割，如果失败尝试用多空格分割
            parts = line.split('\t')
            if len(parts) >= 2:
                prediction_raw = parts[0].strip()
                ground_truth_raw = parts[-1].strip()
            else:
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 2:
                    prediction_raw = parts[0].strip()
                    ground_truth_raw = parts[-1].strip()
                else:
                    continue

            # 去除特殊符号并转为小写比对
            clean_prediction = re.sub(r'[*"]', '', prediction_raw).strip().lower()
            clean_truth = re.sub(r'[*"]', '', ground_truth_raw).strip().lower()

            # 确保 Ground Truth 是合法标签才进行统计
            if clean_truth in valid_labels:
                # 核心逻辑：如果预测结果长篇大论或不在合法标签里，强行判为 none
                if clean_prediction not in valid_labels:
                    clean_prediction = 'none'

                y_pred.append(clean_prediction)
                y_true.append(clean_truth)

    print(f"数据清洗完毕，共提取 {len(y_true)} 条有效数据。\n")

    # 2. 计算每个子标签的精确率(P)、召回率(R)、F1
    labels_list = list(valid_labels)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_list, zero_division=0
    )

    label_metrics = {label: {'P': p[i], 'R': r[i], 'F1': f1[i], 'Support': support[i]}
                     for i, label in enumerate(labels_list)}

    # 3. 按四大类计算 Macro 平均并打印
    print("=" * 70)
    print(f"{'关系类别':<10} | {'Macro-P':<10} | {'Macro-R':<10} | {'Macro-F1':<10} | {'样本数(Support)'}")
    print("-" * 70)

    for cat_name, cat_labels in categories.items():
        # 对属于该类别的各个标签的P/R/F1求算数平均
        cat_p = np.mean([label_metrics[lbl]['P'] for lbl in cat_labels])
        cat_r = np.mean([label_metrics[lbl]['R'] for lbl in cat_labels])
        cat_f1 = np.mean([label_metrics[lbl]['F1'] for lbl in cat_labels])
        cat_support = sum([label_metrics[lbl]['Support'] for lbl in cat_labels])

        print(f"{cat_name:<12} | {cat_p:<10.4f} | {cat_r:<10.4f} | {cat_f1:<10.4f} | {cat_support}")

    print("=" * 70)


# 执行文件（填入你的文件名）
input_file = 'deepseek_v3.2_fewshot3.txt'
clean_and_evaluate(input_file)