# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         main
# Description:
# Author:       梁超
# Date:         2024/5/16
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# This project is for Roberta model.

import os
import time

from Models.my_modules.CGE import FocalLoss
from Models.analyze_expert_routing_ratio_model import MERModel
from data.processe_data import get_dataloader
from utils import calculate_f1, makedir, calculate_macro_f1_3, calculate_accuracy

# Choose a GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(os.environ['CUDA_VISIBLE_DEVICES'])

import numpy as np
import torch
import torch.nn as nn
import tqdm
from datetime import datetime
from transformers import AdamW
from parameter import parse_args
import random
import logging


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()
args = parse_args()  # load parameters


makedir(args.log)

t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + 'analyze_expert_routing_ratio-__' + t + '.txt'

# refine
for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')

logger = logging.getLogger(__name__)


def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)


for attr in vars(args):
    printlog("{}: {}".format(attr, getattr(args, attr)))


# set seed for random number
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


setup_seed(args.seed)

to_add, tokenizer, train_dataloader, dev_dataloader, test_dataloader = get_dataloader(args)

# ---------- analyze ----------

# 专家名称映射
expert_names = {
    0: "MentionExpert",
    1: "SentenceExpert",
    2: "ContextExpert",
    3: "PathExpert",
    4: "GraphExpert"
}
# 分类任务名称
classification_tasks = ["sub", "cau", "temp", "cof"]

# 事件类型名称（根据你的数据集定义）
event_names = [
    "正时序", "负时序", "正因果", "负因果",
    "共指", "正子事件", "负子事件"
]

# Checkpoint paths are configured through parameter.py / CLI arguments.
output_file = f'out/{t}_expert_routing_ratio.csv'

# ---------- network ----------

printlog(f"args.contextual_path:{args.contextual_path}")
printlog(f"args.router_path:{args.router_path}")
printlog(f"args.MOE_path:{args.MOE_path}")
printlog(f"output_file:{output_file}")



net = MERModel(args).to(device)
net.handler(to_add, tokenizer)
best_average_f1 = 0

# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
#     {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
# optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.t_lr)

if args.loss_choice == 1:
    cross_entropy1 = nn.CrossEntropyLoss().to(device)
    cross_entropy2 = nn.CrossEntropyLoss().to(device)
    cross_entropy3 = nn.CrossEntropyLoss().to(device)
    cross_entropy4 = nn.CrossEntropyLoss().to(device)
else:
    cross_entropy1 = FocalLoss(gamma=2).to(device)
    cross_entropy2 = FocalLoss(gamma=2).to(device)
    cross_entropy3 = FocalLoss(gamma=2).to(device)
    cross_entropy4 = FocalLoss(gamma=2).to(device)

# weight_sub = [10991 / 9867, 10991 / 567, 10991 / 557]
# weight_sub = [i ** args.weight_ratio for i in weight_sub]
# weight_cau = [10991 / 8449, 10991 / 1322, 10991 / 1220]
# weight_cau = [i ** args.weight_ratio for i in weight_cau]
# weight_temp = [10991 / 8039, 10991 / 1474, 10991 / 1478]
# weight_temp = [i ** args.weight_ratio for i in weight_temp]
# weight_cof = [10991 / 9702, 10991 / 1289]
# weight_cof = [i ** args.weight_ratio for i in weight_cof]

#######################################################################################################################
###########################################        train       ########################################################
#######################################################################################################################

#######################################################################################################################
###########################################        valid       ########################################################
#######################################################################################################################

#######################################################################################################################
###########################################        test       ########################################################
#######################################################################################################################
# 创建CSV文件并写入表头
import csv
net.eval()
with open(output_file, 'w', newline='') as csvfile:
    # 构建表头
    fieldnames = ['id']
    # 添加专家选择列
    fieldnames.extend([expert_names[i] for i in range(5)])
    # 为每个专家添加四个分类任务的列
    for expert in expert_names.values():
        for task in classification_tasks:
            fieldnames.append(f"{expert.lower()}_{task}")
    # 添加pro_*列
    for task in classification_tasks:
        fieldnames.append(f"pro_{task}")
    # 新增：添加4个任务的标签列
    for task in classification_tasks:
        fieldnames.append(f"label_{task}")  # 新增标签列
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    with torch.no_grad():
        all_predictions_sub_test, all_predictions_cau_test, all_predictions_temp_test, all_predictions_cof_test = [], [], [], []
        all_labels_sub_test, all_labels_cau_test, all_labels_temp_test, all_labels_cof_test = [], [], [], []

        process_test = tqdm.tqdm(total=len(test_dataloader), ncols=75, desc='Test...')
        current_id = 1
        for batch_idx, batch in enumerate(test_dataloader, 1):

            process_test.update(1)
            events, adjacencys, question, labels = batch

            pro_sub, pro_cau, pro_temp, pro_cof, selected, common_sub, \
                common_cau, common_temp, common_cof,title_sub, title_cau, title_temp, title_cof,\
                contextual_sub, contextual_cau, contextual_temp, contextual_cof ,\
                path_sub, path_cau, path_temp, path_cof ,\
                graph_sub, graph_cau, graph_temp, graph_cof = net(events, adjacencys.to(device), question, device, args.stage, 1)
            #selected: tensor([[[0.0000]],
            #
            #         [[0.0000]],
            #
            #         [[0.0000]],
            #
            #         [[0.4839]],
            #
            #         [[0.5161]]], device='cuda:0')
            #其他 tensor([[0.3405, 0.3340, 0.3255]], device='cuda:0')
            # 将所有tensor移到CPU并转为numpy数组
            # selected的形状: [5, 1, 1] 或 [batch_size, 5, 1, 1]
            if len(selected.shape) == 3:
                selected = selected.unsqueeze(0)  # 添加batch维度
            batch_size = selected.shape[0]
            # 确保所有输出都是batch_size的列表
            outputs = [
                pro_sub, pro_cau, pro_temp, pro_cof,
                common_sub, common_cau, common_temp, common_cof,
                title_sub, title_cau, title_temp, title_cof,
                contextual_sub, contextual_cau, contextual_temp, contextual_cof,
                path_sub, path_cau, path_temp, path_cof,
                graph_sub, graph_cau, graph_temp, graph_cof
            ]
            # 处理每个样本
            for i in range(batch_size):
                row = {'id': current_id}

                # 添加专家选择
                for expert_idx in range(5):
                    expert_name = expert_names[expert_idx]
                    row[expert_name] = selected[i, expert_idx].item()

                # 添加各专家的分类预测
                expert_outputs = outputs[4:]  # 跳过pro_*
                for expert_idx in range(5):
                    expert_name = expert_names[expert_idx].lower()
                    for task_idx, task in enumerate(classification_tasks):
                        key = f"{expert_name}_{task}"
                        # 获取对应的tensor并提取值
                        output_idx = expert_idx * 4 + task_idx
                        output_tensor = expert_outputs[output_idx][i]
                        if output_tensor.numel() == 1:
                            value = output_tensor.item()
                        elif output_tensor.numel() > 1:
                            # 若为分类概率，可保存最大值索引（如预测类别）或第一个元素（根据需求调整）
                            value = output_tensor.argmax().item()  # 推荐：取预测类别
                        else:
                            value = 0.0
                        row[key] = value

                # 添加pro_*预测
                for task_idx, task in enumerate(classification_tasks):
                    key = f"pro_{task}"
                    output_tensor = outputs[task_idx][i]

                    # 处理不同形状的张量
                    if output_tensor.numel() == 1:
                        row[key] = output_tensor.item()
                    elif output_tensor.numel() > 1:
                        # 将多元素张量转换为概率列表字符串
                        row[key] = str([round(x.item(), 4) for x in output_tensor])
                    else:
                        row[key] = 0.0

                # 新增：添加4个任务的真实标签（根据原代码中labels的索引对应关系）
                row["label_sub"] = labels[0][i].item()  # sub任务的真实标签（对应all_labels_sub_test收集的labels[0]）
                row["label_cau"] = labels[2][i].item()  # cau任务的真实标签（对应all_labels_cau_test收集的labels[2]）
                row["label_temp"] = labels[1][i].item()  # temp任务的真实标签（对应all_labels_temp_test收集的labels[1]）
                row["label_cof"] = labels[3][i].item()  # cof任务的真实标签（对应all_labels_cof_test收集的labels[3]）

                # 写入CSV
                writer.writerow(row)
                current_id += 1


            pre_sub = torch.argmax(pro_sub, dim=1)
            pre_cau = torch.argmax(pro_cau, dim=1)
            pre_temp = torch.argmax(pro_temp, dim=1)
            pre_cof = torch.argmax(pro_cof, dim=1)

            all_predictions_sub_test.append(int(pre_sub))
            all_predictions_cau_test.append(int(pre_cau))
            all_predictions_temp_test.append(int(pre_temp))
            all_predictions_cof_test.append(int(pre_cof))

            all_labels_sub_test.append(labels[0])
            all_labels_cau_test.append(labels[2])
            all_labels_temp_test.append(labels[1])
            all_labels_cof_test.append(labels[3])

            if (batch_idx % args.print_frequency == 0 and batch_idx != 0) or batch_idx == len(test_dataloader):
                p_sub, r_sub, f1_sub = calculate_macro_f1_3(all_predictions_sub_test, all_labels_sub_test)
                p_cau, r_cau, f1_cau = calculate_macro_f1_3(all_predictions_cau_test, all_labels_cau_test)
                p_tem, r_tem, f1_tem = calculate_macro_f1_3(all_predictions_temp_test, all_labels_temp_test)
                p_cof, r_cof, f1_cof = calculate_f1(all_predictions_cof_test, all_labels_cof_test)

                printlog('--------------------------------------------------------------------------------------------------')
                printlog('Test Batch: {}/{}'.format(batch_idx, len(test_dataloader)))
                printlog('Sub events:')
                printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_sub, r_sub, f1_sub, calculate_accuracy(all_predictions_sub_test, all_labels_sub_test)))
                printlog('Cau events:')
                printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_cau, r_cau, f1_cau, calculate_accuracy(all_predictions_cau_test, all_labels_cau_test)))
                printlog('Tem events:')
                printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_tem, r_tem, f1_tem, calculate_accuracy(all_predictions_temp_test, all_labels_temp_test)))
                printlog('Cof events:')
                printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_cof, r_cof, f1_cof, calculate_accuracy(all_predictions_cof_test, all_labels_cof_test)))
        process_test.close()
    printlog(f"预测结果已保存到 {output_file}")