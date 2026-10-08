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
from Models.model import MERModel
from data.processe_data import get_dataloader
from utils import calculate_f1, makedir, calculate_macro_f1_3, calculate_accuracy

# Choose a GPU（可用环境变量 CUDA_VISIBLE_DEVICES 覆盖，默认沿用原实验的 1 号卡）
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
# device_train = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device_test = torch.device('cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else 'cpu')
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
torch.cuda.empty_cache()
args = parse_args()  # load parameters


makedir(args.log)

t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + '__' + t + '.txt'

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

# ---------- network ----------

net = MERModel(args, tokenizer).to(device)
net.handler(to_add, tokenizer)
best_average_f1 = 0

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.t_lr)

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

weight_sub = [10991 / 9867, 10991 / 567, 10991 / 557]
weight_sub = [i ** args.weight_ratio for i in weight_sub]
weight_cau = [10991 / 8449, 10991 / 1322, 10991 / 1220]
weight_cau = [i ** args.weight_ratio for i in weight_cau]
weight_temp = [10991 / 8039, 10991 / 1474, 10991 / 1478]
weight_temp = [i ** args.weight_ratio for i in weight_temp]
weight_cof = [10991 / 9702, 10991 / 1289]
weight_cof = [i ** args.weight_ratio for i in weight_cof]
#######################################################################################################################
###########################################        train       ########################################################
#######################################################################################################################
for epoch in range(args.num_epoch):
    loss_all_sub = 0
    loss_all_cau = 0
    loss_all_temp = 0
    loss_all_cof = 0
    printlog('Epoch: {}'.format(epoch))
    net.train()
    all_predictions_sub, all_predictions_cau, all_predictions_temp, all_predictions_cof = [], [], [], []
    all_labels_sub, all_labels_cau, all_labels_temp, all_labels_cof = [], [], [], []
    process_train = tqdm.tqdm(total=len(train_dataloader), ncols=75, desc='Training...')
    start = 0
    for batch_idx, batch in enumerate(train_dataloader, 1):
        process_train.update(1)
        events, adjacencys, question, labels = batch
        pro_sub, pro_cau, pro_temp, pro_cof = net(events, adjacencys.to(device), question, device, args.stage, 0)

        pre_sub = torch.argmax(pro_sub, dim=1)
        pre_cau = torch.argmax(pro_cau, dim=1)
        pre_temp = torch.argmax(pro_temp, dim=1)
        pre_cof = torch.argmax(pro_cof, dim=1)

        all_predictions_sub.append(int(pre_sub))
        all_predictions_cau.append(int(pre_cau))
        all_predictions_temp.append(int(pre_temp))
        all_predictions_cof.append(int(pre_cof))

        all_labels_sub.append(labels[0])
        all_labels_cau.append(labels[2])
        all_labels_temp.append(labels[1])
        all_labels_cof.append(labels[3])

        labels = [torch.tensor(i).to(device) for i in labels]
        l_sub = cross_entropy1(pro_sub, labels[0])
        l_cau = cross_entropy2(pro_cau, labels[2])
        l_temp = cross_entropy3(pro_temp, labels[1])
        l_cof = cross_entropy4(pro_cof, labels[3])

        if args.weight_choice == 1:
            loss = weight_sub[all_labels_sub[-1]] * l_sub + weight_cau[all_labels_cau[-1]] * l_cau + weight_temp[all_labels_temp[-1]] * l_temp + weight_cof[all_labels_cof[-1]] * l_cof
        else:
            loss = l_sub + l_cau + l_temp + l_cof

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all_sub += l_sub.item()
        loss_all_cau += l_cau.item()
        loss_all_temp += l_temp.item()
        loss_all_cof += l_cof.item()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (batch_idx % args.print_frequency == 0 and batch_idx != 0) or batch_idx == len(train_dataloader):
            p_sub, r_sub, f1_sub = calculate_macro_f1_3(all_predictions_sub, all_labels_sub)
            p_cau, r_cau, f1_cau = calculate_macro_f1_3(all_predictions_cau, all_labels_cau)
            p_tem, r_tem, f1_tem = calculate_macro_f1_3(all_predictions_temp, all_labels_temp)
            p_cof, r_cof, f1_cof = calculate_f1(all_predictions_cof, all_labels_cof)

            printlog('--------------------------------------------------------------------------------------------------')
            printlog('Epoch: {}/{} Batch: {}/{} Loss: {:.4f}'.format(epoch + 1, args.num_epoch, batch_idx, len(train_dataloader), (loss_all_sub + loss_all_cau + loss_all_temp + loss_all_cof) / batch_idx))
            printlog('Loss_sub: {:.4f} Loss_cau: {:.4f} Loss_temp: {:.4f} Loss_cof: {:.4f}'.format(loss_all_sub / batch_idx, loss_all_cau / batch_idx, loss_all_temp / batch_idx, loss_all_cof / batch_idx))
            printlog('Sub events:')
            printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_sub, r_sub, f1_sub, calculate_accuracy(all_predictions_sub, all_labels_sub)))
            printlog('Cau events:')
            printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_cau, r_cau, f1_cau, calculate_accuracy(all_predictions_cau, all_labels_cau)))
            printlog('Tem events:')
            printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_tem, r_tem, f1_tem, calculate_accuracy(all_predictions_temp, all_labels_temp)))
            printlog('Cof events:')
            printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_cof, r_cof, f1_cof, calculate_accuracy(all_predictions_cof, all_labels_cof)))
    process_train.close()

#######################################################################################################################
###########################################        valid       ########################################################
#######################################################################################################################
    net.eval()
    with torch.no_grad():
        all_predictions_sub_dev, all_predictions_cau_dev, all_predictions_temp_dev, all_predictions_cof_dev = [], [], [], []
        all_labels_sub_dev, all_labels_cau_dev, all_labels_temp_dev, all_labels_cof_dev = [], [], [], []
        process_dev = tqdm.tqdm(total=len(dev_dataloader), ncols=75, desc='Valid...')
        for batch_idx, batch in enumerate(dev_dataloader, 1):
            process_dev.update(1)
            events, adjacencys, question, labels = batch

            pro_sub, pro_cau, pro_temp, pro_cof = net(events, adjacencys.to(device), question, device, args.stage, 1)

            pre_sub = torch.argmax(pro_sub, dim=1)
            pre_cau = torch.argmax(pro_cau, dim=1)
            pre_temp = torch.argmax(pro_temp, dim=1)
            pre_cof = torch.argmax(pro_cof, dim=1)

            all_predictions_sub_dev.append(int(pre_sub))
            all_predictions_cau_dev.append(int(pre_cau))
            all_predictions_temp_dev.append(int(pre_temp))
            all_predictions_cof_dev.append(int(pre_cof))

            all_labels_sub_dev.append(labels[0])
            all_labels_cau_dev.append(labels[2])
            all_labels_temp_dev.append(labels[1])
            all_labels_cof_dev.append(labels[3])

            if (batch_idx % args.print_frequency == 0 and batch_idx != 0) or batch_idx == len(dev_dataloader):
                p_sub, r_sub, f1_sub = calculate_macro_f1_3(all_predictions_sub_dev, all_labels_sub_dev)
                p_cau, r_cau, f1_cau = calculate_macro_f1_3(all_predictions_cau_dev, all_labels_cau_dev)
                p_tem, r_tem, f1_tem = calculate_macro_f1_3(all_predictions_temp_dev, all_labels_temp_dev)
                p_cof, r_cof, f1_cof = calculate_f1(all_predictions_cof_dev, all_labels_cof_dev)
                printlog('--------------------------------------------------------------------------------------------------')
                printlog('Valid Batch: {}/{}'.format(batch_idx, len(dev_dataloader)))
                printlog('Sub events:')
                printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_sub, r_sub, f1_sub, calculate_accuracy(all_predictions_sub_dev, all_labels_sub_dev)))
                printlog('Cau events:')
                printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_cau, r_cau, f1_cau, calculate_accuracy(all_predictions_cau_dev, all_labels_cau_dev)))
                printlog('Tem events:')
                printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_tem, r_tem, f1_tem, calculate_accuracy(all_predictions_temp_dev, all_labels_temp_dev)))
                printlog('Cof events:')
                printlog('Recall: {:.4f} Precision: {:.4f} F1: {:.4f} Acc：{:.4f}'.format(p_cof, r_cof, f1_cof, calculate_accuracy(all_predictions_cof_dev, all_labels_cof_dev)))
        process_dev.close()
        del pro_sub, pro_cau, pro_temp, pro_cof

        average_f1_valid = (f1_sub + f1_cau + f1_tem + f1_cof) / 4
#######################################################################################################################
###########################################        test       ########################################################
#######################################################################################################################

    with torch.no_grad():
        all_predictions_sub_test, all_predictions_cau_test, all_predictions_temp_test, all_predictions_cof_test = [], [], [], []
        all_labels_sub_test, all_labels_cau_test, all_labels_temp_test, all_labels_cof_test = [], [], [], []
        process_test = tqdm.tqdm(total=len(test_dataloader), ncols=75, desc='Test...')
        for batch_idx, batch in enumerate(test_dataloader, 1):

            process_test.update(1)
            events, adjacencys, question, labels = batch

            pro_sub, pro_cau, pro_temp, pro_cof = net(events, adjacencys.to(device), question, device, args.stage, 1)

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
    if average_f1_valid >= best_average_f1:
        best_average_f1 = average_f1_valid
        printlog('Best test performance are as the newest one!')
        if args.stage == 1:
            torch.save(net.MentionExpert.state_dict(), './checkpoint_ernie/MentionExpert_model.pth')
        elif args.stage == 2:
            torch.save(net.SentenceExpert.state_dict(), './checkpoint_ernie/SentenceExpert_model.pth')
        elif args.stage == 3:
            torch.save(net.ContextExpert.state_dict(), './checkpoint_ernie/ContextExpert_model.pth')
        elif args.stage == 4:
            torch.save(net.DocumentExpert.state_dict(), './checkpoint_ernie/DocumentExpert_model.pth')
        elif args.stage == 5:
            torch.save(net.GraphExpert.state_dict(), './checkpoint_ernie/GraphExpert_model.pth')
        elif args.stage == 6:
            torch.save(net.AER.state_dict(), './checkpoint_ernie/AER_model_1.pth')
        else:
            torch.save(net.state_dict(), './checkpoint_ernie/MoE_model.pth')
