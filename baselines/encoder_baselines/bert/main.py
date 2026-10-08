# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         main
# Description:  Updated for Expert Usage Statistics & Complete Metrics (P, R, F1, Acc)
# Author:       梁超
# Date:         2024/5/16 (Updated 2026)
# -------------------------------------------------------------------------------

import os
import time
import torch
import torch.nn as nn
from parameter import parse_args
from collections import Counter

args = parse_args()

# ================= 1. 动态路径与环境设置 =================
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

import numpy as np
import tqdm
from datetime import datetime
from transformers import AdamW
import random
import logging
from Models.my_modules.CGE import FocalLoss
from Models.model import MERModel
from data.processe_data import get_dataloader
from utils import calculate_f1, makedir, calculate_macro_f1_3, calculate_accuracy

if __name__ == '__main__':
    checkpoint_dir = f'./checkpoint_bert_{args.train_ratio}'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created dynamic checkpoint directory: {checkpoint_dir}")
    else:
        print(f"Using existing checkpoint directory: {checkpoint_dir}")

    makedir(args.log)
    t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
    args.log = f"{args.log}_ratio{args.train_ratio}__{t}.txt"

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


    def print_expert_stats(expert_counts, total_samples, mode="Train"):
        if sum(expert_counts.values()) == 0:
            return

        printlog(f"--- {mode} Expert Usage Statistics ---")
        expert_names = {0: "Common", 1: "Title", 2: "Contextual", 3: "Document", 4: "Graph"}

        for exp_id in range(5):
            count = expert_counts.get(exp_id, 0)
            ratio = (count / total_samples * 100) if total_samples > 0 else 0
            printlog(f"  Expert {exp_id} ({expert_names.get(exp_id, 'Unknown')}): {count} samples ({ratio:.2f}%)")
        printlog("---------------------------------------")


    printlog(f"Current Training Ratio: {args.train_ratio}")
    printlog(f"Checkpoints will be saved to: {checkpoint_dir}")
    for attr in vars(args):
        printlog("{}: {}".format(attr, getattr(args, attr)))


    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


    setup_seed(args.seed)

    # ================= 2. 数据加载与模型初始化 =================
    printlog("Start loading data...")
    to_add, tokenizer, train_dataloader, dev_dataloader, test_dataloader = get_dataloader(args)
    printlog(f"Data loaded. Train size: {len(train_dataloader.dataset)}")

    net = MERModel(args, tokenizer).to(device)
    net.handler(to_add, tokenizer)

    best_average_f1 = 0

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.wd},
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

    weight_sub = [i ** args.weight_ratio for i in [10991 / 9867, 10991 / 567, 10991 / 557]]
    weight_cau = [i ** args.weight_ratio for i in [10991 / 8449, 10991 / 1322, 10991 / 1220]]
    weight_temp = [i ** args.weight_ratio for i in [10991 / 8039, 10991 / 1474, 10991 / 1478]]
    weight_cof = [i ** args.weight_ratio for i in [10991 / 9702, 10991 / 1289]]

    # ================= 3. 训练循环 =================
    for epoch in range(args.num_epoch):
        loss_all_sub = 0
        loss_all_cau = 0
        loss_all_temp = 0
        loss_all_cof = 0
        printlog('Epoch: {}'.format(epoch))
        net.train()

        train_expert_counts = Counter()
        total_train_samples = 0

        all_predictions_sub, all_predictions_cau, all_predictions_temp, all_predictions_cof = [], [], [], []
        all_labels_sub, all_labels_cau, all_labels_temp, all_labels_cof = [], [], [], []

        process_train = tqdm.tqdm(total=len(train_dataloader), ncols=75, desc=f'Training (Ratio {args.train_ratio})...')

        for batch_idx, batch in enumerate(train_dataloader, 1):
            process_train.update(1)
            events, adjacencys, question, labels = batch

            current_batch_size = len(labels[0])
            total_train_samples += current_batch_size

            pro_sub, pro_cau, pro_temp, pro_cof, gate_weights = net(events, adjacencys.to(device), question, device,
                                                                    args.stage, 0)

            if gate_weights is not None:
                active_indices = torch.nonzero(gate_weights)
                if active_indices.numel() > 0:
                    if gate_weights.dim() == 1:
                        expert_ids = active_indices[:, 0].cpu().tolist()
                        for eid in expert_ids:
                            train_expert_counts[eid] += current_batch_size
                    else:
                        expert_ids = active_indices[:, 1].cpu().tolist()
                        train_expert_counts.update(expert_ids)

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

            target_labels = [torch.tensor(i).to(device) for i in labels]
            l_sub = cross_entropy1(pro_sub, target_labels[0])
            l_cau = cross_entropy2(pro_cau, target_labels[2])
            l_temp = cross_entropy3(pro_temp, target_labels[1])
            l_cof = cross_entropy4(pro_cof, target_labels[3])

            if args.weight_choice == 1:
                loss = weight_sub[all_labels_sub[-1]] * l_sub + \
                       weight_cau[all_labels_cau[-1]] * l_cau + \
                       weight_temp[all_labels_temp[-1]] * l_temp + \
                       weight_cof[all_labels_cof[-1]] * l_cof
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

                printlog(
                    '--------------------------------------------------------------------------------------------------')
                printlog('Epoch: {}/{} Batch: {}/{} Loss: {:.4f}'.format(epoch + 1, args.num_epoch, batch_idx,
                                                                         len(train_dataloader),
                                                                         (
                                                                                     loss_all_sub + loss_all_cau + loss_all_temp + loss_all_cof) / batch_idx))
                printlog('Loss_sub: {:.4f} Loss_cau: {:.4f} Loss_temp: {:.4f} Loss_cof: {:.4f}'.format(
                    loss_all_sub / batch_idx, loss_all_cau / batch_idx, loss_all_temp / batch_idx,
                    loss_all_cof / batch_idx))

                # 【修改点】修正了训练集的 Precision 和 Recall 打印顺序，使其与变量匹配
                printlog('Sub events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                    p_sub, r_sub, f1_sub, calculate_accuracy(all_predictions_sub, all_labels_sub)))
                printlog('Cau events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                    p_cau, r_cau, f1_cau, calculate_accuracy(all_predictions_cau, all_labels_cau)))
                printlog('Tem events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                    p_tem, r_tem, f1_tem, calculate_accuracy(all_predictions_temp, all_labels_temp)))
                printlog('Cof events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                    p_cof, r_cof, f1_cof, calculate_accuracy(all_predictions_cof, all_labels_cof)))

        process_train.close()
        print_expert_stats(train_expert_counts, total_train_samples, mode=f"Train Epoch {epoch}")

        # ================= 4. 验证循环 =================
        net.eval()
        dev_expert_counts = Counter()
        total_dev_samples = 0

        with torch.no_grad():
            all_predictions_sub_dev, all_predictions_cau_dev, all_predictions_temp_dev, all_predictions_cof_dev = [], [], [], []
            all_labels_sub_dev, all_labels_cau_dev, all_labels_temp_dev, all_labels_cof_dev = [], [], [], []
            process_dev = tqdm.tqdm(total=len(dev_dataloader), ncols=75, desc='Valid...')

            for batch_idx, batch in enumerate(dev_dataloader, 1):
                process_dev.update(1)
                events, adjacencys, question, labels = batch

                current_batch_size = len(labels[0])
                total_dev_samples += current_batch_size

                pro_sub, pro_cau, pro_temp, pro_cof, gate_weights = net(events, adjacencys.to(device), question, device,
                                                                        args.stage, 1)

                if gate_weights is not None:
                    active_indices = torch.nonzero(gate_weights)
                    if active_indices.numel() > 0:
                        if gate_weights.dim() == 1:
                            expert_ids = active_indices[:, 0].cpu().tolist()
                            for eid in expert_ids:
                                dev_expert_counts[eid] += current_batch_size
                        else:
                            expert_ids = active_indices[:, 1].cpu().tolist()
                            dev_expert_counts.update(expert_ids)

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

            process_dev.close()
            print_expert_stats(dev_expert_counts, total_dev_samples, mode="Valid")

            # 【修改点】展开 Valid 指标的打印
            p_sub, r_sub, f1_sub = calculate_macro_f1_3(all_predictions_sub_dev, all_labels_sub_dev)
            p_cau, r_cau, f1_cau = calculate_macro_f1_3(all_predictions_cau_dev, all_labels_cau_dev)
            p_tem, r_tem, f1_tem = calculate_macro_f1_3(all_predictions_temp_dev, all_labels_temp_dev)
            p_cof, r_cof, f1_cof = calculate_f1(all_predictions_cof_dev, all_labels_cof_dev)

            printlog(
                '--------------------------------------------------------------------------------------------------')
            printlog(f'Valid Results (Ratio {args.train_ratio}):')
            printlog('Sub events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                p_sub, r_sub, f1_sub, calculate_accuracy(all_predictions_sub_dev, all_labels_sub_dev)))
            printlog('Cau events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                p_cau, r_cau, f1_cau, calculate_accuracy(all_predictions_cau_dev, all_labels_cau_dev)))
            printlog('Tem events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                p_tem, r_tem, f1_tem, calculate_accuracy(all_predictions_temp_dev, all_labels_temp_dev)))
            printlog('Cof events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                p_cof, r_cof, f1_cof, calculate_accuracy(all_predictions_cof_dev, all_labels_cof_dev)))
            printlog(
                '--------------------------------------------------------------------------------------------------')

            average_f1_valid = (f1_sub + f1_cau + f1_tem + f1_cof) / 4

        # ================= 5. 测试循环 =================
        test_expert_counts = Counter()
        total_test_samples = 0

        with torch.no_grad():
            all_predictions_sub_test, all_predictions_cau_test, all_predictions_temp_test, all_predictions_cof_test = [], [], [], []
            all_labels_sub_test, all_labels_cau_test, all_labels_temp_test, all_labels_cof_test = [], [], [], []
            process_test = tqdm.tqdm(total=len(test_dataloader), ncols=75, desc='Test...')

            for batch_idx, batch in enumerate(test_dataloader, 1):
                process_test.update(1)
                events, adjacencys, question, labels = batch

                current_batch_size = len(labels[0])
                total_test_samples += current_batch_size

                pro_sub, pro_cau, pro_temp, pro_cof, gate_weights = net(events, adjacencys.to(device), question, device,
                                                                        args.stage, 1)

                if gate_weights is not None:
                    active_indices = torch.nonzero(gate_weights)
                    if active_indices.numel() > 0:
                        if gate_weights.dim() == 1:
                            expert_ids = active_indices[:, 0].cpu().tolist()
                            for eid in expert_ids:
                                test_expert_counts[eid] += current_batch_size
                        else:
                            expert_ids = active_indices[:, 1].cpu().tolist()
                            test_expert_counts.update(expert_ids)

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

            process_test.close()
            print_expert_stats(test_expert_counts, total_test_samples, mode="Test")

            # 【修改点】展开 Test 指标的打印
            p_sub_t, r_sub_t, f1_sub_t = calculate_macro_f1_3(all_predictions_sub_test, all_labels_sub_test)
            p_cau_t, r_cau_t, f1_cau_t = calculate_macro_f1_3(all_predictions_cau_test, all_labels_cau_test)
            p_tem_t, r_tem_t, f1_tem_t = calculate_macro_f1_3(all_predictions_temp_test, all_labels_temp_test)
            p_cof_t, r_cof_t, f1_cof_t = calculate_f1(all_predictions_cof_test, all_labels_cof_test)

            printlog(
                '--------------------------------------------------------------------------------------------------')
            printlog(f'Test Results (Ratio {args.train_ratio}):')
            printlog('Sub events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                p_sub_t, r_sub_t, f1_sub_t, calculate_accuracy(all_predictions_sub_test, all_labels_sub_test)))
            printlog('Cau events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                p_cau_t, r_cau_t, f1_cau_t, calculate_accuracy(all_predictions_cau_test, all_labels_cau_test)))
            printlog('Tem events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                p_tem_t, r_tem_t, f1_tem_t, calculate_accuracy(all_predictions_temp_test, all_labels_temp_test)))
            printlog('Cof events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                p_cof_t, r_cof_t, f1_cof_t, calculate_accuracy(all_predictions_cof_test, all_labels_cof_test)))
            printlog(
                '--------------------------------------------------------------------------------------------------')

        # ================= 6. 保存逻辑 =================
        if average_f1_valid >= best_average_f1:
            best_average_f1 = average_f1_valid
            printlog(f'Best performance updated! Saving models to: {checkpoint_dir} ...')

            save_path = checkpoint_dir

            if args.stage == 1:
                torch.save(net.MentionExpert.state_dict(), f'{save_path}/MentionExpert_model.pth')
                printlog(f"Saved MentionExpert.")
            elif args.stage == 2:
                torch.save(net.SentenceExpert.state_dict(), f'{save_path}/SentenceExpert_model.pth')
                printlog(f"Saved SentenceExpert.")
            elif args.stage == 3:
                torch.save(net.ContextExpert.state_dict(), f'{save_path}/ContextExpert_model.pth')
                printlog(f"Saved ContextExpert.")
            elif args.stage == 4:
                torch.save(net.DocumentExpert.state_dict(), f'{save_path}/DocumentExpert_model.pth')
                printlog(f"Saved DocumentExpert.")
            elif args.stage == 5:
                torch.save(net.GraphExpert.state_dict(), f'{save_path}/GraphExpert_model.pth')
                printlog(f"Saved GraphExpert.")
            elif args.stage == 6:
                torch.save(net.AER.state_dict(), f'{save_path}/AER_model.pth')
                printlog(f"Saved Router.")
            else:
                torch.save(net.state_dict(), f'{save_path}/MoE_model.pth')
                printlog(f"Saved MoE Model.")


# ================= 9. 终极诊断：测试所有 5 个专家的单体性能 =================
def test_all_single_experts(net, test_dataloader, device, args):
    printlog("\n" + "=" * 98)
    printlog("🔍 开始独立诊断: 依次测试 5 个专家在测试集上的真实单体性能 (Bypass Router)...")

    # 获取正确的模型并强制加载所有专家的 checkpoint
    model_to_use = net.module if isinstance(net, nn.DataParallel) else net
    model_to_use.load_experts()
    model_to_use.eval()

    expert_map = {
        1: "Common Expert",
        2: "Title Expert",
        3: "Contextual Expert",
        4: "Document Expert",
        5: "Graph Expert"
    }

    for stage_idx, expert_name in expert_map.items():
        printlog(f"\n---> 正在评估: {expert_name} (Stage {stage_idx}) <---")
        all_predictions_sub, all_predictions_cau, all_predictions_temp, all_predictions_cof = [], [], [], []
        all_labels_sub, all_labels_cau, all_labels_temp, all_labels_cof = [], [], [], []

        with torch.no_grad():
            process_test = tqdm.tqdm(total=len(test_dataloader), ncols=75, desc=f'Testing {expert_name}...')
            for batch in test_dataloader:
                process_test.update(1)
                events, adjacencys, question, labels = batch

                events = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in events.items()}
                adjacencys = adjacencys.to(device)

                # 动态调用对应的 stage (1 到 5)，绕过 Router
                outputs = model_to_use(events, adjacencys, question, device, stage_idx, inference=True)

                # 安全解包（兼容 4 个或 5 个返回值）
                if len(outputs) == 5:
                    pro_sub, pro_cau, pro_temp, pro_cof, _ = outputs
                else:
                    pro_sub, pro_cau, pro_temp, pro_cof = outputs[:4]

                all_predictions_sub.extend(torch.argmax(pro_sub, dim=1).cpu().tolist())
                all_predictions_cau.extend(torch.argmax(pro_cau, dim=1).cpu().tolist())
                all_predictions_temp.extend(torch.argmax(pro_temp, dim=1).cpu().tolist())
                all_predictions_cof.extend(torch.argmax(pro_cof, dim=1).cpu().tolist())

                all_labels_sub.extend(labels[0])
                all_labels_cau.extend(labels[2])
                all_labels_temp.extend(labels[1])
                all_labels_cof.extend(labels[3])

            process_test.close()

        # 计算该专家的各项指标
        p_sub, r_sub, f1_sub = calculate_macro_f1_3(all_predictions_sub, all_labels_sub)
        p_cau, r_cau, f1_cau = calculate_macro_f1_3(all_predictions_cau, all_labels_cau)
        p_tem, r_tem, f1_tem = calculate_macro_f1_3(all_predictions_temp, all_labels_temp)
        p_cof, r_cof, f1_cof = calculate_f1(all_predictions_cof, all_labels_cof)

        printlog(f'>>> {expert_name} 测试集结果 <<<')
        printlog('Sub events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(p_sub, r_sub, f1_sub,
                                                                                                calculate_accuracy(
                                                                                                    all_predictions_sub,
                                                                                                    all_labels_sub)))
        printlog('Cau events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(p_cau, r_cau, f1_cau,
                                                                                                calculate_accuracy(
                                                                                                    all_predictions_cau,
                                                                                                    all_labels_cau)))
        printlog('Tem events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(p_tem, r_tem, f1_tem,
                                                                                                calculate_accuracy(
                                                                                                    all_predictions_temp,
                                                                                                    all_labels_temp)))
        printlog('Cof events -> Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(p_cof, r_cof, f1_cof,
                                                                                                calculate_accuracy(
                                                                                                    all_predictions_cof,
                                                                                                    all_labels_cof)))

    printlog("=" * 98 + "\n")


# ==================== 执行开关 ====================
# 注释掉你不需要跑的行，只保留下面这一行，并确保启动时 --num_epoch 设为 0
# 整模测试需要全部 5 个专家的 checkpoint；训练 stage 1-5 时可设 MER_SKIP_FINAL_TEST=1 跳过
if os.getenv('MER_SKIP_FINAL_TEST') != '1':
    test_all_single_experts(net, test_dataloader, device, args)