# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='EKGRR')
    parser.add_argument('--gpu_id', default='1', type=str, help='Choose which GPU to use')

    # Pre-trained Language Model
    parser.add_argument('--ablation', default='none', type=str,
                        choices=['none', 'common', 'title', 'contextual', 'document', 'graph'],
                        help='Choose which expert to remove for ablation study')
    parser.add_argument('--model_name', default='/home/wzl/prompt-learning/PLMs/BERT/BertForMaskedLM/bert-base-uncased', type=str)
    parser.add_argument('--vocab_size', default=30522, type=int, help='Size of RoBERTa vocab')
    parser.add_argument('--train_ratio', default=1.0, type=float)

    # Dataset
    parser.add_argument('--train_data_path', default='./data/train.json', type=str)
    parser.add_argument('--valid_data_path', default='./data/valid.json', type=str)
    parser.add_argument('--test_data_path', default='./data/test.json', type=str)
    parser.add_argument('--len_arg', default=285, type=int)
    parser.add_argument('--len_schema', default=512, type=int)
    parser.add_argument('--len_common_schema', default=75, type=int)
    parser.add_argument('--len_path_schema', default=100, type=int)
    parser.add_argument('--len_key_schema', default=200, type=int)
    parser.add_argument('--diff_type', default=0, type=int)

    # Model Setting
    parser.add_argument('--mlp_drop', default=0.4, type=float)
    parser.add_argument('--GAT_drop', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--mlp_size', default=768, type=int)
    parser.add_argument('--num_heads', default=4, type=int)

    # Train Setting
    parser.add_argument('--num_epoch', default=15, type=int)
    parser.add_argument('--num_experts', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--loss_choice', default=1, type=int)
    parser.add_argument('--weight_choice', default=1, type=int)
    parser.add_argument('--weight_ratio', default=0.85, type=float)
    parser.add_argument('--mention_weight', default=0.5, type=float)
    parser.add_argument('--type_weight', default=0.5, type=float)
    parser.add_argument('--print_frequency', default=4000, type=int)

    # 【核心修改区：学习率与 Router 控制】
    parser.add_argument('--t_lr', default=5e-6, type=float, help='底层模型的安全微调学习率')
    parser.add_argument('--router_lr', default=5e-6, type=float, help='Router专属学习率(千万不能大于1e-4)')
    parser.add_argument('--top_k', default=2, type=int, help='建议设为2，提供容错率')
    parser.add_argument('--top_p', default=1.1, type=float)
    parser.add_argument('--temperature', default=0.9, type=float)
    parser.add_argument('--smoothing_factor', default=1.5, type=float, help='平滑让权系数')
    parser.add_argument('--noise_epsilon', default=0.1, type=float, help='训练时的探索噪声强度')

    parser.add_argument('--wd', default=1e-2, type=float)
    parser.add_argument('--stage', default=6, type=int)

    parser.add_argument('--common_path', default='./checkpoint_bert_1.0/MentionExpert_model.pth', type=str)
    parser.add_argument('--title_path', default='./checkpoint_bert_1.0/SentenceExpert_model.pth', type=str)
    parser.add_argument('--contextual_path', default='./checkpoint_bert_1.0/ContextExpert_model.pth', type=str)
    parser.add_argument('--document_path', default='./checkpoint_bert_1.0/DocumentExpert_model.pth', type=str)
    parser.add_argument('--graph_path', default='./checkpoint_bert_1.0/GraphExpert_model.pth', type=str)
    parser.add_argument('--router_path', default='./checkpoint_bert_1.0/AER_model.pth', type=str)

    parser.add_argument('--seed', default=209, type=int)
    parser.add_argument('--log', default='./out/', type=str)

    args = parser.parse_args()
    return args