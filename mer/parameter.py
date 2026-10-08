# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='EKGRR')

    # Pre-trained Language Model
    parser.add_argument('--model_name', default='roberta-base', type=str, help='Pre-trained Language Model used to be encoder')
    parser.add_argument('--vocab_size', default=50265, type=int, help='Size of RoBERTa vocab')

    # Dataset
    parser.add_argument('--train_data_path', default='./data/train.json', type=str, help='Train dataset path')
    parser.add_argument('--valid_data_path', default='./data/valid.json', type=str, help='Valid dataset path')
    parser.add_argument('--test_data_path', default='./data/test.json', type=str, help='Test dataset path')
    parser.add_argument('--len_arg', default=285, type=int, help='Sentence length')
    parser.add_argument('--len_schema', default=512, type=int, help='Schema length')
    parser.add_argument('--len_common_schema', default=75, type=int, help='Common Schema length')
    parser.add_argument('--len_path_schema', default=100, type=int, help='Common Schema length')
    parser.add_argument('--len_key_schema', default=200, type=int, help='Key Schema length')
    parser.add_argument('--diff_type', default=0, type=int, help='1: different type schema; 0: same type schema')

    # Model Setting
    parser.add_argument('--mlp_drop', default=0.4, type=float, help='MLP dropout rate')
    parser.add_argument('--GAT_drop', type=float, default=0.4, help='GAT Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu in GAT.')
    parser.add_argument('--mlp_size', default=768, type=int, help='Hidden size of mlp.')
    parser.add_argument('--num_heads', default=4, type=int, help='The number of attention head')

    # Train Setting
    parser.add_argument('--num_epoch', default=15, type=int, help='Number of total epochs to run prompt learning')
    parser.add_argument('--num_experts', default=1, type=int, help='Number of every single expert')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for prompt learning')
    parser.add_argument('--loss_choice', default=1, type=int, help='Choice of loss. 1: Cross entropy. 2: Focal loss')
    parser.add_argument('--weight_choice', default=1, type=int, help='Choice of weight. 1: weighted. 2: none')
    parser.add_argument('--weight_ratio', default=0.85, type=float, help='Weight ratio for weighted loss.')
    parser.add_argument('--mention_weight', default=0.5, type=float, help='Weight ratio for Mention schem.')
    parser.add_argument('--type_weight', default=0.5, type=float, help='Weight ratio for Type schema.')
    parser.add_argument('--print_frequency', default=4000, type=int, help='Print every 2000 samples')
    parser.add_argument('--t_lr', default=5e-3, type=float, help='Initial lr')
    parser.add_argument('--top_k', default=2, type=int, help='Initial top_k')
    parser.add_argument('--top_p', default=1.1, type=float, help='Initial top_p')
    parser.add_argument('--temperature', default=0.9, type=float, help='Initial temperature')
    parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
    parser.add_argument('--stage', default=6, type=int, help='Stage choice')

    parser.add_argument('--common_path', default='./checkpoint/MentionExpert_model.pth', type=str, help='Path')
    parser.add_argument('--title_path', default='./checkpoint/SentenceExpert_model.pth', type=str, help='Path')
    parser.add_argument('--contextual_path', default='./checkpoint/ContextExpert_model.pth', type=str, help='Path')
    parser.add_argument('--path_path', default='./checkpoint/PathExpert_model.pth', type=str, help='Path')
    parser.add_argument('--graph_path', default='./checkpoint/GraphExpert_model.pth', type=str, help='Path')
    parser.add_argument('--router_path', default='./checkpoint/AER_model.pth', type=str, help='Path')

    # Others
    parser.add_argument('--seed', default=209, type=int, help='Seed for reproducibility')
    parser.add_argument('--log', default='./out/', type=str, help='Log result file name')
    args = parser.parse_args()
    return args
