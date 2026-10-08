# -*- coding: utf-8 -*-#

import torch
import torch.nn as nn

import os
import torch.nn.functional as F
from transformers import RobertaForMaskedLM


# Choose a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# -------------------------------------------------------------------------------
# Name:         Path_Experts
# Description:
# Author:       梁超
# Date:         2025/5/30
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
answer_space_sub = [50273, 50274, 50275]
answer_space_temp = [50268, 50269, 50270]
answer_space_cau = [50265, 50266, 50267]
answer_space_cof = [50271, 50272]


class PathExpert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.RoBERTa_MLM_for_schema = RobertaForMaskedLM.from_pretrained(args.model_name)
        self.RoBERTa_MLM_for_schema.resize_token_embeddings(args.vocab_size)
        for param in self.RoBERTa_MLM_for_schema.parameters():
            param.requires_grad = True

        self.hidden_size = 768
        self.vocab_size = args.vocab_size

        self.mlp_sub = nn.Sequential(nn.Linear(self.hidden_size, args.mlp_size),
                                 nn.ReLU(), nn.Dropout(args.mlp_drop),
                                 nn.Linear(args.mlp_size, 3))
        self.mlp_cau = nn.Sequential(nn.Linear(self.hidden_size, args.mlp_size),
                                 nn.ReLU(), nn.Dropout(args.mlp_drop),
                                 nn.Linear(args.mlp_size, 3))
        self.mlp_temp = nn.Sequential(nn.Linear(self.hidden_size, args.mlp_size),
                                 nn.ReLU(), nn.Dropout(args.mlp_drop),
                                 nn.Linear(args.mlp_size, 3))
        self.mlp_cof = nn.Sequential(nn.Linear(self.hidden_size, args.mlp_size),
                                 nn.ReLU(), nn.Dropout(args.mlp_drop),
                                 nn.Linear(args.mlp_size, 2))


    def forward(self, path1, path1_mask, path2, path2_mask):
        loc1, loc2 = 0, 0
        for i in range(len(path1_mask[0][0])):
            if path1_mask[0][0][i] == 0:
                loc1 = i-2
                break
            if i == len(path1_mask[0][0])-1:
                loc1 = i-1

        for i in range(len(path2_mask[0][0])):
            if path2_mask[0][0][i] == 0:
                loc2 = i-2
                break
            if i == len(path2_mask[0][0])-1:
                loc2 = i-1

        path1_embd = self.RoBERTa_MLM_for_schema.roberta(path1[0], attention_mask=path1_mask[0], output_hidden_states=True)[0][0][loc1]

        path2_embd = self.RoBERTa_MLM_for_schema.roberta(path2[0], attention_mask=path2_mask[0], output_hidden_states=True)[0][0][loc2]

        embedding = path1_embd - path2_embd

        path_sub = F.softmax(self.mlp_sub(embedding).unsqueeze(dim=0), dim=1)
        path_cau = F.softmax(self.mlp_cau(embedding).unsqueeze(dim=0), dim=1)
        path_temp = F.softmax(self.mlp_temp(embedding).unsqueeze(dim=0), dim=1)
        path_cof = F.softmax(self.mlp_cof(embedding).unsqueeze(dim=0), dim=1)

        return path_sub, path_cau, path_temp, path_cof

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        da = self.RoBERTa_MLM_for_schema.roberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp