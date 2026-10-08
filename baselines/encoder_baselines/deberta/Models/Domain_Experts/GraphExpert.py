# -*- coding: utf-8 -*-#
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from transformers import DebertaForMaskedLM

from Models.my_modules.CGE import GAT

# Choose a GPU（模块级不依赖 args；可用环境变量 CUDA_VISIBLE_DEVICES 覆盖）
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
# -------------------------------------------------------------------------------
# Name:         GraphExpert
# Description:
# Author:       梁超
# Date:         2024/5/30
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphExpert(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.Deberta_MLM = DebertaForMaskedLM.from_pretrained(args.model_name)

        # 使用tokenizer获取特殊token ID（与 data/processe_data.py 中的 <su0> 等保持一致）
        self.answer_space_sub = [tokenizer.convert_tokens_to_ids(token) for token in ['<su0>', '<su1>', '<su2>']]
        self.answer_space_temp = [tokenizer.convert_tokens_to_ids(token) for token in ['<te0>', '<te1>', '<te2>']]
        self.answer_space_cau = [tokenizer.convert_tokens_to_ids(token) for token in ['<ca0>', '<ca1>', '<ca2>']]
        self.answer_space_cof = [tokenizer.convert_tokens_to_ids(token) for token in ['<co0>', '<co1>']]

        self.Deberta_MLM.resize_token_embeddings(args.vocab_size)
        for param in self.Deberta_MLM.parameters():
            param.requires_grad = True

        self.hidden_size = 768
        self.active = nn.Tanh()
        self.to_sub = nn.Linear(self.hidden_size, self.hidden_size)
        self.to_cau = nn.Linear(self.hidden_size, self.hidden_size)
        self.to_temp = nn.Linear(self.hidden_size, self.hidden_size)
        self.to_cof = nn.Linear(self.hidden_size, self.hidden_size)

        self.GAT = GAT(self.hidden_size, self.hidden_size, args.GAT_drop, args.alpha, args.num_heads)

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
        self.vocab_size = args.vocab_size

    def forward(self, idxs, masks, sent_ids, locs, adjacency, question, device):
        # Semantic feature construction
        all_sentences = self.Deberta_MLM.deberta(idxs[0], attention_mask=masks[0], output_hidden_states=True)[0].to(device)
        all_events = torch.zeros((len(locs), self.hidden_size)).to(device)

        for i in range(len(sent_ids)):
            all_events[i] = all_sentences[int(sent_ids[i])][int(locs[i])]

        adjacency = self.pro_adjacency(adjacency[0])

        all_events_sub = self.active(self.to_sub(all_events))
        all_events_cau = self.active(self.to_cau(all_events))
        all_events_temp = self.active(self.to_temp(all_events))
        all_events_cof = self.active(self.to_cof(all_events))

        # Attention
        ###################################################################################
        all_events_sub = self.GAT(all_events_sub, adjacency[0],
                                  adjacency[1] + adjacency[2] + adjacency[3]) + all_events_sub
        all_events_cau = self.GAT(all_events_cau, adjacency[2],
                                  adjacency[0] + adjacency[1] + adjacency[3]) + all_events_cau
        all_events_temp = self.GAT(all_events_temp, adjacency[1],
                                   adjacency[0] + adjacency[2] + adjacency[3]) + all_events_temp
        all_events_cof = self.GAT(all_events_cof, adjacency[3],
                                  adjacency[0] + adjacency[1] + adjacency[2]) + all_events_cof
        ###################################################################################

        sub_r = all_events_sub[question[0]] - all_events_sub[question[1]]
        cau_r = all_events_cau[question[0]] - all_events_cau[question[1]]
        temp_r = all_events_temp[question[0]] - all_events_temp[question[1]]
        cof_r = all_events_cof[question[0]] - all_events_cof[question[1]]

        # del all_events

        pro_sub = F.softmax(self.mlp_sub(sub_r), dim=1)
        pro_cau = F.softmax(self.mlp_cau(cau_r), dim=1)
        pro_temp = F.softmax(self.mlp_temp(temp_r), dim=1)
        pro_cof = F.softmax(self.mlp_cof(cof_r), dim=1)

        return pro_sub, pro_cau, pro_temp, pro_cof

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        da = self.Deberta_MLM.deberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp

    def pro_adjacency(self, adjacency):
        # Delete relation between nodes to be predicted.
        adjacency = adjacency.float().to(device)
        temp = torch.where(adjacency == -1, torch.tensor(0, dtype=torch.float32).to(device), adjacency)
        # add self loop
        for i in range(temp.shape[0]):
            temp[i, torch.arange(temp.shape[1]), torch.arange(temp.shape[1])] = 1
        return temp