# -*- coding: utf-8 -*-#

import torch
import torch.nn as nn

import os

import torch.nn.functional as F

from transformers import RobertaForMaskedLM


# Choose a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# -------------------------------------------------------------------------------
# Name:         Title_Experts
# Description:
# Author:       梁超
# Date:         2025/5/30
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
answer_space_sub = [50273, 50274, 50275]
answer_space_temp = [50268, 50269, 50270]
answer_space_cau = [50265, 50266, 50267]
answer_space_cof = [50271, 50272]


class SentenceExpert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.RoBERTa_MLM_for_schema = RobertaForMaskedLM.from_pretrained(args.model_name)
        self.RoBERTa_MLM_for_schema.resize_token_embeddings(args.vocab_size)
        for param in self.RoBERTa_MLM_for_schema.parameters():
            param.requires_grad = True

        self.hidden_size = 768
        self.vocab_size = args.vocab_size

    def forward(self, title_schema, title_mask, title_loc):
        # Mention schema feature construction
        title_schema = self.RoBERTa_MLM_for_schema.roberta(title_schema[0], attention_mask=title_mask[0], output_hidden_states=True)[0]
        title_mask = title_schema[0][title_loc]
        title_pro = self.RoBERTa_MLM_for_schema.lm_head(title_mask)
        title_sub, title_temp, title_cau, title_cof = F.softmax(title_pro[:, answer_space_sub], dim=1), F.softmax(title_pro[:, answer_space_temp], dim=1), F.softmax(title_pro[:, answer_space_cau], dim=1), F.softmax(title_pro[:, answer_space_cof], dim=1)

        return title_sub, title_cau, title_temp, title_cof

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