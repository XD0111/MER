# -*- coding: utf-8 -*-#

import torch
import torch.nn as nn

import os

import torch.nn.functional as F

from transformers import DebertaForMaskedLM

# Choose a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# -------------------------------------------------------------------------------
# Name:         Title_Experts
# Description:
# Author:       梁超
# Date:         2025/5/30
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SentenceExpert(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.Deberta_MLM_for_schema = DebertaForMaskedLM.from_pretrained(args.model_name)
        self.Deberta_MLM_for_schema.resize_token_embeddings(args.vocab_size)
        for param in self.Deberta_MLM_for_schema.parameters():
            param.requires_grad = True

        self.hidden_size = 768
        self.vocab_size = args.vocab_size

        # 动态获取特殊token的ID
        self.tokenizer = tokenizer
        self.answer_space_sub = [
            tokenizer.convert_tokens_to_ids('<su0>'),
            tokenizer.convert_tokens_to_ids('<su1>'),
            tokenizer.convert_tokens_to_ids('<su2>')
        ]
        self.answer_space_temp = [
            tokenizer.convert_tokens_to_ids('<te0>'),
            tokenizer.convert_tokens_to_ids('<te1>'),
            tokenizer.convert_tokens_to_ids('<te2>')
        ]
        self.answer_space_cau = [
            tokenizer.convert_tokens_to_ids('<ca0>'),
            tokenizer.convert_tokens_to_ids('<ca1>'),
            tokenizer.convert_tokens_to_ids('<ca2>')
        ]
        self.answer_space_cof = [
            tokenizer.convert_tokens_to_ids('<co0>'),
            tokenizer.convert_tokens_to_ids('<co1>')
        ]

    def forward(self, title_schema, title_mask, title_loc):
        # Mention schema feature construction
        title_schema = \
        self.Deberta_MLM_for_schema.deberta(title_schema[0], attention_mask=title_mask[0], output_hidden_states=True)[0]
        title_mask = title_schema[0][title_loc]
        title_pro = self.Deberta_MLM_for_schema.cls(title_mask)

        # 使用动态获取的answer space
        title_sub = F.softmax(title_pro[:, self.answer_space_sub], dim=1)
        title_temp = F.softmax(title_pro[:, self.answer_space_temp], dim=1)
        title_cau = F.softmax(title_pro[:, self.answer_space_cau], dim=1)
        title_cof = F.softmax(title_pro[:, self.answer_space_cof], dim=1)

        return title_sub, title_cau, title_temp, title_cof

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        da = self.Deberta_MLM_for_schema.deberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp