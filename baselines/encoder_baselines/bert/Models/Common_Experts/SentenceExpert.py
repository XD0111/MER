# -*- coding: utf-8 -*-#

import torch
import torch.nn as nn

import os

import torch.nn.functional as F

# 修改导入：使用 AutoModelForMaskedLM
from transformers import AutoModelForMaskedLM

# -------------------------------------------------------------------------------
# Name:         Title_Experts
# Description:
# Author:       梁超
# Date:         2025/5/30
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 删除硬编码的答案空间ID，改为动态获取

class SentenceExpert(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        # 修改：使用 AutoModelForMaskedLM 加载 BERT 模型
        self.BERT_MLM_for_schema = AutoModelForMaskedLM.from_pretrained(args.model_name)
        self.BERT_MLM_for_schema.resize_token_embeddings(args.vocab_size)
        for param in self.BERT_MLM_for_schema.parameters():
            param.requires_grad = True

        self.hidden_size = 768
        self.vocab_size = args.vocab_size

        # 动态获取答案空间的token ID
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
        # 修改：使用 BERT 的 base model
        title_schema = self.BERT_MLM_for_schema.bert(
            title_schema[0],
            attention_mask=title_mask[0],
            output_hidden_states=True
        )[0]
        title_mask = title_schema[0][title_loc]
        title_pro = self.BERT_MLM_for_schema.cls(title_mask)

        # 使用动态获取的答案空间ID
        title_sub = F.softmax(title_pro[:, self.answer_space_sub], dim=1)
        title_temp = F.softmax(title_pro[:, self.answer_space_temp], dim=1)
        title_cau = F.softmax(title_pro[:, self.answer_space_cau], dim=1)
        title_cof = F.softmax(title_pro[:, self.answer_space_cof], dim=1)

        return title_sub, title_cau, title_temp, title_cof

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        # 修改：获取 BERT 的嵌入层路径
        da = self.BERT_MLM_for_schema.bert.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp