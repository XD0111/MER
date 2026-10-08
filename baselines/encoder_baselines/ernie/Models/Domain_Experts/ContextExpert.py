# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         ContextExpert
# Description:
# Author:       梁超
# Date:         2025/5/30
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

# 修改导入：使用 AutoModelForMaskedLM
from transformers import AutoModelForMaskedLM

# Choose a GPU（模块级不依赖 args；可用环境变量 CUDA_VISIBLE_DEVICES 覆盖）
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
# -------------------------------------------------------------------------------
# Name:         model
# Description:
# Author:       梁超
# Date:         2024/5/14
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 删除硬编码的答案空间ID，改为动态获取

class ContextExpert(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        # 修改：使用 AutoModelForMaskedLM 加载 ERNIE 2.0
        self.ERNIE_MLM_for_schema = AutoModelForMaskedLM.from_pretrained(args.model_name)
        self.ERNIE_MLM_for_schema.resize_token_embeddings(args.vocab_size)
        for param in self.ERNIE_MLM_for_schema.parameters():
            param.requires_grad = True

        self.hidden_size = 768
        self.vocab_size = args.vocab_size
        self.mention_ratio = args.mention_weight
        self.type_ratio = args.type_weight

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

    def forward(self, mention_schema, mention_mask, mention_loc, type_schema, type_mask, type_loc):
        # Mention schema feature construction - 使用 ERNIE
        mention_schema = self.ERNIE_MLM_for_schema.ernie(
            mention_schema[0],
            attention_mask=mention_mask[0],
            output_hidden_states=True
        )[0]
        mention_mask = mention_schema[0][mention_loc]
        mention_pro = self.ERNIE_MLM_for_schema.cls(mention_mask)
        mention_sub = mention_pro[:, self.answer_space_sub]
        mention_temp = mention_pro[:, self.answer_space_temp]
        mention_cau = mention_pro[:, self.answer_space_cau]
        mention_cof = mention_pro[:, self.answer_space_cof]

        # Type schema feature construction - 使用 ERNIE
        type_schema = self.ERNIE_MLM_for_schema.ernie(
            type_schema[0],
            attention_mask=type_mask[0],
            output_hidden_states=True
        )[0]
        type_mask = type_schema[0][type_loc]
        type_pro = self.ERNIE_MLM_for_schema.cls(type_mask)
        type_sub = type_pro[:, self.answer_space_sub]
        type_temp = type_pro[:, self.answer_space_temp]
        type_cau = type_pro[:, self.answer_space_cau]
        type_cof = type_pro[:, self.answer_space_cof]

        return self.fuse_probability(mention_sub, type_sub), self.fuse_probability(mention_cau, type_cau), \
            self.fuse_probability(mention_temp, type_temp), self.fuse_probability(mention_cof, type_cof)

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        # 修改：ERNIE 的嵌入层路径
        da = self.ERNIE_MLM_for_schema.ernie.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp

    def fuse_probability(self, b, c):
        return self.mention_ratio * F.softmax(b, dim=1) + self.type_ratio * F.softmax(c, dim=1)