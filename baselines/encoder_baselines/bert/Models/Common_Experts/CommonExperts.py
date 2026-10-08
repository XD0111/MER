# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         Common_Experts
# Description:
# Author:       梁超
# Date:         2025/5/30
# -------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

# 使用 AutoModelForMaskedLM 和 AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Choose a GPU
# -------------------------------------------------------------------------------
# Name:         model
# Description:
# Author:       梁超
# Date:         2024/5/14
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 注意：这些ID需要根据BERT的tokenizer重新映射
# 在初始化时动态获取这些ID，而不是硬编码
class MentionExpert(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        # 修改：使用AutoModelForMaskedLM加载BERT模型 (如 'bert-base-uncased')
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

    def forward(self, common_schema, common_mask, common_loc):
        # 修改：使用BERT的base model
        common_schema = self.BERT_MLM_for_schema.bert(
            common_schema[0],
            attention_mask=common_mask[0],
            output_hidden_states=True
        )[0]

        # 提取 [MASK] 位置的隐藏状态并经过预测头(cls)
        common_mask = common_schema[0][common_loc]
        common_pro = self.BERT_MLM_for_schema.cls(common_mask)

        # 使用动态获取的答案空间ID
        common_sub = F.softmax(common_pro[:, self.answer_space_sub], dim=1)
        common_temp = F.softmax(common_pro[:, self.answer_space_temp], dim=1)
        common_cau = F.softmax(common_pro[:, self.answer_space_cau], dim=1)
        common_cof = F.softmax(common_pro[:, self.answer_space_cof], dim=1)

        return common_sub, common_cau, common_temp, common_cof

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        # 获取BERT的embeddings (BERT和ERNIE的路径结构一致)
        da = self.BERT_MLM_for_schema.bert.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp