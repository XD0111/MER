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

# 修改导入：使用 AutoModelForMaskedLM 和 AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Choose a GPU（模块级不依赖 args；可用环境变量 CUDA_VISIBLE_DEVICES 覆盖）
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
# -------------------------------------------------------------------------------
# Name:         model
# Description:
# Author:       梁超
# Date:         2024/5/14
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 注意：这些ID需要根据ERNIE的tokenizer重新映射
# 在初始化时动态获取这些ID，而不是硬编码
class DocumentExpert(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        # 修改：使用AutoModelForMaskedLM加载ERNIE 2.0
        self.ERNIE_MLM_for_schema = AutoModelForMaskedLM.from_pretrained(args.model_name)
        self.ERNIE_MLM_for_schema.resize_token_embeddings(args.vocab_size)
        for param in self.ERNIE_MLM_for_schema.parameters():
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

    def forward(self, document_schema, document_mask, document_loc):
        # 修改：使用ERNIE的base model而不是RoBERTa
        document_schema = self.ERNIE_MLM_for_schema.ernie(
            document_schema[0],
            attention_mask=document_mask[0],
            output_hidden_states=True
        )[0]
        document_mask = document_schema[0][document_loc]
        document_pro = self.ERNIE_MLM_for_schema.cls(document_mask)

        # 使用动态获取的答案空间ID
        document_sub = F.softmax(document_pro[:, self.answer_space_sub], dim=1)
        document_temp = F.softmax(document_pro[:, self.answer_space_temp], dim=1)
        document_cau = F.softmax(document_pro[:, self.answer_space_cau], dim=1)
        document_cof = F.softmax(document_pro[:, self.answer_space_cof], dim=1)

        return document_sub, document_cau, document_temp, document_cof

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        # 修改：ERNIE的嵌入层路径可能与RoBERTa不同
        da = self.ERNIE_MLM_for_schema.ernie.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp