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

from transformers import RobertaForMaskedLM


# Choose a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# -------------------------------------------------------------------------------
# Name:         model
# Description:
# Author:       梁超
# Date:         2024/5/14
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
answer_space_sub = [50273, 50274, 50275]
answer_space_temp = [50268, 50269, 50270]
answer_space_cau = [50265, 50266, 50267]
answer_space_cof = [50271, 50272]


class ContextExpert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.RoBERTa_MLM_for_schema = RobertaForMaskedLM.from_pretrained(args.model_name)
        self.RoBERTa_MLM_for_schema.resize_token_embeddings(args.vocab_size)
        for param in self.RoBERTa_MLM_for_schema.parameters():
            param.requires_grad = True

        self.hidden_size = 768
        self.vocab_size = args.vocab_size
        self.mention_ratio = args.mention_weight
        self.type_ratio = args.type_weight

    def forward(self, mention_schema, mention_mask, mention_loc, type_schema, type_mask, type_loc):
        # Mention schema feature construction
        mention_schema = self.RoBERTa_MLM_for_schema.roberta(mention_schema[0], attention_mask=mention_mask[0], output_hidden_states=True)[0]
        mention_mask = mention_schema[0][mention_loc]
        mention_pro = self.RoBERTa_MLM_for_schema.lm_head(mention_mask)
        mention_sub, mention_temp, mention_cau, mention_cof = mention_pro[:, answer_space_sub], mention_pro[:, answer_space_temp], mention_pro[:, answer_space_cau], mention_pro[:, answer_space_cof]

        # Type schema feature construction
        type_schema = self.RoBERTa_MLM_for_schema.roberta(type_schema[0], attention_mask=type_mask[0], output_hidden_states=True)[0]
        type_mask = type_schema[0][type_loc]
        type_pro = self.RoBERTa_MLM_for_schema.lm_head(type_mask)
        type_sub, type_temp, type_cau, type_cof = type_pro[:, answer_space_sub], type_pro[:, answer_space_temp], type_pro[:, answer_space_cau], type_pro[:, answer_space_cof]

        return self.fuse_probability(mention_sub, type_sub), self.fuse_probability(mention_cau, type_cau), \
               self.fuse_probability(mention_temp, type_temp), self.fuse_probability(mention_cof, type_cof)

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

    def fuse_probability(self, b, c):
        return self.mention_ratio * F.softmax(b, dim=1) + self.type_ratio * F.softmax(c, dim=1)


