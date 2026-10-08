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


class MentionExpert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.RoBERTa_MLM_for_schema = RobertaForMaskedLM.from_pretrained(args.model_name)
        self.RoBERTa_MLM_for_schema.resize_token_embeddings(args.vocab_size)
        for param in self.RoBERTa_MLM_for_schema.parameters():
            param.requires_grad = True

        self.hidden_size = 768
        self.vocab_size = args.vocab_size

    def forward(self, common_schema, common_mask, common_loc):
        # Mention schema feature construction
        common_schema = self.RoBERTa_MLM_for_schema.roberta(common_schema[0], attention_mask=common_mask[0], output_hidden_states=True)[0]
        common_mask = common_schema[0][common_loc]
        common_pro = self.RoBERTa_MLM_for_schema.lm_head(common_mask)
        common_sub, common_temp, common_cau, common_cof = F.softmax(common_pro[:, answer_space_sub], dim=1), F.softmax(common_pro[:, answer_space_temp], dim=1), F.softmax(common_pro[:, answer_space_cau], dim=1), F.softmax(common_pro[:, answer_space_cof], dim=1)

        return common_sub, common_cau, common_temp, common_cof

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
