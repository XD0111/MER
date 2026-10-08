# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         Router
# Description:
# Author:       梁超
# Date:         2025/5/31
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import DebertaForMaskedLM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class AER(nn.Module):
    def __init__(self, args, input_dim, num_experts=5, temperature=1.0, top_k=3, top_p=0.9):
        """
        Args:
            input_dim (int): 输入维度
            num_experts (int): 专家数量
            temperature (float): softmax 温度
            top_k (int or None): 先选出概率最高的 k 个专家
            top_p (float or None): 再从这 k 个中选出累计概率 >= p 的专家
        """
        super(AER, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_size = 768
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.key = DebertaForMaskedLM.from_pretrained(args.model_name)
        self.key.resize_token_embeddings(args.vocab_size)
        for param in self.key.parameters():
            param.requires_grad = True
        # 定义路由网络
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, inputs, inputs_mask):
        """
        Args:
            inputs: [batch_size, input_dim]

        Returns:
            probs: [batch_size, num_experts]，门控概率
            selected_masks: [batch_size, num_experts]，选中的专家 mask
        """

        key = self.key.deberta(inputs[0], attention_mask=inputs_mask[0], output_hidden_states=True)[0][0][0]

        logits = self.router(key)  # [B, E]
        logits /= self.temperature

        # 计算 softmax 概率
        prob_row = F.softmax(logits, dim=-1)  # [B, E]

        # 初始化 selected_masks
        selected_masks = torch.zeros_like(prob_row)

        # 当前样本的概率分布

        # Step 1: Top-k 筛选
        top_k_indices = torch.topk(prob_row, min(self.top_k, self.num_experts), dim=-1).indices
        top_k_probs = prob_row[top_k_indices]

        # Step 2: 在 top_k 中应用 Top-p 核选择
        sorted_probs, idx_in_topk = torch.sort(top_k_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        nucleus_mask = cumulative_probs <= self.top_p
        if not nucleus_mask.any():
            nucleus_mask[0] = True  # 至少保留一个专家

        # 获取原始索引
        selected_in_topk = idx_in_topk[nucleus_mask]
        selected_indices = top_k_indices[selected_in_topk]

        # 更新 mask
        selected_masks[selected_indices] = 1

        x = prob_row * selected_masks
        non_zero_values = x[x != 0]
        if len(non_zero_values) > 0:
            softmax_values = F.softmax(non_zero_values, dim=0)
        else:
            softmax_values = torch.tensor([])
        # 构造结果
        result = torch.zeros_like(x)
        result[x != 0] = softmax_values
        return result

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        da = self.key.deberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp