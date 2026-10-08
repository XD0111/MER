# -*- coding: utf-8 -*-#
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModelForMaskedLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AER(nn.Module):
    def __init__(self, args, input_dim, num_experts=5, temperature=1.0, top_k=3, top_p=0.9):
        super(AER, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_size = 768
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # 接收防坍塌参数
        self.smoothing_factor = getattr(args, 'smoothing_factor', 1.5)
        self.noise_epsilon = getattr(args, 'noise_epsilon', 0.2)

        # 统一使用 AutoModel 接口加载
        self.key = AutoModelForMaskedLM.from_pretrained(args.model_name)
        self.key.resize_token_embeddings(args.vocab_size)
        for param in self.key.parameters():
            param.requires_grad = True

        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, inputs, inputs_mask):
        # 【神级修改】使用 .base_model，完美兼容 BERT, DeBERTa, RoBERTa
        key = self.key.base_model(
            inputs[0],
            attention_mask=inputs_mask[0],
            output_hidden_states=True
        )[0][0][0]

        logits = self.router(key)

        # 训练时加入探索噪声
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_epsilon

        logits /= self.temperature
        prob_row = F.softmax(logits, dim=-1)

        selected_masks = torch.zeros_like(prob_row)
        top_k_indices = torch.topk(prob_row, min(self.top_k, self.num_experts), dim=-1).indices
        top_k_probs = prob_row[top_k_indices]

        sorted_probs, idx_in_topk = torch.sort(top_k_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = cumulative_probs <= self.top_p

        if not nucleus_mask.any():
            if nucleus_mask.dim() > 0:
                nucleus_mask[0] = True
            else:
                nucleus_mask = torch.tensor(True).to(device)

        selected_in_topk = idx_in_topk[nucleus_mask]
        selected_indices = top_k_indices[selected_in_topk]
        selected_masks[selected_indices] = 1

        # 【核心平滑逻辑】将未选中专家置为负无穷，完美分配权重
        masked_logits = torch.where(selected_masks == 1, logits, torch.tensor(float('-inf')).to(device))
        result = F.softmax(masked_logits / self.smoothing_factor, dim=-1)

        return result

    def handler(self, to_add, tokenizer):
        # 使用 .base_model 同样兼容词表扩充
        da = self.key.base_model.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)
                da[tokenizer.convert_tokens_to_ids(i)] = temp