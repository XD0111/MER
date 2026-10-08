# -*- coding: utf-8 -*-#
from Models.Common_Experts.CommonExperts import MentionExpert
from Models.Common_Experts.SentenceExpert import SentenceExpert
from Models.Domain_Experts.PathExpert import PathExpert
from Models.Domain_Experts.ContextExpert import ContextExpert
from Models.Domain_Experts.GraphExpert import GraphExpert
from Models.Router import AER

# -------------------------------------------------------------------------------
# Name:         model
# Description:
# Author:       梁超
# Date:         2025/5/30
# -------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# Choose a GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
answer_space_sub = [50273, 50274, 50275]
answer_space_temp = [50268, 50269, 50270]
answer_space_cau = [50265, 50266, 50267]
answer_space_cof = [50271, 50272]

class MERModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = 768
        self.vocab_size = args.vocab_size
        self.SentenceExpert = SentenceExpert(args)
        self.MentionExpert = MentionExpert(args)
        self.ContextExpert = ContextExpert(args)
        self.PathExpert = PathExpert(args)
        self.GraphExpert = GraphExpert(args)
        self.AER = AER(args, input_dim=self.hidden_size, num_experts= 5 * args.num_experts, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        self.experts_loaded = False   #true不会把原始的专家导进来，false会导入专家# True表示Experts已加载
        self.router_loaded = False    # True表示Router已加载

    def forward(self, events_info, adjacency, question, device, stage, inference):
        # Stage 1
        global selected
        if stage == 1:
            if inference:
                self.freeze_module([self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.PathExpert, self.GraphExpert])
            else:
                self.freeze_module([self.AER])
                self.unfreeze_module([self.MentionExpert])
            sub_prob, cau_prob, temp_prob, cof_prob = self.MentionExpert(events_info['common_schema'].to(device), events_info['common_schema_mask'].to(device), events_info['event_common_loc'])
        elif stage == 2:
            if inference:
                self.freeze_module([self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.PathExpert, self.GraphExpert])
            else:
                self.freeze_module([self.AER])
                self.unfreeze_module([self.SentenceExpert])
            sub_prob, cau_prob, temp_prob, cof_prob = self.SentenceExpert(events_info['core_schema'].to(device), events_info['core_schema_mask'].to(device), events_info['event_core_loc'])
        elif stage == 3:
            if inference:
                self.freeze_module([self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.PathExpert, self.GraphExpert])
            else:
                self.freeze_module([self.AER])
                self.unfreeze_module([self.ContextExpert])
            sub_prob, cau_prob, temp_prob, cof_prob = self.ContextExpert(events_info['event_schema'].to(device), events_info['event_schema_mask'].to(device), events_info['event_mention_loc'], events_info['type_schema'].to(device), events_info['type_schema_mask'].to(device), events_info['event_type_loc'])
        elif stage == 4:
            if inference:
                self.freeze_module([self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.PathExpert, self.GraphExpert])
            else:
                self.freeze_module([self.AER])
                self.unfreeze_module([self.PathExpert])
            sub_prob, cau_prob, temp_prob, cof_prob = self.PathExpert(events_info['path1_schema'].to(device), events_info['path1_schema_mask'].to(device), events_info['path2_schema'].to(device), events_info['path2_schema_mask'].to(device))
        elif stage == 5:
            if inference:
                self.freeze_module([self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.PathExpert, self.GraphExpert])
            else:
                self.freeze_module([self.AER])
                self.unfreeze_module([self.GraphExpert])
            sub_prob, cau_prob, temp_prob, cof_prob = self.GraphExpert(events_info['idx'].to(device), events_info['mask'].to(device), events_info['sentence_ids'], events_info['location'], adjacency, question, device)

        # Stage 2
        elif stage == 6:
            self.load_experts()
            self.load_router()
            self.freeze_module([self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.PathExpert, self.GraphExpert])
            common_sub, common_cau, common_temp, common_cof = self.MentionExpert(events_info['common_schema'].to(device), events_info['common_schema_mask'].to(device), events_info['event_common_loc'])
            common_sub = common_sub.detach()  # 对每个张量单独detach
            common_cau = common_cau.detach()
            common_temp = common_temp.detach()
            common_cof = common_cof.detach()
            title_sub, title_cau, title_temp, title_cof = self.SentenceExpert(events_info['core_schema'].to(device), events_info['core_schema_mask'].to(device), events_info['event_core_loc'])
            title_sub = title_sub.detach()
            title_cau = title_cau.detach()
            title_temp = title_temp.detach()
            title_cof = title_cof.detach()
            contextual_sub, contextual_cau, contextual_temp, contextual_cof = self.ContextExpert(events_info['event_schema'].to(device), events_info['event_schema_mask'].to(device), events_info['event_mention_loc'], events_info['type_schema'].to(device), events_info['type_schema_mask'].to(device), events_info['event_type_loc'])
            contextual_sub = contextual_sub.detach()
            contextual_cau = contextual_cau.detach()
            contextual_temp = contextual_temp.detach()
            contextual_cof = contextual_cof.detach()
            path_sub, path_cau, path_temp, path_cof = self.PathExpert(events_info['path1_schema'].to(device), events_info['path1_schema_mask'].to(device), events_info['path2_schema'].to(device), events_info['path2_schema_mask'].to(device))
            path_sub = path_sub.detach()
            path_cau = path_cau.detach()
            path_temp = path_temp.detach()
            path_cof = path_cof.detach()
            graph_sub, graph_cau, graph_temp, graph_cof = self.GraphExpert(events_info['idx'].to(device), events_info['mask'].to(device), events_info['sentence_ids'], events_info['location'], adjacency, question, device)
            graph_sub = graph_sub.detach()
            graph_cau = graph_cau.detach()
            graph_temp = graph_temp.detach()
            graph_cof = graph_cof.detach()
            selected = self.AER(events_info['key_schema'].to(device), events_info['key_schema_mask'].to(device)).unsqueeze(1).unsqueeze(2)

            all_sub = torch.stack([common_sub, title_sub, contextual_sub, path_sub, graph_sub])
            all_cau = torch.stack([common_cau, title_cau, contextual_cau, path_cau, graph_cau])
            all_temp = torch.stack([common_temp, title_temp, contextual_temp, path_temp, graph_temp])
            all_cof = torch.stack([common_cof, title_cof, contextual_cof, path_cof, graph_cof])

            sub_prob = torch.sum(selected * all_sub, dim=0)
            cau_prob = torch.sum(selected * all_cau, dim=0)
            temp_prob = torch.sum(selected * all_temp, dim=0)
            cof_prob = torch.sum(selected * all_cof, dim=0)


        # Stage 3
        else:
            self.load_MOE()
            self.freeze_module([self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.PathExpert, self.GraphExpert])
            selected = self.AER(events_info['key_schema'].to(device), events_info['key_schema_mask'].to(device)).unsqueeze(1).unsqueeze(2)

            all_sub = torch.zeros((5, 1, 3)).to(device)
            all_cau = torch.zeros((5, 1, 3)).to(device)
            all_temp = torch.zeros((5, 1, 3)).to(device)
            all_cof = torch.zeros((5, 1, 2)).to(device)
            if selected[0]:
                common_sub, common_cau, common_temp, common_cof = self.MentionExpert(events_info['common_schema'].to(device), events_info['common_schema_mask'].to(device), events_info['event_common_loc'])
                all_sub[0] += common_sub
                all_cau[0] += common_cau
                all_temp[0] += common_temp
                all_cof[0] += common_cof

            if selected[1]:
                title_sub, title_cau, title_temp, title_cof = self.SentenceExpert(events_info['core_schema'].to(device), events_info['core_schema_mask'].to(device), events_info['event_core_loc'])
                all_sub[1] += title_sub
                all_cau[1] += title_cau
                all_temp[1] += title_temp
                all_cof[1] += title_cof

            if selected[2]:
                contextual_sub, contextual_cau, contextual_temp, contextual_cof = self.ContextExpert(events_info['event_schema'].to(device), events_info['event_schema_mask'].to(device), events_info['event_mention_loc'], events_info['type_schema'].to(device), events_info['type_schema_mask'].to(device), events_info['event_type_loc'])
                all_sub[2] += contextual_sub
                all_cau[2] += contextual_cau
                all_temp[2] += contextual_temp
                all_cof[2] += contextual_cof
            if selected[3]:
                path_sub, path_cau, path_temp, path_cof = self.PathExpert(events_info['path1_schema'].to(device), events_info['path1_schema_mask'].to(device), events_info['path2_schema'].to(device), events_info['path2_schema_mask'].to(device))
                all_sub[3] += path_sub
                all_cau[3] += path_cau
                all_temp[3] += path_temp
                all_cof[3] += path_cof
            if selected[4]:
                graph_sub, graph_cau, graph_temp, graph_cof = self.GraphExpert(events_info['idx'].to(device), events_info['mask'].to(device), events_info['sentence_ids'], events_info['location'], adjacency, question, device)
                all_sub[4] += graph_sub
                all_cau[4] += graph_cau
                all_temp[4] += graph_temp
                all_cof[4] += graph_cof

            sub_prob = torch.sum(selected * all_sub, dim=0)
            cau_prob = torch.sum(selected * all_cau, dim=0)
            temp_prob = torch.sum(selected * all_temp, dim=0)
            cof_prob = torch.sum(selected * all_cof, dim=0)
        return sub_prob, cau_prob, temp_prob, cof_prob, selected,\
            common_sub, common_cau, common_temp, common_cof,\
            title_sub, title_cau, title_temp, title_cof,\
            contextual_sub, contextual_cau, contextual_temp, contextual_cof ,\
            path_sub, path_cau, path_temp, path_cof ,\
            graph_sub, graph_cau, graph_temp, graph_cof

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        self.AER.handler(to_add, tokenizer)
        self.SentenceExpert.handler(to_add, tokenizer)
        self.MentionExpert.handler(to_add, tokenizer)
        self.ContextExpert.handler(to_add, tokenizer)
        self.PathExpert.handler(to_add, tokenizer)
        self.GraphExpert.handler(to_add, tokenizer)
        return

    def freeze_module(self, single_modules):
        for single_module in single_modules:
            for param in single_module.parameters():
                param.requires_grad = False
            single_module.eval()

    def unfreeze_module(self, single_modules):
        for single_module in single_modules:
            for param in single_module.parameters():
                param.requires_grad = True
            single_module.train()

    def load_experts(self):
        self.SentenceExpert.load_state_dict(torch.load(self.args.title_path), strict=False)
        self.MentionExpert.load_state_dict(torch.load(self.args.common_path), strict=False)
        self.ContextExpert.load_state_dict(torch.load(self.args.contextual_path), strict=False)
        self.PathExpert.load_state_dict(torch.load(self.args.path_path), strict=False)
        self.GraphExpert.load_state_dict(torch.load(self.args.graph_path), strict=False)

    def load_router(self):
        self.AER.load_state_dict(torch.load(self.args.router_path), strict=False)

    def load_MOE(self):
        """加载完整MoE模型权重（包括所有Experts和Router）"""
        self.load_state_dict(torch.load(self.args.MOE_path), strict=False)
        self.moe_loaded = True