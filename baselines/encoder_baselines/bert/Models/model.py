# -*- coding: utf-8 -*-#
from Models.Common_Experts.CommonExperts import MentionExpert
from Models.Common_Experts.SentenceExpert import SentenceExpert
from Models.Common_Experts.DocumentExpert import DocumentExpert
from Models.Domain_Experts.ContextExpert import ContextExpert
from Models.Domain_Experts.GraphExpert import GraphExpert
from Models.Router import AER

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MERModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.hidden_size = 768
        self.vocab_size = args.vocab_size
        self.SentenceExpert = SentenceExpert(args, tokenizer)
        self.MentionExpert = MentionExpert(args, tokenizer)
        self.ContextExpert = ContextExpert(args, tokenizer)
        self.DocumentExpert = DocumentExpert(args, tokenizer)
        self.GraphExpert = GraphExpert(args, tokenizer)
        self.AER = AER(args, input_dim=self.hidden_size, num_experts=5 * args.num_experts,
                                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        self.experts_loaded = False
        self.router_loaded = False

    def forward(self, events_info, adjacency, question, device, stage, inference):
        gate_weights = None

        if stage == 1:
            if inference:
                self.freeze_module(
                    [self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.DocumentExpert,
                     self.GraphExpert])
            else:
                self.freeze_module([self.AER]); self.unfreeze_module([self.MentionExpert])
            sub_prob, cau_prob, temp_prob, cof_prob = self.MentionExpert(events_info['common_schema'].to(device),
                                                                        events_info['common_schema_mask'].to(device),
                                                                        events_info['event_common_loc'])

        elif stage == 2:
            if inference:
                self.freeze_module(
                    [self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.DocumentExpert,
                     self.GraphExpert])
            else:
                self.freeze_module([self.AER]); self.unfreeze_module([self.SentenceExpert])
            sub_prob, cau_prob, temp_prob, cof_prob = self.SentenceExpert(events_info['core_schema'].to(device),
                                                                       events_info['core_schema_mask'].to(device),
                                                                       events_info['event_core_loc'])

        elif stage == 3:
            if inference:
                self.freeze_module(
                    [self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.DocumentExpert,
                     self.GraphExpert])
            else:
                self.freeze_module([self.AER]); self.unfreeze_module([self.ContextExpert])
            sub_prob, cau_prob, temp_prob, cof_prob = self.ContextExpert(events_info['event_schema'].to(device),
                                                                            events_info['event_schema_mask'].to(device),
                                                                            events_info['event_mention_loc'],
                                                                            events_info['type_schema'].to(device),
                                                                            events_info['type_schema_mask'].to(device),
                                                                            events_info['event_type_loc'])

        elif stage == 4:
            if inference:
                self.freeze_module(
                    [self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.DocumentExpert,
                     self.GraphExpert])
            else:
                self.freeze_module([self.AER]); self.unfreeze_module([self.DocumentExpert])
            sub_prob, cau_prob, temp_prob, cof_prob = self.DocumentExpert(events_info['document_schema'].to(device),
                                                                          events_info['document_schema_mask'].to(
                                                                              device),
                                                                          events_info['event_document_loc'])

        elif stage == 5:
            if inference:
                self.freeze_module(
                    [self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.DocumentExpert,
                     self.GraphExpert])
            else:
                self.freeze_module([self.AER]); self.unfreeze_module([self.GraphExpert])
            sub_prob, cau_prob, temp_prob, cof_prob = self.GraphExpert(events_info['idx'].to(device),
                                                                       events_info['mask'].to(device),
                                                                       events_info['sentence_ids'],
                                                                       events_info['location'], adjacency, question,
                                                                       device)

        elif stage == 6:
            if not self.experts_loaded: self.load_experts(); self.experts_loaded = True
            if inference:
                self.freeze_module(
                    [self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.DocumentExpert,
                     self.GraphExpert])
            else:
                self.unfreeze_module([self.AER]); self.freeze_module(
                    [self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.DocumentExpert, self.GraphExpert])

            ablation = getattr(self.args, 'ablation', 'none')

            common_sub, common_cau, common_temp, common_cof = self.MentionExpert(events_info['common_schema'].to(device),
                                                                                events_info['common_schema_mask'].to(
                                                                                    device),
                                                                                events_info['event_common_loc'])
            common_sub, common_cau, common_temp, common_cof = common_sub.detach(), common_cau.detach(), common_temp.detach(), common_cof.detach()
            if ablation == 'common': common_sub, common_cau, common_temp, common_cof = map(torch.zeros_like,
                                                                                           [common_sub, common_cau,
                                                                                            common_temp, common_cof])

            title_sub, title_cau, title_temp, title_cof = self.SentenceExpert(events_info['core_schema'].to(device),
                                                                           events_info['core_schema_mask'].to(device),
                                                                           events_info['event_core_loc'])
            title_sub, title_cau, title_temp, title_cof = title_sub.detach(), title_cau.detach(), title_temp.detach(), title_cof.detach()
            if ablation == 'title': title_sub, title_cau, title_temp, title_cof = map(torch.zeros_like,
                                                                                      [title_sub, title_cau, title_temp,
                                                                                       title_cof])

            contextual_sub, contextual_cau, contextual_temp, contextual_cof = self.ContextExpert(
                events_info['event_schema'].to(device), events_info['event_schema_mask'].to(device),
                events_info['event_mention_loc'], events_info['type_schema'].to(device),
                events_info['type_schema_mask'].to(device), events_info['event_type_loc'])
            contextual_sub, contextual_cau, contextual_temp, contextual_cof = contextual_sub.detach(), contextual_cau.detach(), contextual_temp.detach(), contextual_cof.detach()
            if ablation == 'contextual': contextual_sub, contextual_cau, contextual_temp, contextual_cof = map(
                torch.zeros_like, [contextual_sub, contextual_cau, contextual_temp, contextual_cof])

            document_sub, document_cau, document_temp, document_cof = self.DocumentExpert(
                events_info['document_schema'].to(device), events_info['document_schema_mask'].to(device),
                events_info['event_document_loc'])
            document_sub, document_cau, document_temp, document_cof = document_sub.detach(), document_cau.detach(), document_temp.detach(), document_cof.detach()
            if ablation == 'document': document_sub, document_cau, document_temp, document_cof = map(torch.zeros_like,
                                                                                                     [document_sub,
                                                                                                      document_cau,
                                                                                                      document_temp,
                                                                                                      document_cof])

            graph_sub, graph_cau, graph_temp, graph_cof = self.GraphExpert(events_info['idx'].to(device),
                                                                           events_info['mask'].to(device),
                                                                           events_info['sentence_ids'],
                                                                           events_info['location'], adjacency, question,
                                                                           device)
            graph_sub, graph_cau, graph_temp, graph_cof = graph_sub.detach(), graph_cau.detach(), graph_temp.detach(), graph_cof.detach()
            if ablation == 'graph': graph_sub, graph_cau, graph_temp, graph_cof = map(torch.zeros_like,
                                                                                      [graph_sub, graph_cau, graph_temp,
                                                                                       graph_cof])

            gate_weights = self.AER(events_info['key_schema'].to(device), events_info['key_schema_mask'].to(device))

            if gate_weights.dim() == 1:
                gate_weights = gate_weights.unsqueeze(0)

            all_sub = torch.stack([common_sub, title_sub, contextual_sub, document_sub, graph_sub], dim=1)
            all_cau = torch.stack([common_cau, title_cau, contextual_cau, document_cau, graph_cau], dim=1)
            all_temp = torch.stack([common_temp, title_temp, contextual_temp, document_temp, graph_temp], dim=1)
            all_cof = torch.stack([common_cof, title_cof, contextual_cof, document_cof, graph_cof], dim=1)

            sub_prob = torch.sum(gate_weights.unsqueeze(-1) * all_sub, dim=1)
            cau_prob = torch.sum(gate_weights.unsqueeze(-1) * all_cau, dim=1)
            temp_prob = torch.sum(gate_weights.unsqueeze(-1) * all_temp, dim=1)
            cof_prob = torch.sum(gate_weights.unsqueeze(-1) * all_cof, dim=1)

        else:
            if inference:
                self.freeze_module(
                    [self.AER, self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.DocumentExpert,
                     self.GraphExpert])
            else:
                self.unfreeze_module([self.AER]); self.unfreeze_module(
                    [self.SentenceExpert, self.MentionExpert, self.ContextExpert, self.DocumentExpert, self.GraphExpert])
            if not self.experts_loaded: self.load_experts(); self.experts_loaded = True

            gate_weights = self.AER(events_info['key_schema'].to(device), events_info['key_schema_mask'].to(device))
            if gate_weights.dim() == 1:
                gate_weights = gate_weights.unsqueeze(0)

            ablation = getattr(self.args, 'ablation', 'none')
            B = gate_weights.shape[0]

            sub_prob = torch.zeros((B, 3)).to(device)
            cau_prob = torch.zeros((B, 3)).to(device)
            temp_prob = torch.zeros((B, 3)).to(device)
            cof_prob = torch.zeros((B, 2)).to(device)

            if gate_weights[:, 0].sum() > 0 and ablation != 'common':
                e_sub, e_cau, e_temp, e_cof = self.MentionExpert(events_info['common_schema'].to(device),
                                                                events_info['common_schema_mask'].to(device),
                                                                events_info['event_common_loc'])
                sub_prob += gate_weights[:, 0].unsqueeze(1) * e_sub
                cau_prob += gate_weights[:, 0].unsqueeze(1) * e_cau
                temp_prob += gate_weights[:, 0].unsqueeze(1) * e_temp
                cof_prob += gate_weights[:, 0].unsqueeze(1) * e_cof

            if gate_weights[:, 1].sum() > 0 and ablation != 'title':
                e_sub, e_cau, e_temp, e_cof = self.SentenceExpert(events_info['core_schema'].to(device),
                                                               events_info['core_schema_mask'].to(device),
                                                               events_info['event_core_loc'])
                sub_prob += gate_weights[:, 1].unsqueeze(1) * e_sub
                cau_prob += gate_weights[:, 1].unsqueeze(1) * e_cau
                temp_prob += gate_weights[:, 1].unsqueeze(1) * e_temp
                cof_prob += gate_weights[:, 1].unsqueeze(1) * e_cof

            if gate_weights[:, 2].sum() > 0 and ablation != 'contextual':
                e_sub, e_cau, e_temp, e_cof = self.ContextExpert(events_info['event_schema'].to(device),
                                                                    events_info['event_schema_mask'].to(device),
                                                                    events_info['event_mention_loc'],
                                                                    events_info['type_schema'].to(device),
                                                                    events_info['type_schema_mask'].to(device),
                                                                    events_info['event_type_loc'])
                sub_prob += gate_weights[:, 2].unsqueeze(1) * e_sub
                cau_prob += gate_weights[:, 2].unsqueeze(1) * e_cau
                temp_prob += gate_weights[:, 2].unsqueeze(1) * e_temp
                cof_prob += gate_weights[:, 2].unsqueeze(1) * e_cof

            if gate_weights[:, 3].sum() > 0 and ablation != 'document':
                e_sub, e_cau, e_temp, e_cof = self.DocumentExpert(events_info['document_schema'].to(device),
                                                                  events_info['document_schema_mask'].to(device),
                                                                  events_info['event_document_loc'])
                sub_prob += gate_weights[:, 3].unsqueeze(1) * e_sub
                cau_prob += gate_weights[:, 3].unsqueeze(1) * e_cau
                temp_prob += gate_weights[:, 3].unsqueeze(1) * e_temp
                cof_prob += gate_weights[:, 3].unsqueeze(1) * e_cof

            if gate_weights[:, 4].sum() > 0 and ablation != 'graph':
                e_sub, e_cau, e_temp, e_cof = self.GraphExpert(events_info['idx'].to(device),
                                                               events_info['mask'].to(device),
                                                               events_info['sentence_ids'], events_info['location'],
                                                               adjacency, question, device)
                sub_prob += gate_weights[:, 4].unsqueeze(1) * e_sub
                cau_prob += gate_weights[:, 4].unsqueeze(1) * e_cau
                temp_prob += gate_weights[:, 4].unsqueeze(1) * e_temp
                cof_prob += gate_weights[:, 4].unsqueeze(1) * e_cof

        return sub_prob, cau_prob, temp_prob, cof_prob, gate_weights

    def handler(self, to_add, tokenizer):
        self.AER.handler(to_add, tokenizer)
        self.SentenceExpert.handler(to_add, tokenizer)
        self.MentionExpert.handler(to_add, tokenizer)
        self.ContextExpert.handler(to_add, tokenizer)
        self.DocumentExpert.handler(to_add, tokenizer)
        self.GraphExpert.handler(to_add, tokenizer)

    def freeze_module(self, single_modules):
        for single_module in single_modules:
            for param in single_module.parameters(): param.requires_grad = False
            single_module.eval()

    def unfreeze_module(self, single_modules):
        for single_module in single_modules:
            for param in single_module.parameters(): param.requires_grad = True
            single_module.train()

    def load_experts(self):
        self.SentenceExpert.load_state_dict(torch.load(self.args.title_path), strict=False)
        self.MentionExpert.load_state_dict(torch.load(self.args.common_path), strict=False)
        self.ContextExpert.load_state_dict(torch.load(self.args.contextual_path), strict=False)
        self.DocumentExpert.load_state_dict(torch.load(self.args.document_path), strict=False)
        self.GraphExpert.load_state_dict(torch.load(self.args.graph_path), strict=False)

    def load_router(self):
        self.AER.load_state_dict(torch.load(self.args.router_path), strict=False)