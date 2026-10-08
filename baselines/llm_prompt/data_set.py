# -*- coding: utf-8 -*-#
import torch


# -------------------------------------------------------------------------------
# Name:         dataset
# Description:
# Author:       梁超
# Date:         2024/5/16
# -------------------------------------------------------------------------------
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item
