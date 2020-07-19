# -*- coding: utf-8 -*-
# @Time    : 2020-07-19 08:17
# @Author  : wangjianfeng
# @File    : bilstm.py

"""
双向lstm
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, out_size, emb_size=128, hidden_size=128):
        """
        对LSTM模型进行训练和测试
        :param vocab_size: 词典大小
        :param emb_size: 词向量维度
        :param hidden_size: 隐藏向量维度
        :param out_size:标注的种类
        """
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)

        packed = pad_packed_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.lin(rnn_out)
        return scores

    def test(self, sents_tensor, lengths, _):
        """
        第三个参数不会用到，
        :param sents_tensor:
        :param lengths:
        :param _:
        :return:
        """
        logits = self.forward(sents_tensor, lengths)
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids
