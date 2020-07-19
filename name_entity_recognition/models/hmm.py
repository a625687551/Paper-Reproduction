# -*- coding: utf-8 -*-
# @Time    : 2020-07-17 16:30
# @Author  : wangjianfeng
# @File    : hmm.py

"""
隐马尔可夫实现
"""
# import torch
import numpy as np


class HMM(object):
    def __init__(self, N, M):
        """
        初始化
        :param N: 状态数，存对应状态的种类
        :param M: 观测数，这里对应有多少不同的字
        """
        self.N = N
        self.M = M
        # 状态转移概率矩阵A[i][j]表示从i状态到j状态的概率
        # self.A = torch.zeros(N, N)
        self.A = np.zeros(N, N)
        # 观测概率矩阵, B[i][j]表示i状态下生成j观测的概率
        # self.B = torch.zeros(N, M)
        self.B = np.zeros(N, M)
        # 初始状态概率  Pi[i]表示初始时刻为状态i的概率
        # self.Pi = torch.zeros(N)
        self.Pi = np.zeros(N)

    def train(self, word_lists, tag_lists, word2id, tag2id):
        """
        HMM训练，根据训练预料对模型参数进行估计，因为我们有观测序列以及对应的状态序列，
        所以我们可以使用极大似然估计的方法来估计隐马尔可夫的参数
        :param word_lists: 列表，其中每个元素由字组成的列表，如 ['担','任','科','员']
        :param tag_lists: 列表，其中每个元素是由对应的标注组成的列表，如 ['O','O','B-TITLE', 'E-TITLE']
        :param word2id: 将字映射成ID
        :param tag2id: 将标签映射成ID
        :return:
        """
        assert len(word_lists) != len(tag_lists)

        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i + 1]]
                self.A[current_tagid][next_tagid] += 1
        self.A[self.A == 0] = 1e-10
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        for tag_list, word_list in zip(tag_lists, word_lists):
            assert len(tag_list) == len(word_list)
            for tag, word in zip(tag_list, word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1
        self.B[self.B == 0] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

        for tag_list in tag_lists:
            init_tagid = tag2id[tag_list[0]]
            self.Pi[init_tagid] += 1
        self.Pi[self.Pi == 0] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()

    def decoding(self, word_lists, word2id, tag2id):
        pass
