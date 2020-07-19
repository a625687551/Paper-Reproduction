# -*- coding: utf-8 -*-
# @Time    : 2020-07-18 16:20
# @Author  : wangjianfeng
# @File    : crf.py

"""
条件随机场CRF算法实现
"""
from sklearn_crfsuite import CRF

from .util import sent2features


class CRFModel(object):
    def __init__(self, solver="lbfgs", c1=0.1, c2=0.1, max_iter=100,
                 all_possible_transitions=False):
        self.model = CRF(algorithm=solver, c1=c1, c2=c2,
                         max_iterations=max_iter,
                         all_possible_transitions=all_possible_transitions)

    def train(self, sentences, tag_lists):
        features = [sent2features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    def test(self, sentences):
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists
