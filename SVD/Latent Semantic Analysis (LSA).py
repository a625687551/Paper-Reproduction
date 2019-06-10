# -*- coding:utf-8 -*-

"""
lsa tutorial
https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/
"""
import numpy as np

from numpy import asarray
from numpy import sum

titles = [
    "The Neatest Little Guide to Stock Market Investing",
    "Investing For Dummies, 4th Edition",
    "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
    "The Little Book of Value Investing",
    "Value Investing: From Graham to Buffett and Beyond",
    "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
    "Investing in Real Estate, 5th Edition",
    "Stock Investing For Dummies",
    "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
]
stopwords = ['and', 'edition', 'for', 'in', 'little', 'of', 'the', 'to']
ignorechars = ''',:'!'''


class LSA(object):

    def __init__(self, stopwords, ignore_chars):
        self.stopwords = stopwords
        self.ignore_chars = ignore_chars
        self.word_dict = {}
        self.dcount = 0

    def parse(self, doc):
        words = doc.split()
        for word in words:
            word = word.lower().translate(self.ignore_chars)
            if word in self.stopwords:
                continue
            elif word in self.word_dict:
                self.word_dict[word].append(self.dcount)
            else:
                self.word_dict[word] = [self.dcount]
        self.dcount += 1

    def build(self):
        self.word_keys = [k for k in self.word_dict.keys() if
                     len(self.word_dict[k]) > 1]
        self.word_keys.sort()
        self.A = np.zeros([len(self.word_keys), self.dcount])
        for i, k in enumerate(self.word_keys):
            for d in self.word_dict[k]:
                self.A[i, d] += 1

    def calc(self):
        self.U, self.S, self.Vt = np.linalg.svd(self.A)

    def tf_idf(self):
        words_per_doc = sum(self.A, axis=0)
        docs_per_word = sum(asarray(self.A > 0, "i"), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i, j] = (self.A[i, j] / words_per_doc[j]) * np.log(
                    float(cols) / docs_per_word[i])

    def print_A(self):
        print("here is the count matrix")
        print(self.A)

    def print_svd(self):
        print("here are the singular values")
        print(np.diag(self.S)[:3, :3])
        print("here are the first 3 columns of the U matrix")
        print(-1*self.U[:, 0:3])
        print("here are the first 3 rows of the Vt matrix")
        print(-1*self.Vt[0:3, :])


if __name__ == '__main__':
    mylsa = LSA(stopwords, ignorechars)
    for t in titles:
        mylsa.parse(t)
    mylsa.build()
    mylsa.print_A()
    mylsa.calc()
    mylsa.print_svd()
