# -*- coding: utf-8 -*-

import os
import sys

import json
import pickle
from tqdm import tqdm


class myCorpus(object):
    def __init__(self, **kwargs):
        words_file = kwargs.get('words_file', '../data/一二级汉字表.txt')
        with open(words_file, 'rb') as f:
            self.words = list(f.readline().decode('gbk').strip())
        self.words_count = dict.fromkeys(self.words, 0)
        self.gram_2_count = dict()
        self.gram_3_count = dict()

    def load_corpus_from_sina(self, corpus_file):
        with open(corpus_file) as f:
            corpus_list = f.readlines()
        for corpus_json in tqdm(corpus_list):
            corpus_dict = json.loads(corpus_json.strip())
            self.corpus_extract(corpus_dict['html'])

    def corpus_extract(self, corpus):
        for i in range(len(corpus)):
            if corpus[i] in self.words:
                self.words_count[corpus[i]] += 1
                if ((i + 1) < len(corpus)) and (corpus[i + 1] in self.words):
                    gram_2 = corpus[i:(i + 2)]
                    if gram_2 in self.gram_2_count.keys():
                        self.gram_2_count[gram_2] += 1
                    else:
                        self.gram_2_count[gram_2] = 1
                    if ((i + 2) < len(corpus)) and (corpus[i + 2] in self.words):
                        gram_3 = corpus[i:(i + 3)]
                        if gram_3 in self.gram_3_count.keys():
                            self.gram_3_count[gram_3] += 1
                        else:
                            self.gram_3_count[gram_3] = 1

    def dump_result(self, words_count_file, gram_2_count_file, gram_3_count_file):
        with open(words_count_file, 'wb') as wcf:
            pickle.dump(self.words_count, wcf)
        with open(gram_2_count_file, 'wb') as g2cf:
            pickle.dump(self.gram_2_count, g2cf)
        with open(gram_3_count_file, 'wb') as g3cf:
            pickle.dump(self.gram_3_count, g3cf)

if __name__ == '__main__':
    corpus_file = sys.argv[1]
    num = sys.argv[2]
    words_count_file = '../tmp/words_count_' + num + '.pkl'
    gram_2_count_file = '../tmp/gram_2_count_' + num + '.pkl'
    gram_3_count_file = '../tmp/gram_3_count_' + num + '.pkl'
    corpus_test = myCorpus()
    corpus_test.load_corpus_from_sina(corpus_file)
    corpus_test.dump_result(words_count_file, gram_2_count_file, gram_3_count_file)
