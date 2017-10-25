# -*- coding: utf-8 -*-

import os

import json
import pickle


class myCorpus(object):
    def __init__(self, **kwargs):
        words_file = kwargs.get('words_file', '../data/一二级汉字表.txt')
        with open(words_file, 'rb') as f:
            self.words = list(f.readline().decode('gbk').strip())
        self.words_count = dict.fromkeys(self.words, 0)
        self.gram_2_count = dict()

    def load_corpus_from_sina(self, corpus_dir):
        files = os.listdir(corpus_dir)
        files_path = [os.path.join(corpus_dir, file_name) for file_name in files]
        for file_path in files_path:
            with open(file_path) as f:
                corpus_list = f.readlines()
            for corpus_json in corpus_list:
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

    def dump_result(self, **kwargs):
        words_count_file = kwargs.get('words_count_file', '../tmp/words_count.pkl')
        gram_2_count_file = kwargs.get('gram_2_count_file', '../tmp/gram_2_count.pkl')
        with open(words_count_file, 'wb') as wcf:
            pickle.dump(self.words_count, wcf)
        with open(gram_2_count_file, 'wb') as gcf:
            pickle.dump(self.gram_2_count, gcf)
