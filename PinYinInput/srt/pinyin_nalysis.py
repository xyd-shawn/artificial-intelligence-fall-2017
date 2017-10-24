# -*- coding: utf-8 -*-

import os
import math

import pickle


class myPinYin(object):
    def __init__(self, **kwargs):
        pinyin_maps_file = kwargs.get('pinyin_maps_file', '../data/拼音汉字表.txt')
        words_count_file = kwargs.get('words_count_file', '../tmp/words_count.pkl')
        gram_2_count_file = kwargs.get('gram_2_count_file', '../tmp/gram_2_count.pkl')
        with open(pinyin_maps_file, 'rb') as f:
            pinyin_maps = f.readlines()
        self.pinyin_dict = dict()
        for pinyin_map in pinyin_maps:
            pinyin_map = pinyin_map.decode('gbk').strip().split(' ')
            self.pinyin_dict[pinyin_map[0]] = pinyin_map[1:]
        self.load_corpus_data(words_count_file, gram_2_count_file)
        self.total_count = sum(self.words_count.values())

    def load_corpus_data(self, words_count_file, gram_2_count_file):
        with open(words_count_file, 'rb') as f1:
            self.words_count = pickle.load(f1)
        with open(gram_2_count_file, 'rb') as f2:
            self.gram_2_count = pickle.load(f2)

    def pinyins2words(self, pinyins, **kwargs):
        first_word = kwargs.get('first_word', True)
        trans_table = [self.pinyin_dict[pinyin] for pinyin in pinyins]
        length_table = []
        for i in range(len(trans_table)):
            remove_word = []
            for j in range(len(trans_table[i])):
                if self.words_count[trans_table[i][j]] == 0:
                     remove_word.append(trans_table[i][j])
            for j in range(len(remove_word)):
                trans_table[i].remove(remove_word[j])
            length_table.append(len(trans_table[i]))
        weights = []
        paths = []
        adj_weights = {}
        if first_word:
            adj_weights[0] = [math.log(self.words_count[x]) for x in trans_table[0]]
        else:
            adj_weights[0] = [0] * len(trans_table[0])
        for i in range(1, len(trans_table)):
            adj_weights[i] = []
            for j in range(len(trans_table[i - 1])):
                temp = []
                frac_1 = self.words_count[trans_table[i - 1][j]]
                for k in range(len(trans_table[i])):
                    ss = trans_table[i - 1][j] + trans_table[i][k]
                    frac_2 = self.gram_2_count.get(ss, 0)
                    '''
                    if frac_2 == 0:
                        temp.append(-100)
                    else:
                        temp.append(math.log(frac_2 / frac_1))
                    '''
                    temp.append(math.log(((frac_2 / frac_1) + (self.words_count[ss[-1]] / self.total_count)) / 2))
                adj_weights[i].append(temp)
        new_weights = adj_weights[0]
        new_paths = [-1] * len(trans_table[0])
        weights.append(new_weights)
        paths.append(new_paths)
        for i in range(1, len(trans_table)):
            new_weights = [0] * len(trans_table[i])
            new_paths = [-1] * len(trans_table[i])
            for j in range(len(trans_table[i])):
                temp =  [(weights[i - 1][k] + (adj_weights[i])[k][j]) for k in range(len(trans_table[i - 1]))]
                new_weights[j] = max(temp)
                new_paths[j] = temp.index(new_weights[j])
            weights.append(new_weights)
            paths.append(new_paths)
        maxval = max(weights[-1])
        route = [weights[-1].index(maxval)]
        node = route[-1]
        words = [trans_table[-1][node]]
        for i in range(len(trans_table) - 1, 0, -1):
            route.append(paths[i][node])
            node = route[-1]
            words.append(trans_table[i - 1][node])
        words.reverse()
        words = ''.join(words)
        return words


if __name__ == '__main__':
    input_file = '../data/input.txt'
    output_file = '../data/output.txt'
    with open(input_file) as f:
        pinyins = f.readlines()
        print(pinyins)
    my_pinyin = myPinYin()
    for ww in pinyins:
        pinyin_list = ww.strip().split(' ')
        print(pinyin_list)
        print(my_pinyin.pinyins2words(pinyin_list))
