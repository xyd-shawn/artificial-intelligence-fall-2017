# -*- coding: utf-8 -*-

import os
import sys
import math

import pickle
import argparse
from tqdm import tqdm



class myPinYin(object):
    def __init__(self, **kwargs):
        self.ngram = kwargs.get('ngram', 2)
        pinyin_maps_file = kwargs.get('pinyin_maps_file', '../data/拼音汉字表.txt')
        words_count_file = kwargs.get('words_count_file', '../tmp/words_count.pkl')
        gram_2_count_file = kwargs.get('gram_2_count_file', '../tmp/gram_2_count.pkl')
        gram_3_count_file = kwargs.get('gram_3_count_file', '../tmp/gram_3_count.pkl')
        with open(pinyin_maps_file, 'rb') as f:
            pinyin_maps = f.readlines()
        self.pinyin_dict = dict()
        for pinyin_map in pinyin_maps:
            pinyin_map = pinyin_map.decode('gbk').strip().split(' ')
            self.pinyin_dict[pinyin_map[0]] = pinyin_map[1:]
        print('--------------------')
        print('加载已经提取好的语料信息')
        print('--------------------')
        self.load_corpus_data(words_count_file, gram_2_count_file, gram_3_count_file)
        print('加载完毕')
        self.total_count = sum(self.words_count.values())

    def load_corpus_data(self, words_count_file, gram_2_count_file, gram_3_count_file):
        with open(words_count_file, 'rb') as f1:
            self.words_count = pickle.load(f1)
        with open(gram_2_count_file, 'rb') as f2:
            self.gram_2_count = pickle.load(f2)
        if self.ngram == 3:
            with open(gram_3_count_file, 'rb') as f3:
                self.gram_3_count = pickle.load(f3)

    def pinyins2words(self, pinyins, **kwargs):
        if self.ngram == 2:
            return self.pinyins2words_gram_2(pinyins, **kwargs)
        else:
            return self.pinyins2words_gram_3(pinyins, **kwargs)

    def preprocess(self, pinyins):
        trans_table = [self.pinyin_dict[pinyin] for pinyin in pinyins]
        for i in range(len(trans_table)):
            remove_word = []
            for j in range(len(trans_table[i])):
                if self.words_count[trans_table[i][j]] == 0:
                     remove_word.append(trans_table[i][j])
            for j in range(len(remove_word)):
                trans_table[i].remove(remove_word[j])    # 去除语料中没有出现的字
        return trans_table

    def compute_adj_weight_gram_2(self, trans_table, **kwargs):
        is_first_word_weighed = kwargs.get('is_first_word_weighed', True)
        is_forward = kwargs.get('is_forward', True)
        smooth_weight_2 = kwargs.get('smooth_weight_2', 0.64)
        adj_weights = {}
        if is_forward:
            if is_first_word_weighed:
                adj_weights[0] = [math.log(self.words_count[x] / self.total_count) for x in trans_table[0]]
            else:
                adj_weights[0] = [0] * len(trans_table[0])
            for i in range(1, len(trans_table)):
                adj_weights[i] = []
                for wj in trans_table[i - 1]:
                    temp = []
                    frac_1 = self.words_count[wj]
                    for wk in trans_table[i]:
                        ss = wj + wk
                        frac_2 = self.gram_2_count.get(ss, 0)
                        p1 = frac_2 / frac_1
                        p2 = self.words_count[wk] / self.total_count
                        temp.append(math.log(p1 * smooth_weight_2 + p2 * (1 - smooth_weight_2)))
                    adj_weights[i].append(temp)
        else:
            if is_first_word_weighed:
                adj_weights[len(trans_table) - 1] = [math.log(self.words_count[x] / self.total_count) for x in trans_table[-1]]
            else:
                adj_weights[len(trans_table) - 1] = [0] * len(trans_table[-1])
            for i in range(len(trans_table) - 2, -1, -1):
                adj_weights[i] = []
                for wj in trans_table[i + 1]:
                    temp = []
                    frac_1 = self.words_count[wj]
                    for wk in trans_table[i]:
                        ss = wk + wj
                        frac_2 = self.gram_2_count.get(ss, 0)
                        p1 = frac_2 / frac_1
                        p2 = self.words_count[wk] / self.total_count
                        temp.append(math.log(p1 * smooth_weight_2 + p2 * (1 - smooth_weight_2)))
                    adj_weights[i].append(temp)
        return adj_weights

    def compute_adj_weight_gram_3(self, trans_table, **kwargs):
        smooth_weight_2 = kwargs.get('smooth_weight_2', 0.64)
        smooth_weight_3 = kwargs.get('smooth_weight_3', [0.5, 0.3, 0.2])
        adj_weights = {}
        adj_weights[0] = [math.log(self.words_count[x] / self.total_count) for x in trans_table[0]]
        adj_weights[1] = []
        for wj in trans_table[0]:
            temp = []
            frac_1 = self.words_count[wj]
            for wk in trans_table[1]:
                ss = wj + wk
                frac_2 = self.gram_2_count.get(ss, 0)
                p1 = frac_2 / frac_1
                p2 = self.words_count[wk] / self.total_count
                temp.append(math.log(p1 * smooth_weight_2 + p2 * (1 - smooth_weight_2)))
            adj_weights[1].append(temp)
        for i in range(2, len(trans_table)):
            adj_weights[i] = []
            for wj in trans_table[i - 2]:
                temp1 = []
                for wk in trans_table[i - 1]:
                    temp2 = []
                    ss1 = wj + wk
                    count_1 = self.gram_2_count.get(ss1, 0)
                    count_2 = self.words_count[wk]
                    for wl in trans_table[i]:
                        ss2 = wk + wl
                        sss = ss + wl
                        count_3 = self.gram_2_count.get(ss2, 0)
                        if count_1 == 0 or count_3 == 0:
                            p2 = count_3 / self.words_count[wk]
                            p3 = self.words_count[wl] / self.total_count
                            temp2.append(math.log(p2 * smooth_weight_3[1] + p3 * smooth_weight_3[2]))
                        else:
                            count_4 = self.gram_3_count.get(sss, 0)
                            p1 = count_4 / count_1
                            p2 = count_3 / self.words_count[wk]
                            p3 = self.words_count[wl] / self.total_count
                            temp2.append(math.log(p1 * smooth_weight_3[0] + p2 * smooth_weight_3[1] + p3 * smooth_weight_3[2]))
                    temp1.append(temp2)
                adj_weights[i].append(temp1)
        return adj_weights

    def pinyins2words_gram_2(self, pinyins, **kwargs):
        trans_table = self.preprocess(pinyins)
        adj_weights = self.compute_adj_weight_gram_2(trans_table, **kwargs)
        weights = []
        paths = []
        is_forward = kwargs.get('is_forward', True)
        if is_forward:
            new_weights = adj_weights[0]
            new_paths = [-1] * len(trans_table[0])
            weights.append(new_weights)
            paths.append(new_paths)
            for i in range(1, len(trans_table)):
                new_weights = [0] * len(trans_table[i])
                new_paths = [-1] * len(trans_table[i])
                for j in range(len(trans_table[i])):
                    temp = [(weights[i - 1][k] + (adj_weights[i])[k][j]) for k in range(len(trans_table[i - 1]))]
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
        else:
            new_weights = adj_weights[len(trans_table) - 1]
            new_paths = [-1] * len(trans_table[-1])
            weights.append(new_weights)
            paths.append(new_paths)
            for i in range(1, len(trans_table)):
                new_weights = [0] * len(trans_table[-(i + 1)])
                new_paths = [-1] * len(trans_table[-(i + 1)])
                for j in range(len(trans_table[-(i + 1)])):
                    temp = [(weights[i - 1][k] + (adj_weights[len(trans_table) - i - 1])[k][j]) for k in range(len(trans_table[-i]))]
                    new_weights[j] = max(temp)
                    new_paths[j] = temp.index(new_weights[j])
                weights.append(new_weights)
                paths.append(new_paths)
            maxval = max(weights[-1])
            route = [weights[-1].index(maxval)]
            node = route[-1]
            words = [trans_table[0][node]]
            for i in range(len(trans_table) - 1, 0, -1):
                route.append(paths[i][node])
                node = route[-1]
                words.append(trans_table[len(trans_table) - i][node])
            words = ''.join(words)
        return words

    def pinyins2words_gram_3(self, pinyins, **kwargs):
        trans_table = self.preprocess(pinyins)
        adj_weights = self.compute_adj_weight_gram_3(trans_table, **kwargs)
        weights = []
        paths = []
        new_weights = adj_weights[0]
        new_paths = [-1] * len(trans_table[0])
        weights.append(new_weights)
        paths.append(new_paths)
        new_weights = []
        new_paths = [-1] * len(trans_table[1])
        for j in range(len(trans_table[1])):
            temp = [(weights[0][k] + (adj_weights[1])[k][j]) for k in range(len(trans_table[0]))]
            new_paths[j] = temp.index(max(temp))
            new_weights.append(temp)
        weights.append(new_weights)
        paths.append(new_paths)
        for i in range(2, len(trans_table)):
            new_weights = []
            new_paths = []
            for j in range(len(trans_table[i])):
                temp1 = [0] * len(trans_table[i - 1])
                temp2 = [-1] * len(trans_table[i - 1])
                for k in range(len(trans_table[i - 1])):
                    temp = [(weights[i - 1][k][l] + (adj_weights[i])[l][k][j]) for l in range(len(trans_table[i - 2]))]
                    temp1[k] = max(temp)
                    temp2[k] = temp.index(temp1[k])
                new_weights.append(temp1)
                new_paths.append(temp2)
            weights.append(new_weights)
            paths.append(new_paths)
        node1 = -1
        node2 = -1
        node3 = -1
        maxval = -100000000
        for j in range(len(weights[-1])):
            for k in range(len(weights[-1][j])):
                if weights[-1][j][k] > maxval:
                    maxval = weights[-1][j][k]
                    node1 = j
                    node2 = k
        words = [trans_table[-1][node1], trans_table[-2][node2]]
        for i in range(len(trans_table) - 1, 1, -1):
            node3 = paths[i][node1][node2]
            words.append(trans_table[i - 2][node3])
            node1 = node2
            node2 = node3
        words.reverse()
        words = ''.join(words)
        return words



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('-i', '--inputFile', default='../data/input.txt', help='input file')
    parser.add_argument('-o', '--outputFile', default='../data/output.txt', help='output file')
    parser.add_argument('-n', '--ngram', type=int, default=2, help='N-gram')
    parser.add_argument('-s', '--shell', action='store_true', default=False, help='use shell')
    args = parser.parse_args()
    input_file = args.inputFile
    output_file = args.outputFile
    ngram = args.ngram
    use_shell = args.shell

    print('--------------------')
    if ngram == 2:
        print('使用基于字的二元模型')
    else:
        print('使用基于字的三元模型')
    my_pinyin = myPinYin(ngram=ngram)
    print('--------------------')

    if use_shell:

        while True:
            input_pinyins = input('>>> ')
            if input_pinyins == 'exit':
                sys.exit()
            else:
                try:
                    pinyin_list = input_pinyins.strip().lower().split(' ')
                    print(my_pinyin.pinyins2words(pinyin_list))
                except:
                    print('输入拼音无法转化，请重试！')
    else:
        with open(input_file) as f:
            pinyins = f.readlines()
        out = open(output_file, 'w')
        print('开始处理输入文件')
        for ww in tqdm(pinyins):
            try:
                pinyin_list = ww.strip().split(' ')
                res = my_pinyin.pinyins2words(pinyin_list)
            except:
                res = '错误：拼音无法转化'
            out.write(res + '\n')
        out.close()
        print('处理完毕')
