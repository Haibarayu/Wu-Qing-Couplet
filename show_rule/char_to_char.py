import numpy as np
import pickle
import json


class CharToChar:
    # ——————字的词性一致—————— #
    json_dict = open('show_rule/data/dict.json', encoding='utf-8').read()
    dict = json.loads(json_dict)

    word2id = {}
    for k in dict:
        word2id[k['letter']] = k['more']
    # print(word2id)

    def part_of_speech(self, first, second):
        first_list = []
        second_list = []
        final_list = []
        for ff in first:
            first_list.append(self.word2id[ff])
        for i, nn in enumerate(second):
            second_list.append(self.word2id[nn])
            flag = 0
            for nn_word in self.word2id[nn]:  # 下联单个句的词性
                if nn_word in first_list[i]:
                    flag = 1
            final_list.append(flag)
        # print(final_list)
        return final_list


    # ——————基于Word2Vec的余弦相似度—————— #
    char_dict = pickle.load(open("show_rule/data/char_dict", "rb"))
    char_embedding = np.load("show_rule/data/char_embedding")

    def cos(self, vector_a, vector_b):
        numerator = np.dot(vector_a, vector_b.T)
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = numerator / denominator
        char_sim = 0.5 + 0.5 * cos  # 归一化
        # print(char_sim)
        return char_sim

    def char_cos(self, first, second):
        sim = []
        for idx, ch in enumerate(first):
            vector_f = self.char_embedding[self.char_dict.get(first[idx])]
            vector_s = self.char_embedding[self.char_dict.get(second[idx])]
            sim.append(self.cos(vector_f, vector_s))
        # print(sim)
        return sim


    '''字字相对'''
    def char_to_char(self, first, second):
        pos = self.part_of_speech(first, second)
        sim = self.char_cos(first, second)
        word_ctc = []
        for idx, ch in enumerate(first):
            word_ctc.append(pos[idx] * 0.4 + sim[idx] * 0.6)
        # print(word_ctc)
        sen_ctc = sum(word_ctc) / len(word_ctc)
        # print(sen_ctc)
        return sen_ctc


