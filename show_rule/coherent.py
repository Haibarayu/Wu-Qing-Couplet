from collections import Counter
import numpy as np
import math
import pickle


class Coherent:

    def __init__(self):
        try:
            self.word2id = pickle.load(open("./show_rule/data/preprocessed-data/con_word2id.pkl", "rb"))
            self.id2word = pickle.load(open("./show_rule/data/preprocessed-data/con_id2word.pkl", "rb"))
            self.unigram = np.load("./show_rule/data/preprocessed-data/unigram.npy")
            self.bigram = np.load("./show_rule/data/preprocessed-data/bigram.npy")
        except Exception as e:
            self.construct()

    def construct(self):
        """语料"""
        f = open("show_rule/data/coherent.txt", encoding='utf-8').read()
        corpus = f.split()

        """语料预处理"""
        counter = Counter()  # 词频统计
        for sentence in corpus:
            for word in sentence:
                counter[word] += 1
        count = counter.most_common()
        lec = len(count)
        # word2id = {count[i][0]: i for i in range(lec)}
        self.word2id = {}
        for i in range(lec):
            self.word2id[count[i][0]] = i
        self.id2word = {i: w for w, i in self.word2id.items()}
        pickle.dump(self.word2id, open("./show_rule/data/preprocessed-data/con_word2id.pkl", "wb"))
        pickle.dump(self.id2word, open("./show_rule/data/preprocessed-data/con_id2word.pkl", "wb"))

        """N-gram建模训练"""
        self.unigram = np.array([i[1] for i in count]) / sum(i[1] for i in count)

        self.bigram = np.zeros((lec, lec)) + 1e-8
        for sentence in corpus:
            # sentence = [word2id[w] for w in sentence]
            sen = []
            for w in sentence:
                sen.append(self.word2id[w])
            sentence = sen
            for i in range(1, len(sentence)):
                self.bigram[[sentence[i - 1]], [sentence[i]]] += 1
        for i in range(lec):
            self.bigram[i] /= self.bigram[i].sum()

        np.save("./show_rule/data/preprocessed-data/unigram", self.unigram)
        np.save("./show_rule/data/preprocessed-data/bigram", self.bigram)

    """句子概率"""

    def prob(self, sentence):
        s = [self.word2id[w] for w in sentence]
        les = len(s)
        if les < 1:
            return 0
        p = self.unigram[s[0]]
        if les < 2:
            return p * math.pow(len(sentence), len(sentence) * 4.5)
        for i in range(1, les):
            p *= self.bigram[s[i - 1], s[i]]
        return p * math.pow(len(sentence), len(sentence) * 4.5)
