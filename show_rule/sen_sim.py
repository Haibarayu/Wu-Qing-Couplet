#encoding=utf-8
import numpy as np
import pickle


class SenSim:
    # word_dict = pickle.load(open("show_rule/data/dict.pkl", "rb"))
    # word_embedding = np.load("show_rule/data/vectors.npy")
    word_dict = pickle.load(open("show_rule/data/char_dict", "rb"))
    word_embedding = np.load("show_rule/data/char_embedding")

    # 计算两个向量的余弦相似度
    def cos(self, vector_a, vector_b):
        numerator = np.dot(vector_a, vector_b.T)
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = numerator / denominator
        sen_sim = 0.5 + 0.5 * cos  # 归一化
        # print(sen_sim)
        return sen_sim

    # 计算句向量
    def sen_vec(self, sen):
        s_vec = 0
        for idx, ch in enumerate(sen):
            word_vec = self.word_embedding[self.word_dict.get(sen[idx])]
            s_vec = s_vec + word_vec
        s_vec = s_vec / len(sen)
        return s_vec

    # 计算句子相似度
    def sen_cos(self, first, second):
        sim = self.cos(self.sen_vec(first), self.sen_vec(second))
        # print(sim)
        return sim
