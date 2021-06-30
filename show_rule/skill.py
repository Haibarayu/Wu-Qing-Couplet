import copy
import fool
import pynlpir
import thulac
import pkuseg
import jieba
import jieba.posseg as pseg
jieba.initialize()

from pyhanlp import *
from snownlp import SnowNLP
from collections import Counter


class Skill:
    def partition(self, sentence, pku_seg, thu_seg):
        all = []
        # HanLP
        han = []
        han_ps = HanLP.segment(sentence)
        for han_p in han_ps:
            han.append(han_p.word)
        all.append(han)

        # Jieba
        jie = []
        jie_ps = pseg.cut(sentence)
        for jie_p in jie_ps:
            jie.append(jie_p.word)  # list.append(w.word+ w.flag)
        all.append(jie)

        # pynlpir
        # pynlpir.open()  # 打开分词器
        pyn = pynlpir.segment(sentence, pos_tagging=False)
        all.append(pyn)
        # pynlpir.close()

        # thu
        thu = []
        thu_ps = thu_seg.cut(sentence, text=False)
        for thu_p in thu_ps:
            thu.append(thu_p[0])
        all.append(thu)

        # pku
        pku_ps = pku_seg.cut(sentence)
        all.append(pku_ps)

        # fool
        foo = []
        foo_ps = fool.cut(sentence)
        for foo_p in foo_ps[0]:
            foo.append(foo_p)
        all.append(foo)

        # snowNLP
        sno = SnowNLP(sentence).words
        all.append(sno)

        # print(all)
        return all

    def count_choose(self, sentence, pku_seg, thu_seg):
        all = self.partition(sentence, pku_seg, thu_seg)  # 直接对列表元素进行remove可能会出现删不干净的情况，故进行一个深拷贝的备份
        '''消除分词情况不佳的情况'''
        screen_all = copy.deepcopy(all)
        for i in all:  # 消除不分词，以及全部分成一个字的情况
            temp_list = []
            sum = 0
            for ii in i:
                temp_list.append(len(ii))
                sum += len(ii)  # 需要分词的这句话的长度
            if sum >= 5:
                for ii in i:
                    if len(ii) >= sum:  # 若分出的词大于等于句子的长度，则舍弃
                        if len(screen_all) <= 1:
                            continue
                        screen_all.remove(i)
                        continue
                if set(temp_list) == {1}:  # 消除全部分为一个字的情况
                    if len(screen_all) <= 1:
                        continue
                    screen_all.remove(i)

        '''选出出现次数最多的情况'''
        count_times = []
        for i in screen_all:
            count_times.append(screen_all.count(i))
        m = max(count_times)
        n = screen_all[0]
        if m > 2:
            n = screen_all[count_times.index(m)]
        elif m <= 2:
            list_first = []
            for i in screen_all:
                list_first.append(len(i[0]))
            count_list_first = Counter(list_first)
            number = count_list_first.most_common(1)[0][0]  # 获得次数最多的数
            for i in screen_all:
                if len(i[0]) == number:
                    n = i
                    break
        # str_fall = ' '.join(n)
        return n
        # q.put(n)

    def count_more(self, first_fall, second_fall):
        # first_fall = self.count_choose(first)
        # second_fall = self.count_choose(second)
        # print(first_fall, second_fall)
        all_result = 0
        len_second = 0
        len_first = 0
        # 去掉结构断点相同的情况
        for i in range(max(len(first_fall), len(second_fall))):
            if len(first_fall) == max(len(first_fall), len(second_fall)):
                len_first += len(first_fall[i])
                len_second = 0
                for ii in range(len(second_fall)):
                    len_second += len(second_fall[ii])
                    if len_first == len_second:
                        all_result += 1
            else:
                len_second += len(second_fall[i])
                len_first = 0
                for ii in range(len(first_fall)):
                    len_first += len(first_fall[ii])
                    if len_first == len_second:
                        all_result += 1
        point = max(len(first_fall), len(second_fall)) - all_result
        result = 0
        if point == 1:
            result = 0.5
        elif point == 2:
            result = 0.6
        elif point == 3:
            result = 0.8
        elif point > 3:
            result = 1
        return result
