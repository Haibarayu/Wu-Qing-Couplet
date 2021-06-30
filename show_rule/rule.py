from show_rule.hard_rule import HardRule
from show_rule.coherent import Coherent
from show_rule.char_to_char import CharToChar
from show_rule.skill import *
from show_rule.sen_sim import SenSim

con = Coherent()
char2char = CharToChar()
sensim = SenSim()


class Rule:
    '''
    —————————————参数——————————————
    first：用户所给上联 secondset：生成备选下联
    x：连贯性权重 y：字字工对权重 z：借义转义权重
    pku_seg, thu_seg: 预加载的分词模型对象
    '''

    def scores(self, first, secondgen, x, y, z, pku_seg, thu_seg):
        # ——————————硬规则——————————— #
        secondset = HardRule().hard_rule_filter(first, secondgen)
        if len(secondset) == 0:
            print("备选对句均不满足硬规则！")
            return secondgen
        # ——————————连贯性——————————— #
        coh = []
        for second in secondset:
            item1 = {}
            item1["cou"] = second
            item1["prob"] = con.prob(second)
            coh.append(item1)
        # ——————————字字工对——————————— #
        ctc = []
        for second in secondset:
            item2 = {}
            item2["cou"] = second
            item2["ctc"] = char2char.char_to_char(first, second)
            ctc.append(item2)
        # ————————————借义转义————————————— #
        # ——————————节奏改变——————————— #
        ski = []
        pynlpir.open()  # 打开pynlpir分词器
        fisrt_skill = Skill().count_choose(first, pku_seg, thu_seg)
        for second in secondset:
            item3 = {}
            item3["cou"] = second
            second_skill = Skill().count_choose(second, pku_seg, thu_seg)
            item3["skill"] = Skill().count_more(fisrt_skill, second_skill)
            ski.append(item3)
        pynlpir.close()  # 关闭pynlpir分词器
        # ——————————意意相离——————————— #
        sim = []
        for second in secondset:
            item4 = {}
            item4["cou"] = second
            item4["sim"] = sensim.sen_cos(first, second)
            sim.append(item4)

        # =============规则============= #
        rule = []
        for idx, second in enumerate(secondset):
            item = {}
            item["first"] = first
            item["second"] = second
            item["score"] = coh[idx]["prob"] * x + ctc[idx]["ctc"] * y + (
                    0.35 * (1 - sim[idx]["sim"]) + 0.65 * ski[idx]["skill"]) * z
            rule.append(item)
        score = sorted(rule, key=lambda x: x['score'], reverse=True)
        result = [cou['second'] for cou in score]
        # print(result)
        return result
