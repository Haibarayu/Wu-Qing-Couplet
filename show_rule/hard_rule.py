# ----------硬规则---------- #
class HardRule:
    def hard_rule(self, first, second):  # first：出句，second：备选对句
        upidx = {}
        uplist = []
        downidx = {}
        downlist = []
        if len(first) != len(second):
            # print("上下联字数不等！")
            return 0
        for ch in first:
            if ch not in upidx.keys():
                upidx[ch] = len(upidx)
            uplist.append(upidx[ch])
        # print(uplist)
        for ch in second:
            if ch in upidx.keys():
                # print("下联出现与上联相同的字！")
                return 0
            if ch not in downidx.keys():
                downidx[ch] = len(downidx)
            downlist.append(downidx[ch])
        # print(downlist)
        if uplist != downlist:
            # print("下联对应位置的字不满足上联约束！")
            return 0
        else:
            # print("符合硬规则。")
            return 1

    # 筛选
    def hard_rule_filter(self, first, secondset):  # first：出句，secondset：备选对句集
        # print("出句：" + first)
        result = []
        for second in secondset:
            if self.hard_rule(first, second):
                result.append(second)
        return result
