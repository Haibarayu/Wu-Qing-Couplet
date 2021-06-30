import json
import time

from flask import Flask, render_template, request
from generation_model.generate import Gen
from show_rule.rule import *


# ——————预加载—————— #
# 生成模型
gen_model = Gen()
# 分词
HanLP.segment("无情对自动对联系统")
pseg.cut("无情对自动对联系统")
fool.cut("无情对自动对联系统")
pku_seg = pkuseg.pkuseg()
thu_seg = thulac.thulac(seg_only=True)
# 无情对语料
file = open("couplet_dict.json", 'r', encoding='utf-8')
couplets = json.load(file)
file.close()
# —————————————————— #


app = Flask(__name__)

@app.route('/')
def wuqing():
    return render_template("wuqing.html")

@app.route('/couplet', methods=['POST'])
def couplet():
    st = time.time()
    first = request.form.get("first")  # 接收用户输入的上联
    print(first)
    if len(first) == 0:
        output = ["您的输入为空！"]
    elif len(first) > 50:
        output = ["您的输入太长了!"]
    else:
        gen_first = " ".join(first)
        gen_second = gen_model.gen(gen_first)
        secondset = [second.replace(" ", "") for second in gen_second]
        print(secondset)
        score = Rule().scores(first, secondset, 0.25, 0.35, 0.4, pku_seg, thu_seg)
        print(score)
        if len(score) < 5:
            output = score  # 符合硬规则的数目小于N的情况
        else:
            output = score[:5]  # 选择TopN
        # 添加语料中已有结果
        for cou in couplets:
            if first == cou['first']:
                if cou['second'] not in output:
                    output.insert(0, cou['second'])
                    output.pop()
                break
        print(output)

    ed = time.time()
    print("消耗时间:", ed - st)
    return render_template("wuqing.html", input=first, out=output)


if __name__ == '__main__':
    app.run(port="5555")
