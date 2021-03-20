# -*- coding: utf-8 -*-
# @time : 2020/4/13  15:00
from ds_nre.re_code.data_loader import data_loader
from ds_nre.re_code.util import load_data_file, build_maps


import heapq


#通过最大频率算法获取初始的pattern
def get_mf_pattern(config,logger,max_throld=20):

    train_datas = load_data_file('data/train.json', zero=True, lower=False)
    config = build_maps(train_datas, config, logger)

    train_data_loader = data_loader(train_datas, config)
    data_count, badcase, ignore, pos_max = train_data_loader.load(config['use_small'])

    id2label = {}
    for label, id in config["label_dict"].items():
        id2label[id] = label

    pattern_dict = {}
    for data in train_data_loader.dataset:
        pattern = data[-1]
        label = data[-2]
        if label not in pattern_dict:
            pattern_dict[label] = {}
        if pattern not in pattern_dict[label]:
            pattern_dict[label][pattern] = 0
        pattern_dict[label][pattern] += 1

    trustable_pattern = {}
    for k, v in pattern_dict.items():
        sel_num = int(config['init_patterns_ratio'] * len(v))
        sel_num = max_throld if sel_num > max_throld else sel_num
        v = heapq.nlargest(sel_num, v.items(), key=lambda x: x[1])
        trustable_pattern[k] = list([v1 for v1, v2 in v])

    with open("models/"+config["model_name"]+"/pattern_initial.txt","w",encoding="utf-8") as fout:
        for key,values in trustable_pattern.items():
            fout.write("*******************************************************\n")
            fout.write(str(id2label.get(key))+":\n")
            for value in values:
                fout.write(str(value) +"\n")

    with open("models/"+config["model_name"]+"/pattern_filtered.txt","w",encoding="utf-8") as fout:
        for key,values in trustable_pattern.items():
            fout.write("*******************************************************\n")
            fout.write(str(id2label.get(key))+":\n")
            for value in values:
                fout.write(str(value) +"\n")



#读取初始的pattern
def read_initial_pattern(config):
    trustable_pattern_label_str={}
    with open("models/"+config["model_name"]+"/pattern_initial.txt", "r", encoding="utf8") as f:
        lines = f.readlines()
        startFlag = False
        label = ""
        for line in lines:
            line = line.strip()
            if startFlag:
                label = line[0:line.__len__() - 1]
                trustable_pattern_label_str[label] = []
                startFlag = False
            else:
                if line.startswith("**********************************"):
                    startFlag = True
                else:
                    trustable_pattern_label_str[label].append(line)
    return trustable_pattern_label_str


def pattern_checkout(config,ralation,patterns):
    trustable_pattern_label_str = {}
    with open("models/"+config["model_name"]+"/pattern_filtered.txt", "r", encoding="utf8") as f:
        lines = f.readlines()
        startFlag = False
        label = ""
        for line in lines:
            line = line.strip()
            if startFlag:
                label = line[0:line.__len__() - 1]
                trustable_pattern_label_str[label] = []
                startFlag = False
            else:
                if line.startswith("**********************************"):
                    startFlag = True
                else:
                    trustable_pattern_label_str[label].append(line)

    trustable_pattern_label_str[ralation]=[]
    for pattern in patterns:
        trustable_pattern_label_str[ralation].append(pattern)

    with open("models/" + config["model_name"] + "/pattern_filtered.txt", "w", encoding="utf8") as fout:
        for key, values in trustable_pattern_label_str.items():
            fout.write("*******************************************************\n")
            fout.write(str(key) + ":\n")
            for value in values:
                fout.write(str(value) + "\n")
