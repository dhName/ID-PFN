from django.shortcuts import render, HttpResponse

import os, json
from ds_nre.re_code.util import *
from ds_nre.re_code.Config import *
from ds_nre.re_code.pattern import *
from ds_nre.re_code import run_train
from ds_nre.re_code import run_test

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def index(request):
    return render(request, "index.html")


def parameters_setting(request):
    return render(request, "parameters_setting.html")


def pattern_filters(request):
    return render(request, "pattern_filters.html")


def trainning_result(request):
    return render(request, "trainning_result.html")


def model_prediction(request):
    return render(request, "model_prediction.html")


# 1.parameter_setting
# 根据ajax传递的json来进行config默认的修改。
def setting_parameters(request):
    request_body=str(request.body, encoding="utf-8")
    args = json.loads(request_body)

    config = get_config()
    config = update_config(config, args)

    check_env(config)

    logger = get_logger("models/"+config["model_name"]+"/train")

    get_mf_pattern(config, logger, max_throld=config["init_patterns_max"])

    return HttpResponse("上传成功")


# 2.pattern_filter
# 加载现有的模型的参数文件名称返回列表
def load_parameters_file(request):
    model_dir = os.path.join(BASE_DIR, "models")
    model_files = os.listdir(model_dir)

    return HttpResponse(json.dumps(model_files), content_type="application/json")


# 返回一个字符串，这个字符串是一个个pattern,每个pattern用<br>分隔。前端直接Innerhtml就可以实现
def pattern_filter(request):
    request_body = str(request.body,encoding="utf-8")
    req_dict = json.loads(request_body)
    model_name = req_dict["model_name"]
    relation_type = req_dict["relation_type"]

    with open("models/" + model_name + '/config_file', 'r', encoding='utf-8') as r:
        config = json.load(r)

    config["model_name"]=model_name

    trustable_pattern_label_str = read_initial_pattern(config)
    result = trustable_pattern_label_str[relation_type]

    return_str = "\r\n".join(result)

    return HttpResponse(return_str)


# 读取ajax中携带的type，过滤后的pattern，文件名存储起来
def filtered_pattern_checkout(request):
    request_body = str(request.body,encoding="utf-8")
    req_dict = json.loads(request_body)

    model_name = req_dict["model_name"]
    pattern_type = req_dict["pattern_type"]
    filtered_pattern = req_dict["filtered_pattern"]

    patterns = filtered_pattern.split("\r\n")

    with open("models/" + model_name + '/config_file', 'r', encoding='utf-8') as r:
        config = json.load(r)

    config["model_name"]=model_name

    pattern_checkout(config, pattern_type, patterns)

    return HttpResponse("上传成功")


# 后台开始训练
def train(request):
    request_body = str(request.body,encoding="utf-8")
    req_dict = json.loads(request_body)

    model_name = req_dict["model_name"]

    with open("models/" + model_name + '/config_file', 'r', encoding='utf-8') as r:
        config = json.load(r)

    run_train.train(config)

    run_test.model_test(model_name)

    return HttpResponse("开始训练...")


# 3.tranning_result
# 动态加载现有训练好的model,返回一个列表
def load_model_name(request):
    model_dir = os.path.join(BASE_DIR, "models")
    # print(model_dir)
    model_files = os.listdir(model_dir)
    # print(model_files)
    return HttpResponse(json.dumps(model_files), content_type="application/json")


# 根据ajax传递的model_name进行找到此模型的性能指标，返回json
def train_result(request):
    model_name = str(request.body, encoding="utf-8")
    result = []
    with open("models/" + model_name + '/test_result.txt', 'r', encoding='utf-8') as r:
        lines = r.readlines()
        for line in lines:
            splits = line.split("\t")
            if splits.__len__() == 4:
                result.append(
                    {"relation_name": splits[0], "precision": splits[1], "recall": splits[2], "F1": splits[3]})

    return HttpResponse(json.dumps(result), content_type="application/json")


# 4.prediction
# 预测,返回关系类别字符串
def prediction(request):
    request_body = str(request.body,encoding="utf-8")
    req_dict = json.loads(request_body)

    model_name = req_dict["model_name"].strip()
    sentence = req_dict["sentence"].strip()
    entity1 = req_dict["entity1"].strip()
    entity1_type = req_dict["entity1_type"].strip()
    entity2 = req_dict["entity2"].strip()
    entity2_type = req_dict["entity2_type"].strip()

    prediction = run_test.model_prediction(model_name=model_name, sentence=sentence,
                                           entity1=entity1, entity1_type=entity1_type,
                                           entity2=entity2, entity2_type=entity2_type)

    return HttpResponse(prediction)
