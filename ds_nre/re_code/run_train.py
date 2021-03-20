# -*- coding: utf-8 -*-
# @time : 2020/4/13  14:31
from ds_nre.re_code.Config import get_config,update_config
from ds_nre.re_code.model import model
from ds_nre.re_code.data_loader import data_loader
from ds_nre.re_code.util import load_data_file, build_maps, clean, get_logger, check_env, parser_score

import os
import heapq
import json
import numpy as np
import tensorflow as tf

def train(config=None):
    logger = get_logger("models/"+config["model_name"]+"/train")

    check_env(config)

    logger.info('------load train data------')
    train_datas = load_data_file('data/train.json', zero=config['zero'], lower=config['lower'])
    config = build_maps(train_datas, config, logger)

    train_data_loader = data_loader(train_datas, config)
    data_count, badcase, ignore, pos_max = train_data_loader.load(config['use_small'])

    logger.info('---Data count---')
    for label, num in data_count.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Badcase count---')
    for label, num in badcase.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Ignore count---')
    for label, num in ignore.items():
        logger.info('{:<50}\t{:>}'.format(label, num))
    logger.info('--------------------------\n')

    logger.info('------load val data------')
    val_datas = load_data_file('data/dev.json', zero=config['zero'], lower=config['lower'])
    val_data_loader = data_loader(val_datas, config)
    data_count, badcase, ignore, _ = val_data_loader.load(config['use_small'])

    logger.info('---Data count---')
    for label, num in data_count.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Badcase count---')
    for label, num in badcase.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Ignore count---')
    for label, num in ignore.items():
        logger.info('{:<50}\t{:>}'.format(label, num))
    logger.info('--------------------------\n')

    logger.info('------load test data------')
    test_datas = load_data_file('data/test.json', zero=config['zero'], lower=config['lower'])
    test_data_loader = data_loader(test_datas, config)
    data_count, badcase, ignore, _ = test_data_loader.load(config['use_small'])

    logger.info('---Data count---')
    for label, num in data_count.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Badcase count---')
    for label, num in badcase.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Ignore count---')
    for label, num in ignore.items():
        logger.info('{:<50}\t{:>}'.format(label, num))
    logger.info('--------------------------\n')

    config['position_num'] = pos_max

    id2label = {}
    for label, id in config["label_dict"].items():
        id2label[id] = label

    #开始读取优质的pattern
    trustable_pattern_label_str = {}
    with open("models/"+config["model_name"]+"pattern_filterd.txt", "r+", encoding="utf8") as f:
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

    trustable_pattern = {}
    for label, values in trustable_pattern_label_str.items():
        label = label
        values = values
        labelid = config["label_dict"][label]
        trustable_pattern[labelid] = values

    with tf.Graph().as_default() as g:
        train_model = model(config, 'train')
        train_model.build(g)

    with tf.Graph().as_default() as g:
        val_model = model(config, 'val')
        val_model.build(g)

    best_score = [-1., 0, None, None]
    loop = config['first_loop_epoch'] if config['bootstrap'] else config['epoch']

    logger.info('not bootstrap or first loop of bootstrap:{}'.format(loop))
    for epoch in range(loop):

        logger.info('***TRAIN: {}***'.format(epoch))
        train_model.run_train(train_data_loader, trustable_pattern)

        logger.info('***VAL***')
        cur_score, accs = val_model.run_evaule(val_data_loader, None)
        best_score, is_new = parser_score(epoch, best_score, cur_score, accs, logger)

        if is_new:
            logger.info('***TEST***')
            test_score, test_accs = val_model.run_evaule(test_data_loader, None)
            parser_score(epoch, '', test_score, test_accs, logger, 'test')
        logger.info('******\n')

    # for other loop
    if config['bootstrap']:
        logger.info('for other bootstrap loop')

        # update patterns
        kls, patterns, labels = train_model.get_kls_patterns_labels(train_data_loader)
        pattern_condidates = {}
        for kl, pattern, label in zip(kls, patterns, labels):
            pattern_score = 1. / (1. + kl)
            if pattern_score > config['pattern_threshold'] and pattern not in trustable_pattern.get(label):
                if label not in pattern_condidates:
                    pattern_condidates[label] = {}
                pattern_condidates[label][pattern] = pattern_score

        print('for update patterns')
        for label in id2label.keys():
            print(str(label) + ":" + str(id2label.get(label)))
            if label not in pattern_condidates:
                continue
            num = len(pattern_condidates[label]) if len(pattern_condidates[label]) < config['pattern_max'] \
                else config['pattern_max']
            pattern_condidates_label = heapq.nlargest(num, pattern_condidates[label].items(), key=lambda x: x[1])
            for pattern_condidate in pattern_condidates_label:
                trustable_pattern[label].append(pattern_condidate[0])

        for epoch in range(config['epoch'] - config['first_loop_epoch']):
            logger.info('***TRAIN: {}***'.format(epoch))
            train_model.run_train(train_data_loader, trustable_pattern)

            logger.info('***VAL***')
            cur_score, accs = val_model.run_evaule(val_data_loader, None)
            best_score, is_new = parser_score(epoch, best_score, cur_score, accs, logger)

            if is_new:
                logger.info('***TEST***')
                test_score, test_accs = val_model.run_evaule(test_data_loader, None)
                parser_score(epoch, '', test_score, test_accs, logger, 'test')
            logger.info('******\n')

            # update patterns
            kls, patterns, labels = train_model.get_kls_patterns_labels(train_data_loader)
            pattern_condidates = {}
            for kl, pattern, label in zip(kls, patterns, labels):
                pattern_score = 1. / (1. + kl)
                if pattern_score > config['pattern_threshold'] and pattern not in trustable_pattern.get(label):
                    if label not in pattern_condidates:
                        pattern_condidates[label] = {}
                    pattern_condidates[label][pattern] = pattern_score

            print('for update patterns')
            for label in id2label.keys():
                if label not in pattern_condidates:
                    continue
                num = len(pattern_condidates[label]) if len(pattern_condidates[label]) < config['pattern_max'] \
                    else config['pattern_max']
                pattern_condidates_label = heapq.nlargest(num, pattern_condidates[label].items(), key=lambda x: x[1])
                for pattern_condidate in pattern_condidates_label:
                    trustable_pattern[label].append(pattern_condidate[0])

if __name__ == '__main__':
    train()