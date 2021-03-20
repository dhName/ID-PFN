# -*- coding: utf-8 -*-
# @time : 2020/4/13  17:30

from ds_nre.re_code.model import model
from ds_nre.re_code.data_loader import data_loader
from ds_nre.re_code.util import load_data_file, get_logger, conver_token_to_id, get_positions, get_types, parser_score

import re
import json
import numpy as np
import tensorflow as tf


def model_test(model_name):
    logger = get_logger("models/" + model_name + "/test")

    with open("models/" + model_name + '/config_file', 'r', encoding='utf-8') as r:
        config = json.load(r)

    print('load test data')

    vocab, type_dict, label_dict = np.load("models/" + model_name + '/maps.npy', allow_pickle=True)
    config['redistribution'] = False
    config['restore'] = True
    config['clean'] = False
    config['train'] = False

    for k, v in config.items():
        logger.info('config {}: {}'.format(k, v))

    config['vocab'] = vocab
    config['type_dict'] = type_dict
    config['label_dict'] = label_dict

    config['word_num'] = len(vocab)
    config['type_num'] = len(type_dict)
    config['label_num'] = len(label_dict)
    config['position_num'] = config['pos_max']

    test_datas = load_data_file('data/test.json', zero=config['zero'], lower=config['lower'])
    test_data_loader = data_loader(test_datas, config)
    data_count, badcase, ignore, _ = test_data_loader.load()
    logger.info('test data: {}'.format(len(data_count)))
    logger.info('test badcase: {}'.format(badcase))

    with tf.Graph().as_default() as g:
        test_model = model(config, 'test')
        test_model.build(g)

    logger.info('***TEST***')
    test_score, test_accs = test_model.run_evaule(test_data_loader)
    parser_score(0, '', test_score, test_accs, logger, 'test')
    logger.info('******\n')

    with open("models/" + model_name + '/test_result.txt', "w", encoding="utf-8") as fout:
        for relation, name2score in test_score.items():
            fout.write(relation + "\t" + name2score["P"] + "\t" + name2score["R"] + "\t" + name2score["F1"] + "\n")


def model_prediction(model_name, sentence, entity1, entity1_type, entity2, entity2_type):
    model_dir = "models/" + model_name + "/best_model/"

    #预加载工作
    with open("models/" + model_name + '/config_file', 'r', encoding='utf-8') as r:
        config = json.load(r)

    vocab, type_dict, label_dict = np.load("models/" + model_name + '/maps.npy', allow_pickle=True)
    config['redistribution'] = False
    config['restore'] = True
    config['clean'] = False
    config['train'] = False

    config['vocab'] = vocab
    config['type_dict'] = type_dict
    config['label_dict'] = label_dict

    config['word_num'] = len(vocab)
    config['type_num'] = len(type_dict)
    config['label_num'] = len(label_dict)
    config['position_num'] = config['pos_max']

    with tf.Graph().as_default() as g:
        test_model = model(config, 'test')
        test_model.build(g)

    print('restore model from {} for val or test'.format(model_dir))
    test_model.saver.restore(test_model.sess, model_dir)

    #输入数据的处理
    sentence = re.sub(r'"', ' \'\' ', sentence)  # deal case like "aaa bbb"
    sentence = re.sub(r'\s+', ' ', sentence.strip())

    # 存储entity和entity_type
    entity_to_type = {}
    entity2type = {}

    entity1_mentions = '<START>{}<END>'.format(entity1).split(' ')
    entity2_mentions = '<START>{}<END>'.format(entity2).split(' ')
    for entity_mention in entity1_mentions:
        entity_to_type[entity_mention] = entity1_type
    for entity_mention in entity2_mentions:
        entity_to_type[entity_mention] = entity2_type

    entity2type[entity1] = entity1_type
    entity2type[entity2] = entity2_type

    # 正则匹配
    ent1_ent2_search = re.finditer(r'(?: |^)({0}( (?:.*?) | ){1})(?: |$)'.format(entity1, entity2), sentence)
    ent2_ent1_search = re.finditer(r'(?: |^)({1}( (?:.*?) | ){0})(?: |$)'.format(entity1, entity2), sentence)

    predict_label=""
    # entity1_entity2
    for ent1_ent2 in ent1_ent2_search:
        start, end = ent1_ent2.span(1)

        short = re.sub(r'^{} '.format(entity1), '<START>{}<END> '.format(entity1), ent1_ent2.group(1))
        short = re.sub(r' {}$'.format(entity2), ' <START>{}<END>'.format(entity2), short)

        sentence = sentence[:start] + short + sentence[end:]
        sentence = sentence.split(' ')

        positions = get_positions(sentence, ent1_ent2, config['pos_max'])
        types = get_types(sentence, entity_to_type, config['type_dict'])

        feature = []
        for i, word in enumerate(sentence):
            if word[:7] == '<START>':
                word = word[7:]
            if word[-5:] == '<END>':
                word = word[:-5]

            wordid = conver_token_to_id(word, config['vocab'])
            feature.append(wordid)

        predict = test_model.predict([[feature], [positions[0]], [positions[1]], [types], [len(feature)]])
        predict_label = test_model.id_to_label[predict[0]]

    # entity2_entity1
    for ent2_ent1 in ent2_ent1_search:
        start, end = ent2_ent1.span(1)

        short = re.sub(r'^{} '.format(entity2), '<START>{}<END> '.format(entity2), ent2_ent1.group(1))
        short = re.sub(r' {}$'.format(entity1), ' <START>{}<END>'.format(entity1), short)

        sentence = sentence[:start] + short + sentence[end:]
        sentence = sentence.split(' ')

        positions = get_positions(sentence, ent2_ent1, config['pos_max'])
        types = get_types(sentence, entity_to_type, config['type_dict'])

        # conver word to id and build attetion regulation label
        feature = []

        for i, word in enumerate(sentence):
            if word[:7] == '<START>':
                word = word[7:]
            if word[-5:] == '<END>':
                word = word[:-5]
                # att_label[i] = 1.
            wordid = conver_token_to_id(word, config['vocab'])
            feature.append(wordid)

        predict = test_model.predict([[feature], [positions[0]], [positions[1]], [types], [len(feature)]])
        predict_label = test_model.id_to_label[predict[0]]

    return predict_label
