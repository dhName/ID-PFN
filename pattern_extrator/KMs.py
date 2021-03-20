'''
k-means
'''

from ds_nre.re_code.data_loader import data_loader
from ds_nre.re_code.util import load_data_file, build_maps, clean, get_logger, check_env, parser_score,conver_token_to_id

import os
import heapq
import json
import numpy as np
import tensorflow as tf
from pattern_extrator.K_means import KMeansClusterer

from collections import OrderedDict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


flags = tf.app.flags
flags.DEFINE_boolean("clean",                           True,          "Clean train file")
flags.DEFINE_boolean("train",                           True,          "Whether train the model")
flags.DEFINE_boolean("restore",                         False,          "Wither restore ckpt")
flags.DEFINE_boolean("use_small",                       False,          "Wither use small data")

# configurations for the model
flags.DEFINE_integer("type_dim",                        50,             "Embedding size for entity type")
flags.DEFINE_integer("position_dim",                    50,             "Embedding size for position")
flags.DEFINE_integer("word_dim",                        100,            "Embedding size for word")
flags.DEFINE_integer("lstm_dim",                        500,            "Num of hidden units in LSTM")
flags.DEFINE_integer("pos_max",                         100,            "Max position")

# configurations for training
flags.DEFINE_float("init_patterns_ratio",               0.1,            "Top percentage for init patterns")
flags.DEFINE_integer("init_patterns_max",               20,             "Max num of init patterns")
flags.DEFINE_float("pattern_threshold",                 0.5,            "Pattern threshold for update trustable pattern")
flags.DEFINE_integer("pattern_max",                     5,              'Max num of updating pattern')
flags.DEFINE_float("beta",                              1.,             "bate for attention regularization")
flags.DEFINE_integer("first_loop_epoch",                10,             "First loop epoch")
flags.DEFINE_integer("epoch",                           50,             "Epoch")
flags.DEFINE_float("clip",                              5.,            "Gradient clip")
flags.DEFINE_float("dropout",                           0.5,            "Dropout keep prob")
flags.DEFINE_integer("batchsize",                       160,            "Batch size")
flags.DEFINE_float("lr",                                0.001,          "Initial learning rate")
flags.DEFINE_string("optimizer",                        "adam",         "Optimizer for training")
flags.DEFINE_boolean("zero",                            True,           "Wither replace digits with zero")
flags.DEFINE_boolean("lower",                           False,          "Wither lower case")
flags.DEFINE_boolean("redistribution",                  True,           "Wither redistribution")
flags.DEFINE_boolean("attention_regularization",        True,           "Wither attention regularization")
flags.DEFINE_boolean("bootstrap",                       True,           "Wither bootstrap")
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_boolean("pretrained_word", True, "use pretrained word")

word2embd={}
with open(r"E:\PycharmProjects\my_graduation_project\word_vec\vectors.txt","r",encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        raw = line.strip().split()
        word = raw[0]
        vec = list(map(float, raw[1:]))
        word2embd[word] = vec

def get_config():
    config = OrderedDict()
    config['batchsize'] = FLAGS.batchsize
    config['word_dim'] = FLAGS.word_dim
    config['position_dim'] = FLAGS.position_dim
    config['type_dim'] = FLAGS.type_dim
    config['hidden_dim'] = FLAGS.lstm_dim
    config['pos_max'] = FLAGS.pos_max
    config['redistribution'] = FLAGS.redistribution
    config['attention_regularization'] = FLAGS.attention_regularization
    config['bootstrap'] = FLAGS.bootstrap
    config['zero'] = FLAGS.zero
    config['lower'] = FLAGS.lower
    config['first_loop_epoch'] = FLAGS.first_loop_epoch
    config['epoch'] = FLAGS.epoch
    config['init_patterns_ratio'] = FLAGS.init_patterns_ratio
    config['init_patterns_max'] = FLAGS.init_patterns_max
    config['pattern_threshold'] = FLAGS.pattern_threshold
    config['pattern_max'] = FLAGS.pattern_max
    config['beta'] = FLAGS.beta
    config['lr'] = FLAGS.lr
    config['clip'] = FLAGS.clip
    config['lr_method'] = FLAGS.optimizer
    config['restore'] = FLAGS.restore
    config['use_small'] = FLAGS.use_small
    config['dropout'] = FLAGS.dropout
    config['pretrained_word'] = FLAGS.pretrained_word
    config['sel_label'] = ['/people/person/children',                   '/business/company/founders',
                           '/people/deceased_person/place_of_death',    '/people/person/place_of_birth',
                           '/location/neighborhood/neighborhood_of',    '/business/person/company',
                           '/people/person/place_lived',                '/location/country/capital',
                           '/people/person/nationality',                '/location/location/contains']

    config['label_map'] = {'/location/country/administrative_divisions': '/location/location/contains'}


    return config


def train(config, logger):
    logger.info('------load train data------')
    train_datas = load_data_file(r'E:\PycharmProjects\my_graduation_project\data\train.json', zero=config['zero'], lower=config['lower'])
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

    config['position_num'] = pos_max
    print(config['position_num'], 'position_num')

    id2label={}
    for label,id in config["label_dict"].items():
        id2label[id]=label

    pattern_dict = {}
    for data in train_data_loader.dataset_load:
        pattern = data[-1]
        label = data[-2]
        if label not in pattern_dict:
            pattern_dict[label] = {}
        if pattern not in pattern_dict[label]:
            pattern_dict[label][pattern] = 0
        pattern_dict[label][pattern] += 1

    for label,patterns in pattern_dict.items():
        filename = str(id2label.get(label)).replace("/", "_")
        if "contains" in filename:
            continue

        pattern_str_2_emb={}
        pattern_embs=[]
        for pattern in patterns:
            pattern_emb=get_pattern_embeding(pattern,config['vocab'])
            if pattern_str_2_emb.keys().__contains__(str(pattern)):
                continue
            pattern_str_2_emb[pattern]=pattern_emb
            pattern_embs.append(pattern_emb)

        print(filename+"开始聚类")
        km = KMeansClusterer(ndarray=np.asarray(pattern_embs),cluster_num=int(pattern_embs.__len__()/10))
        result = km.cluster()

        cluster_max_num=0
        good_cluster=[]
        for per_cluster in result:
            if cluster_max_num < per_cluster.__len__():
                cluster_max_num=per_cluster.__len__()
                good_cluster=per_cluster

        with open(filename+".txt","w+",encoding="utf8") as fin:
            for point in good_cluster:
                for pattern,pattern_emb in pattern_str_2_emb.items():
                    if arrays_equal(np.asarray(pattern_emb),np.asarray(point)):
                        fin.write(pattern+"\n")



def get_pattern_embeding(pattern,vocab):
    result=0
    words = pattern.split(" ")
    for word in words:
        if word2embd.keys().__contains__(word):
            word_emb = word2embd[word]
        else:
            word_emb=[0.0 for temp_i in range(100)]
        result+=np.asarray(word_emb)

    result=result/pattern.__len__()
    return result

def arrays_equal(a, b):
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True


def main(_):
    logger = get_logger("k-means-pattren")
    if FLAGS.train:
        if os.path.exists('config_file'):
            with open('config_file', 'r', encoding='utf-8') as r:
                config = json.load(r)
        else:
            config = get_config()

        train(config, logger)



if __name__ == '__main__':
    tf.app.run(main)

