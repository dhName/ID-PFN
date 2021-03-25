# -*- coding: utf-8 -*-
# @time : 2020/4/13  14:31
import tensorflow as tf
from collections import OrderedDict
import json

flags = tf.app.flags
flags.DEFINE_boolean("restore", False, "Wither restore ckpt")
flags.DEFINE_boolean("use_small", False, "Wither use small data")

# configurations for the model
flags.DEFINE_integer("type_dim", 50, "Embedding size for entity type")
flags.DEFINE_integer("position_dim", 50, "Embedding size for position")
flags.DEFINE_integer("word_dim", 100, "Embedding size for word")
flags.DEFINE_integer("lstm_dim", 500, "Num of hidden units in LSTM")
flags.DEFINE_integer("pos_max", 50, "Max position")

# configurations for training
flags.DEFINE_float("init_patterns_ratio", 0.1, "Top percentage for init patterns")
flags.DEFINE_integer("init_patterns_max", 40, "Max num of init patterns")
flags.DEFINE_float("pattern_threshold", 0.5, "Pattern threshold for update trustable pattern")
flags.DEFINE_integer("pattern_max", 5, 'Max num of updating pattern')
flags.DEFINE_float("beta", 1., "bate for attention regularization")
flags.DEFINE_integer("first_loop_epoch", 10, "First loop epoch")
flags.DEFINE_integer("epoch", 20, "Epoch")
flags.DEFINE_float("clip", 0., "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout keep prob")
flags.DEFINE_integer("batchsize", 160, "Batch size")
flags.DEFINE_float("lr", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
flags.DEFINE_boolean("zero", True, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", False, "Wither lower case")
flags.DEFINE_boolean("positive_found", True, "Wither redistribution")
flags.DEFINE_boolean("pattern_attention", True, "Wither attention regularization")
flags.DEFINE_boolean("iteration", True, "Wither bootstrap")

flags.DEFINE_boolean("pretrained_word", True, "use pretrained word")
FLAGS = tf.app.flags.FLAGS

def get_config():
    config = OrderedDict()
    config['batchsize'] = FLAGS.batchsize
    config['word_dim'] = FLAGS.word_dim
    config['position_dim'] = FLAGS.position_dim
    config['type_dim'] = FLAGS.type_dim
    config['hidden_dim'] = FLAGS.lstm_dim
    config['pos_max'] = FLAGS.pos_max
    config['positive_found'] = FLAGS.positive_found
    config['pattern_attention'] = FLAGS.pattern_attention
    config['iteration'] = FLAGS.iteration
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

    config['sel_label'] = ['/people/person/children', '/business/company/founders',
                           '/people/deceased_person/place_of_death', '/people/person/place_of_birth',
                           '/location/neighborhood/neighborhood_of', '/business/person/company',
                           '/people/person/place_lived', '/location/country/capital',
                           '/people/person/nationality', '/location/location/contains']
    # 'None']

    config['label_map'] = {'/location/country/administrative_divisions': '/location/location/contains'}

    return config


def update_config(config,args):
    parameters=args
    for key,value in parameters.items():
        config[key]=value
    return config


