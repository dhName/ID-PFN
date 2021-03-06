# -*- coding: utf-8 -*-
# @time : 2020/3/12  15:33

import numpy as np
from tqdm import tqdm
import nltk
import re
import random
from ds_nre.re_code.util import *



class data_loader(object):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.sel_relation = set(config['sel_label']) if config['sel_label'] else None
        self.label_pre_map = config['label_map'] if config['label_map'] else None
        self.dataset_confident = []
        self.dataset_unconfident = []
        self.redistribution = config['redistribution']

    def load(self, use_small=False):
        dataset_ret = []
        pbar = tqdm(total=len(self.dataset) if not use_small else 100)
        datas_count = {'all':0}
        badcase = {}
        ignore = {}
        pos_max = 0
        for i, data in enumerate(self.dataset):
            if use_small and i == 100:
                print('use_samll', i)
                break

            sentText = data['sentText']
            sentText = re.sub(r'"', ' \'\' ', sentText) # deal case like "aaa bbb"
            sentText = re.sub(r'\s+', ' ', sentText.strip())  # remove extra spaces
            relation = data['relationMentions']
            entitys = data['entityMentions']

            # build entity to type mapping for type embedding
            entity_to_type = {}
            for entity in entitys:
                mentions = '<START>{}<END>'.format(entity['text']).split(' ')
                for mention in mentions:
                    entity_to_type[mention] = entity['label']

            for rel in relation:

                if self.label_pre_map and rel['label'] in self.label_pre_map:
                    rel['label'] = self.label_pre_map[rel['label']]

                if (self.sel_relation and rel['label'] not in self.sel_relation):
                    if rel['label'] not in ignore:
                        ignore[rel['label']] = 0
                    ignore[rel['label']] += 1
                    continue

                entity1 = rel['em1Text']
                entity2 = rel['em2Text']
                label = conver_token_to_id(rel['label'], self.config['label_dict'])

                ent1_ent2_search = re.finditer(r'(?: |^)({0}( (?:.*?) | ){1})(?: |$)'.format(entity1, entity2), sentText)
                ent2_ent1_search = re.finditer(r'(?: |^)({1}( (?:.*?) | ){0})(?: |$)'.format(entity1, entity2), sentText)

                for ent1_ent2 in ent1_ent2_search:
                    pattern = ent1_ent2.group(2).strip()
                    start, end = ent1_ent2.span(1)

                    short = re.sub(r'^{} '.format(entity1), '<START>{}<END> '.format(entity1), ent1_ent2.group(1))
                    short = re.sub(r' {}$'.format(entity2), ' <START>{}<END>'.format(entity2), short)

                    sentence = sentText[:start] + short + sentText[end:]
                    sentence = sentence.split(' ')

                    positions = get_positions(sentence, ent1_ent2, self.config['pos_max'])
                    types = get_types(sentence, entity_to_type, self.config['type_dict'])

                    # conver word to id and build attetion regulation label
                    feature = []

                    att_label = [0.] * len(sentence)  # for attetion regulation

                    att_flag = 0
                    change_falg=False
                    for i, word in enumerate(sentence):
                        if word[:7] == '<START>':
                            word = word[7:]
                            # att_label[i] = 1.
                            att_flag += 1
                        if word[-5:] == '<END>':
                            word = word[:-5]
                            # att_label[i] = 1.
                            att_flag += 1
                            change_falg=True
                        if not change_falg and att_flag == 2:
                            att_label[i] = 1.
                        change_falg=False

                        wordid = conver_token_to_id(word, self.config['vocab'])
                        feature.append(wordid)

                    fenmu = sum(att_label)
                    if int(fenmu) == 0:
                        continue

                    att_label = [f / fenmu for f in att_label]
                    pos_max = max(pos_max, max(positions[0]), max(positions[1]))
                    dataset_ret.append([feature, positions, types, att_label, label, pattern])
                    if rel['label'] not in datas_count:
                        datas_count[rel['label']] = 0
                    datas_count[rel['label']] += 1
                    datas_count['all'] += 1

                for ent2_ent1 in ent2_ent1_search:
                    pattern = ent2_ent1.group(2).strip()
                    start, end = ent2_ent1.span(1)

                    short = re.sub(r'^{} '.format(entity2), '<START>{}<END> '.format(entity2), ent2_ent1.group(1))
                    short = re.sub(r' {}$'.format(entity1), ' <START>{}<END>'.format(entity1), short)

                    sentence = sentText[:start] + short + sentText[end:]
                    sentence = sentence.split(' ')

                    positions = get_positions(sentence, ent2_ent1, self.config['pos_max'])
                    types = get_types(sentence, entity_to_type, self.config['type_dict'])

                    # conver word to id and build attetion regulation label
                    feature = []
                    att_label = [0.] * len(sentence)  # for attetion regulation

                    att_flag = 0
                    change_falg = False
                    for i, word in enumerate(sentence):
                        if word[:7] == '<START>':
                            word = word[7:]
                            # att_label[i] = 1.
                            att_flag += 1
                        if word[-5:] == '<END>':
                            word = word[:-5]
                            # att_label[i] = 1.
                            att_flag += 1
                            change_falg = True
                        if not change_falg and att_flag == 2:
                            att_label[i] = 1.
                        change_falg = False

                        wordid = conver_token_to_id(word, self.config['vocab'])
                        feature.append(wordid)

                    fenmu = sum(att_label)
                    if int(fenmu) == 0:
                        continue
                    att_label = [f / fenmu for f in att_label]
                    pos_max = max(pos_max, max(positions[0]), max(positions[1]))
                    dataset_ret.append([feature, positions, types, att_label, label, pattern])
                    if rel['label'] not in datas_count:
                        datas_count[rel['label']] = 0
                    datas_count[rel['label']] += 1
                    datas_count['all'] += 1

            pbar.update(1)
        pbar.close()
        self.dataset_load = dataset_ret
        if self.redistribution:
            self.dataset_unconfident = self.dataset_load
        else:
            self.dataset_confident = self.dataset_load

        return datas_count, badcase, ignore, pos_max


    def minibatch(self, batch_size, shuffle=False):
        idxs = list(range(len(self.dataset_load)))
        if shuffle:
            random.shuffle(idxs)

        features = []
        position1s = []
        position2s = []
        typess = []
        labels = []
        att_labels = []
        lengths = []
        patterns = []

        for idx in idxs:
            feature, positions, types, att_label, label, pattern = self.dataset_load[idx]
            features.append(feature)
            lengths.append(len(feature))
            position1s.append(positions[0])
            position2s.append(positions[1])
            typess.append(types)
            labels.append(label)
            att_labels.append(att_label)
            patterns.append(pattern)

            if len(features) % batch_size == 0:
                features = np.array(padding(features))
                position1s = np.array(padding(position1s))
                position2s = np.array(padding(position2s))
                typess = np.array(padding(typess))
                att_labels = np.array(padding(att_labels))
                lengths = np.array(lengths)
                labels = np.array(labels)

                yield patterns, features, position1s, position2s, typess, lengths, att_labels, labels
                features = []
                position1s = []
                position2s = []
                typess = []
                labels = []
                att_labels = []
                lengths = []
                patterns = []

        if len(features) > 0:
            features = np.array(padding(features))
            position1s = np.array(padding(position1s))
            position2s = np.array(padding(position2s))
            typess = np.array(padding(typess))
            att_labels = np.array(padding(att_labels))
            lengths = np.array(lengths)
            labels = np.array(labels)

            yield patterns, features, position1s, position2s, typess, lengths, att_labels, labels


    def minibatch_from_confident(self, batch_size, shuffle=False):
        idxs = list(range(len(self.dataset_confident)))
        if shuffle:
            random.shuffle(idxs)
        features = []
        position1s = []
        position2s = []
        typess = []
        labels = []
        att_labels = []
        lengths = []
        patterns = []

        for idx in idxs:
            feature, positions, types, att_label, label, pattern = self.dataset_load[idx]
            features.append(feature)
            lengths.append(len(feature))
            position1s.append(positions[0])
            position2s.append(positions[1])
            typess.append(types)
            labels.append(label)
            att_labels.append(att_label)
            patterns.append(pattern)

            if len(features) % batch_size == 0:
                features = np.array(padding(features))
                position1s = np.array(padding(position1s))
                position2s = np.array(padding(position2s))
                typess = np.array(padding(typess))
                att_labels = np.array(padding(att_labels))
                lengths = np.array(lengths)
                labels = np.array(labels)

                yield patterns, features, position1s, position2s, typess, lengths, att_labels, labels
                features = []
                position1s = []
                position2s = []
                typess = []
                labels = []
                att_labels = []
                lengths = []
                patterns = []

        if len(features) > 0:
            features = np.array(padding(features))
            position1s = np.array(padding(position1s))
            position2s = np.array(padding(position2s))
            typess = np.array(padding(typess))
            att_labels = np.array(padding(att_labels))
            lengths = np.array(lengths)
            labels = np.array(labels)

            yield patterns, features, position1s, position2s, typess, lengths, att_labels, labels


    def minibatch_from_unconfident(self, batch_size, shuffle=False):
        idxs = list(range(len(self.dataset_unconfident)))
        if shuffle:
            random.shuffle(idxs)
        features = []
        position1s = []
        position2s = []
        typess = []
        labels = []
        att_labels = []
        lengths = []
        patterns = []

        for idx in idxs:
            feature, positions, types, att_label, label, pattern = self.dataset_load[idx]
            features.append(feature)
            lengths.append(len(feature))
            position1s.append(positions[0])
            position2s.append(positions[1])
            typess.append(types)
            labels.append(label)
            att_labels.append(att_label)
            patterns.append(pattern)

            if len(features) % batch_size == 0:
                features = np.array(padding(features))
                position1s = np.array(padding(position1s))
                position2s = np.array(padding(position2s))
                typess = np.array(padding(typess))
                att_labels = np.array(padding(att_labels))
                lengths = np.array(lengths)
                labels = np.array(labels)

                yield patterns, features, position1s, position2s, typess, lengths, att_labels, labels
                features = []
                position1s = []
                position2s = []
                typess = []
                labels = []
                att_labels = []
                lengths = []
                patterns = []

        if len(features) > 0:
            features = np.array(padding(features))
            position1s = np.array(padding(position1s))
            position2s = np.array(padding(position2s))
            typess = np.array(padding(typess))
            att_labels = np.array(padding(att_labels))
            lengths = np.array(lengths)
            labels = np.array(labels)

            yield patterns, features, position1s, position2s, typess, lengths, att_labels, labels


    def update_confidence(self, trustable_pattern):
        dataset_unconfident = []
        for d in self.dataset_unconfident:
            if d[-1] in trustable_pattern.get(d[-2]):
                self.dataset_confident.append(d)
            else:
                dataset_unconfident.append(d)

        self.dataset_unconfident = dataset_unconfident



    def __len__(self):
        return len(self.dataset_confident)


    def load_v2(self, use_small=None):
        # Deprecated
        dataset_ret = []
        pbar = tqdm(total=len(self.dataset) if not use_small else use_small)

        for i, data in enumerate(self.dataset):
            if use_small and i == use_small:
                print('use_samll', i)
                break

            sentText = data['sentText']
            relation = data['relationMentions']
            entitys = data['entityMentions']

            # ??????entity???type????????????
            entity_to_type = {}
            for entity in entitys:
                mentions = '<START>{}<END>'.format(entity['text']).split(' ')
                for mention in mentions:
                    entity_to_type[mention] = entity['label']

            for rel in relation:
                entity1 = rel['em1Text']
                entity2 = rel['em2Text']
                label = conver_token_to_id(rel['label'], self.config['label_dict'])

                # ???????????????entity pair, ??????????????????entity1(.*?)entity2 ??? entity2(.*?)entity1????????????
                # ????????????????????????????????????????????????
                findalls = []
                findall = re.finditer(r'(?:[ ]|^)({0}( .*? ){1})(?:[ ]|$)'.format(entity1, entity2), sentText)
                for find in findall:
                    if find:
                        findalls.append((find, 0))
                findall = re.finditer(r'(?:[ ]|^)({1}( .*? ){0})(?:[ ]|$)'.format(entity1, entity2), sentText)
                for find in findall:
                    if find:
                        findalls.append((find, 1))

                # ????????????????????????????????????????????????
                for find, flag in findalls:
                    start, end = find.span(1)
                    if flag == 0:
                        short = re.sub(r'^{} '.format(entity1), '<START>{}<END> '.format(entity1), find.group(1))
                        short = re.sub(r' {}$'.format(entity2), ' <START>{}<END>'.format(entity2), short)
                    elif flag == 1:
                        short = re.sub(r'^{} '.format(entity2), '<START>{}<END> '.format(entity2), find.group(1))
                        short = re.sub(r' {}$'.format(entity1), ' <START>{}<END>'.format(entity1), short)
                    sentence = sentText[:start] + short + sentText[end:]
                    pattern = find.group(2)
                    sentence = sentence.split(' ')

                    positions = get_positions(sentence, flag, self.config['pos_max'])
                    types = get_types(sentence, entity_to_type, self.config['type_dict'])

                    # ??????????????????id ??? att_label
                    feature = []
                    att_label = [0.] * len(sentence)  # ??????attetion regulation
                    att_flag = 0
                    for i, word in enumerate(sentence):
                        if word[:7] == '<START>':
                            word = word[7:]
                            att_label[i] = 1.
                            att_flag += 1
                        if word[-5:] == '<END>':
                            word = word[:-5]
                            att_label[i] = 1.
                            att_flag += 1
                        if att_flag >=1 and att_flag<4:
                            att_label[i] = 1.
                        wordid = conver_token_to_id(word, self.config['vocab'])
                        feature.append(wordid)
                    fenmu = sum(att_label)
                    att_label = [f / fenmu for f in att_label]

                    dataset_ret.append([feature, positions, types, att_label, label, pattern])
            pbar.update(1)
        pbar.close()

        print('load data: {}'.format(len(dataset_ret)))
















